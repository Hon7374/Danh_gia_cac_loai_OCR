from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import (
    PADDLE_VIETOCR_REFINE,
    ROOT_DIR,
    VIETOCR_BATCH_SIZE,
    VIETOCR_CONFIG_PATH,
    VIETOCR_WEIGHTS_PATH,
)
from app.services.devices import resolve_paddle_device, resolve_torch_device
from .base import BaseOCREngine, OCRBox, OCRResult
from .paddle_runtime import install_modelscope_stub

_PADDLE = None
_VIETOCR = None
_VIETOCR_MODEL_INFO: dict[str, Any] = {}


def _bbox_from_poly(poly: Any) -> list[int] | None:
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    try:
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            return [int(v) for v in poly]
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def _normalize_paddle_result(result: Any) -> list[OCRBox]:
    boxes: list[OCRBox] = []
    if result is None:
        return boxes

    if isinstance(result, list):
        for page in result:
            if not isinstance(page, dict) or "rec_texts" not in page:
                continue
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or []
            for idx, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue
                conf = scores[idx] if idx < len(scores) else None
                bbox = _bbox_from_poly(polys[idx] if idx < len(polys) else None)
                boxes.append(OCRBox(text=text, confidence=float(conf) if conf is not None else None, bbox=bbox))
        if boxes:
            return boxes

    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]
    for item in candidates or []:
        try:
            poly = item[0]
            rec = item[1]
            text = str(rec[0]).strip()
            conf = float(rec[1]) if len(rec) > 1 else None
            bbox = _bbox_from_poly(poly)
            if text:
                boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox))
        except Exception:
            continue
    return boxes


def _load_vietocr_predictor():
    global _VIETOCR, _VIETOCR_MODEL_INFO
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except Exception:
        return None

    try:
        if _VIETOCR is None:
            config_path = Path(VIETOCR_CONFIG_PATH).expanduser() if VIETOCR_CONFIG_PATH else None
            weights_path = Path(VIETOCR_WEIGHTS_PATH).expanduser() if VIETOCR_WEIGHTS_PATH else None
            if config_path and config_path.exists():
                config = Cfg.load_config_from_file(str(config_path))
                config_source = str(config_path)
            else:
                config = Cfg.load_config_from_name("vgg_transformer")
                config_source = "vietocr:vgg_transformer"
            if weights_path and weights_path.exists():
                config["weights"] = str(weights_path)
            config["device"] = resolve_torch_device()
            config["predictor"]["beamsearch"] = False
            _VIETOCR_MODEL_INFO = {
                "config": config_source,
                "weights": config.get("weights"),
                "device": config["device"],
                "batch_size": VIETOCR_BATCH_SIZE,
            }
            _VIETOCR = Predictor(config)
        return _VIETOCR
    except Exception:
        _VIETOCR = None
        return None


def _try_vietocr_recognize_local(image_path: Path, boxes: list[OCRBox]) -> list[OCRBox] | None:
    predictor = _load_vietocr_predictor()
    if predictor is None:
        return None

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            new_boxes = list(boxes)
            crops: list[Image.Image] = []
            crop_indices: list[int] = []
            for box_index, b in enumerate(boxes):
                if not b.bbox:
                    continue
                x0, y0, x1, y1 = b.bbox
                pad = 2
                crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
                if crop.width <= 1 or crop.height <= 1:
                    continue
                crops.append(crop.copy())
                crop_indices.append(box_index)
            if not crops:
                return new_boxes

            predictions: list[str] = []
            batch_size = max(1, int(VIETOCR_BATCH_SIZE or 1))
            for start in range(0, len(crops), batch_size):
                batch = crops[start:start + batch_size]
                if hasattr(predictor, "predict_batch"):
                    predictions.extend([str(x).strip() for x in predictor.predict_batch(batch)])
                else:
                    predictions.extend([str(predictor.predict(crop)).strip() for crop in batch])

            for idx, text in zip(crop_indices, predictions):
                original = new_boxes[idx]
                new_boxes[idx] = OCRBox(text=text or original.text, confidence=original.confidence, bbox=original.bbox)
            return new_boxes
    except Exception:
        return None


def _try_vietocr_recognize(image_path: Path, boxes: list[OCRBox]) -> tuple[list[OCRBox] | None, dict[str, Any], str]:
    """Run VietOCR in a fresh process to avoid Paddle/Torch CUDA DLL conflicts on Windows."""
    global _VIETOCR_MODEL_INFO
    if not boxes:
        return boxes, {}, ""
    try:
        with tempfile.TemporaryDirectory(prefix="vietocr_refine_worker_") as tmp:
            tmp_dir = Path(tmp)
            boxes_json = tmp_dir / "boxes.json"
            out_json = tmp_dir / "refined.json"
            boxes_json.write_text(
                json.dumps(
                    [
                        {"text": b.text, "confidence": b.confidence, "bbox": b.bbox, "label": b.label}
                        for b in boxes
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.ocr_engines.vietocr_refine_worker",
                    str(image_path),
                    str(boxes_json),
                    str(out_json),
                ],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                return None, {}, detail or "VietOCR refine worker failed"
            if not out_json.exists():
                return None, {}, "VietOCR refine worker did not create output JSON"
            data = json.loads(out_json.read_text(encoding="utf-8"))
            if data.get("status") != "ok":
                return None, data.get("model_info") or {}, data.get("error") or "VietOCR refine worker error"
            refined = [
                OCRBox(text=b.get("text", ""), confidence=b.get("confidence"), bbox=b.get("bbox"), label=b.get("label"))
                for b in data.get("boxes") or []
                if isinstance(b, dict)
            ]
            model_info = data.get("model_info") or {}
            _VIETOCR_MODEL_INFO = model_info
            return refined, model_info, ""
    except subprocess.TimeoutExpired:
        return None, {}, "VietOCR refine quá thời gian 15 phút."
    except Exception as exc:
        return None, {}, str(exc)


def run_paddle_vietocr_direct(image_path: Path, variant: str = "preprocessed") -> OCRResult:
    start = time.perf_counter()
    global _PADDLE
    try:
        install_modelscope_stub()
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="skipped",
            error="Chưa cài paddleocr/paddlepaddle. Chạy: pip install -r requirements-optional-ocr.txt",
            elapsed_sec=time.perf_counter() - start,
        )
    try:
        paddle_device = resolve_paddle_device()
        if _PADDLE is None:
            try:
                _PADDLE = PaddleOCR(
                    lang="vi",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=paddle_device,
                )
            except (TypeError, ValueError):
                _PADDLE = PaddleOCR(lang="vi")

        try:
            if hasattr(_PADDLE, "predict"):
                raw = _PADDLE.predict(str(image_path))
            else:
                raw = _PADDLE.ocr(str(image_path), cls=True)
        except TypeError:
            raw = _PADDLE.ocr(str(image_path))

        boxes = _normalize_paddle_result(raw)
        refine_note = "VietOCR refine disabled"
        vietocr_model_info: dict[str, Any] | None = None
        if PADDLE_VIETOCR_REFINE:
            viet_boxes, vietocr_model_info, refine_error = _try_vietocr_recognize(image_path, boxes)
            if viet_boxes:
                boxes = viet_boxes
                refine_note = "VietOCR refined each crop"
            else:
                refine_note = f"VietOCR refine requested but unavailable: {refine_error}".strip()
        text = "\n".join([b.text for b in boxes])
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="ok",
            text=text,
            boxes=boxes,
            elapsed_sec=time.perf_counter() - start,
            raw={
                "note": (
                    "PaddleOCR detects text regions, then fine-tuned VietOCR recognizes each crop."
                    if PADDLE_VIETOCR_REFINE
                    else "PaddleOCR detect/recognize. VietOCR refinement is off by default because it is very slow on CPU."
                ),
                "refine": refine_note,
                "paddle_device": paddle_device,
                "vietocr_device": resolve_torch_device() if PADDLE_VIETOCR_REFINE else None,
                "vietocr_model": vietocr_model_info if PADDLE_VIETOCR_REFINE else None,
                "worker": "subprocess",
            },
        )
    except Exception as exc:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="error",
            error=f"PaddleOCR + VietOCR lỗi: {exc}\n{traceback.format_exc()}",
            elapsed_sec=time.perf_counter() - start,
        )


def _result_from_dict(data: dict[str, Any], fallback_variant: str, elapsed: float) -> OCRResult:
    boxes = [
        OCRBox(text=b.get("text", ""), confidence=b.get("confidence"), bbox=b.get("bbox"))
        for b in data.get("boxes") or []
        if isinstance(b, dict)
    ]
    return OCRResult(
        engine=data.get("engine") or "paddle_vietocr",
        variant=data.get("variant") or fallback_variant,
        status=data.get("status") or "error",
        text=data.get("text") or "",
        boxes=boxes,
        elapsed_sec=elapsed,
        error=data.get("error") or "",
        raw=data.get("raw"),
    )


class PaddleVietOCREngine(BaseOCREngine):
    name = "paddle_vietocr"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="paddle_vietocr_worker_") as tmp:
                out_json = Path(tmp) / "result.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.ocr_engines.paddle_vietocr_worker",
                        str(image_path),
                        variant,
                        str(out_json),
                    ],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                elapsed = time.perf_counter() - start
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error=f"PaddleOCR worker lỗi: {detail}",
                        elapsed_sec=elapsed,
                    )
                if not out_json.exists():
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error="PaddleOCR worker không tạo result.json",
                        elapsed_sec=elapsed,
                    )
                return _result_from_dict(json.loads(out_json.read_text(encoding="utf-8")), variant, elapsed)
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error="PaddleOCR quá thời gian 30 phút.",
                elapsed_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"PaddleOCR + VietOCR lỗi: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )
