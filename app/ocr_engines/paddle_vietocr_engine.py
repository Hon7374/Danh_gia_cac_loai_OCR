from __future__ import annotations

import json
import math
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
    PADDLE_VIETOCR_MAX_BOXES,
    PADDLE_VIETOCR_REFINE_TIMEOUT_SEC,
    ROOT_DIR,
    VIETOCR_BATCH_SIZE,
    VIETOCR_CONFIG_PATH,
    VIETOCR_MODEL_PROFILE,
    VIETOCR_WEIGHTS_PATH,
)
from app.services.devices import resolve_paddle_device, resolve_torch_device
from .base import BaseOCREngine, OCRBox, OCRResult
from .paddle_runtime import install_modelscope_stub

_PADDLE = None
_VIETOCR = None
_VIETOCR_MODEL_INFO: dict[str, Any] = {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        vals = [_safe_float(item) for item in value]
        nums = [item for item in vals if item is not None and math.isfinite(item)]
        return sum(nums) / len(nums) if nums else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _normalize_polygon(poly: Any) -> list[list[int]] | None:
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    try:
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            return None
        points: list[list[int]] = []
        for point in poly[:4]:
            if hasattr(point, "tolist"):
                point = point.tolist()
            if len(point) < 2:
                return None
            points.append([int(round(float(point[0]))), int(round(float(point[1])))])
        return points if len(points) == 4 else None
    except Exception:
        return None


def _bbox_from_poly(poly: Any) -> list[int] | None:
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    try:
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            x0, y0, x1, y1 = [int(round(float(v))) for v in poly]
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    except Exception:
        return None
    polygon = _normalize_polygon(poly)
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _make_box(text: Any, confidence: Any, poly: Any) -> OCRBox | None:
    clean_text = str(text or "").strip()
    if not clean_text:
        return None
    return OCRBox(
        text=clean_text,
        confidence=_safe_float(confidence),
        bbox=_bbox_from_poly(poly),
        polygon=_normalize_polygon(poly),
    )


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
                conf = scores[idx] if idx < len(scores) else None
                box = _make_box(text, conf, polys[idx] if idx < len(polys) else None)
                if box:
                    boxes.append(box)
        if boxes:
            return boxes

    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]
    for item in candidates or []:
        try:
            poly = item[0]
            rec = item[1]
            box = _make_box(rec[0], rec[1] if len(rec) > 1 else None, poly)
            if box:
                boxes.append(box)
        except Exception:
            continue
    return boxes


def _crop_by_bbox(image: Image.Image, bbox: list[int], pad: int = 2) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    crop = image.crop(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(image.width, x1 + pad),
            min(image.height, y1 + pad),
        )
    )
    if crop.width <= 1 or crop.height <= 1:
        return None
    return crop.convert("RGB")


def _crop_by_polygon(image: Image.Image, polygon: list[list[int]]) -> Image.Image | None:
    try:
        import cv2
        import numpy as np

        pts = np.array(polygon, dtype=np.float32)
        crop_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        crop_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if crop_width <= 1 or crop_height <= 1:
            return None

        dst = np.array(
            [[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            np.array(image),
            matrix,
            (crop_width, crop_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped.shape[0] / max(1, warped.shape[1]) >= 1.5:
            warped = np.rot90(warped)
        return Image.fromarray(warped).convert("RGB")
    except Exception:
        return None


def _crop_text_region(image: Image.Image, box: OCRBox) -> Image.Image | None:
    if box.polygon:
        crop = _crop_by_polygon(image, box.polygon)
        if crop is not None:
            return crop
    if box.bbox:
        return _crop_by_bbox(image, box.bbox)
    return None


def _predict_vietocr_batch(predictor: Any, batch: list[Image.Image]) -> list[tuple[str, float | None]]:
    if hasattr(predictor, "predict_batch"):
        try:
            texts, probs = predictor.predict_batch(batch, return_prob=True)
            return [(str(text).strip(), _safe_float(prob)) for text, prob in zip(texts or [], probs or [])]
        except TypeError:
            texts = predictor.predict_batch(batch)
            return [(str(text).strip(), None) for text in texts or []]

    predictions: list[tuple[str, float | None]] = []
    for crop in batch:
        try:
            text, prob = predictor.predict(crop, return_prob=True)
            predictions.append((str(text).strip(), _safe_float(prob)))
        except TypeError:
            predictions.append((str(predictor.predict(crop)).strip(), None))
    return predictions


def _load_vietocr_predictor():
    global _VIETOCR, _VIETOCR_MODEL_INFO
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except Exception:
        return None

    try:
        if _VIETOCR is None:
            profile = (VIETOCR_MODEL_PROFILE or "pretrained").strip().lower()
            use_pretrained = profile in {"pretrained", "official", "vgg_transformer", "default"}
            config_path = Path(VIETOCR_CONFIG_PATH).expanduser() if VIETOCR_CONFIG_PATH and not use_pretrained else None
            weights_path = Path(VIETOCR_WEIGHTS_PATH).expanduser() if VIETOCR_WEIGHTS_PATH and not use_pretrained else None
            if not use_pretrained and config_path and config_path.exists():
                config = Cfg.load_config_from_file(str(config_path))
                config_source = str(config_path)
            else:
                config = Cfg.load_config_from_name("vgg_transformer")
                config_source = "vietocr:vgg_transformer"
            if not use_pretrained and weights_path and weights_path.exists():
                config["weights"] = str(weights_path)
            config["device"] = resolve_torch_device()
            config["predictor"]["beamsearch"] = False
            _VIETOCR_MODEL_INFO = {
                "profile": profile,
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
                crop = _crop_text_region(image, b)
                if crop is None:
                    continue
                crops.append(crop.copy())
                crop_indices.append(box_index)
            if not crops:
                return new_boxes

            predictions: list[tuple[str, float | None]] = []
            batch_size = max(1, int(VIETOCR_BATCH_SIZE or 1))
            for start in range(0, len(crops), batch_size):
                batch = crops[start:start + batch_size]
                predictions.extend(_predict_vietocr_batch(predictor, batch))

            for idx, (text, confidence) in zip(crop_indices, predictions):
                original = new_boxes[idx]
                new_boxes[idx] = OCRBox(
                    text=text or original.text,
                    confidence=confidence if confidence is not None else original.confidence,
                    bbox=original.bbox,
                    label=original.label,
                    polygon=original.polygon,
                )
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
                        {
                            "text": b.text,
                            "confidence": b.confidence,
                            "bbox": b.bbox,
                            "label": b.label,
                            "polygon": b.polygon,
                        }
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
                timeout=PADDLE_VIETOCR_REFINE_TIMEOUT_SEC,
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
                OCRBox(
                    text=b.get("text", ""),
                    confidence=b.get("confidence"),
                    bbox=b.get("bbox"),
                    label=b.get("label"),
                    polygon=b.get("polygon"),
                )
                for b in data.get("boxes") or []
                if isinstance(b, dict)
            ]
            model_info = data.get("model_info") or {}
            _VIETOCR_MODEL_INFO = model_info
            return refined, model_info, ""
    except subprocess.TimeoutExpired:
        return None, {}, f"VietOCR refine qua thoi gian {PADDLE_VIETOCR_REFINE_TIMEOUT_SEC}s."
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
    except Exception as exc:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="error",
            error=f"PaddleOCR import failed: {exc}",
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
            if len(boxes) > PADDLE_VIETOCR_MAX_BOXES:
                refine_note = (
                    f"VietOCR refine skipped: {len(boxes)} text boxes exceeds "
                    f"PADDLE_VIETOCR_MAX_BOXES={PADDLE_VIETOCR_MAX_BOXES}"
                )
            else:
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
                    "PaddleOCR detects text regions, then VietOCR recognizes each crop."
                    if PADDLE_VIETOCR_REFINE
                    else "PaddleOCR detect/recognize. VietOCR refinement is off by default because it is very slow on CPU."
                ),
                "refine": refine_note,
                "crop_mode": "paddle_polygon_rotated_crop_then_vietocr_batch",
                "integration_reference": "longvu358/VietnameseOCR and bmd1905/vietnamese-ocr; adapted to installed PaddleOCR/VietOCR APIs.",
                "paddle_device": paddle_device,
                "vietocr_device": resolve_torch_device() if PADDLE_VIETOCR_REFINE else None,
                "vietocr_model": vietocr_model_info if PADDLE_VIETOCR_REFINE else None,
                "vietocr_refine_timeout_sec": PADDLE_VIETOCR_REFINE_TIMEOUT_SEC if PADDLE_VIETOCR_REFINE else None,
                "vietocr_max_boxes": PADDLE_VIETOCR_MAX_BOXES if PADDLE_VIETOCR_REFINE else None,
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
        OCRBox(
            text=b.get("text", ""),
            confidence=b.get("confidence"),
            bbox=b.get("bbox"),
            label=b.get("label"),
            polygon=b.get("polygon"),
        )
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
