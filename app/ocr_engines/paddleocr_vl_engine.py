from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from app.config import PADDLEOCR_VL_CMD, PADDLEOCR_VL_DEVICE, PADDLEOCR_VL_ENGINE, ROOT_DIR
from app.services.devices import resolve_paddle_device
from .base import BaseOCREngine, OCRBox, OCRResult
from .paddle_runtime import install_modelscope_stub

_PADDLEOCR_VL = None


def _extract_text(obj: Any) -> str:
    chunks: list[str] = []
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ["markdown", "text", "content", "result", "html", "recognized_text", "layout_text", "rec_text"]:
            if key in obj:
                s = _extract_text(obj[key])
                if s:
                    chunks.append(s)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                s = _extract_text(v)
                if s:
                    chunks.append(s)
    if isinstance(obj, list):
        for v in obj:
            s = _extract_text(v)
            if s:
                chunks.append(s)
    return "\n".join(dict.fromkeys(chunks)).strip()


def _read_generated_output(output_dir: Path) -> tuple[str, dict[str, Any]]:
    raw: dict[str, Any] = {"output_dir": str(output_dir), "files": []}
    chunks: list[str] = []
    if not output_dir.exists():
        return "", raw
    for p in output_dir.rglob("*"):
        if not p.is_file():
            continue
        raw["files"].append(str(p))
        if p.suffix.lower() in {".md", ".markdown", ".txt"}:
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        elif p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                raw[str(p)] = obj
                s = _extract_text(obj)
                if s:
                    chunks.append(s)
            except Exception:
                pass
    return "\n".join([c.strip() for c in chunks if c.strip()]).strip(), raw


def _extract_layout_boxes(obj: Any) -> list[OCRBox]:
    boxes: list[OCRBox] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            for match in re.finditer(
                r"label:\s*(?P<label>[^\n]+)\s*\nbbox:\s*\[(?P<bbox>[^\]]+)\]\s*\ncontent:\s*(?P<content>.*?)(?=\n#################|\Z)",
                value,
                flags=re.DOTALL,
            ):
                try:
                    bbox = [int(float(part.strip())) for part in match.group("bbox").split(",")[:4]]
                except ValueError:
                    continue
                content = match.group("content").strip()
                content = re.sub(r"\n?#################\s*$", "", content).strip()
                if content:
                    boxes.append(OCRBox(text=content, bbox=bbox, label=match.group("label").strip()))
            return
        if isinstance(value, dict):
            text = value.get("content") or value.get("text") or value.get("rec_text")
            raw_bbox = value.get("bbox") or value.get("coordinate")
            if isinstance(text, str) and isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
                try:
                    bbox = [int(float(part)) for part in raw_bbox[:4]]
                    boxes.append(OCRBox(text=text.strip(), bbox=bbox, label=str(value.get("label") or "") or None))
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                if isinstance(child, (dict, list, str)):
                    visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)

    visit(obj)
    unique: list[OCRBox] = []
    seen = set()
    for box in boxes:
        key = (box.text, tuple(box.bbox or []), box.label)
        if key not in seen:
            unique.append(box)
            seen.add(key)
    return unique


class PaddleOCRVLEngine(BaseOCREngine):
    name = "paddleocr_vl"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="paddleocr_vl_worker_") as tmp:
                out_json = Path(tmp) / "result.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.ocr_engines.paddleocr_vl_worker",
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
                        error=f"PaddleOCR-VL worker lỗi: {detail}",
                        elapsed_sec=elapsed,
                    )
                if not out_json.exists():
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error="PaddleOCR-VL worker không tạo result.json",
                        elapsed_sec=elapsed,
                    )
                data = json.loads(out_json.read_text(encoding="utf-8"))
                return OCRResult(
                    engine=data.get("engine") or self.name,
                    variant=data.get("variant") or variant,
                    status=data.get("status") or "error",
                    text=data.get("text") or "",
                    boxes=[
                        OCRBox(
                            text=box.get("text", ""),
                            confidence=box.get("confidence"),
                            bbox=box.get("bbox"),
                            label=box.get("label"),
                        )
                        for box in (data.get("boxes") or [])
                        if isinstance(box, dict)
                    ],
                    elapsed_sec=elapsed,
                    error=data.get("error") or "",
                    raw=data.get("raw"),
                )
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error="PaddleOCR-VL quá thời gian 30 phút.",
                elapsed_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"PaddleOCR-VL lỗi: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )


def run_paddleocr_vl_direct(image_path: Path, variant: str = "preprocessed") -> OCRResult:
    start = time.perf_counter()

    # 1) Cách bền nhất: cấu hình command. Hỗ trợ placeholder {input}, {output_dir}.
    if PADDLEOCR_VL_CMD:
        try:
            with tempfile.TemporaryDirectory(prefix="paddleocr_vl_") as tmp:
                out_dir = Path(tmp) / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                device = resolve_paddle_device(PADDLEOCR_VL_DEVICE)
                cmd = (
                    PADDLEOCR_VL_CMD
                    .replace("{input}", str(image_path))
                    .replace("{output_dir}", str(out_dir))
                    .replace("{device}", device)
                )
                completed = subprocess.run(
                    shlex.split(cmd),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                stdout = completed.stdout.strip()
                stderr = completed.stderr.strip()
                file_text, raw_files = _read_generated_output(out_dir)
                if completed.returncode != 0:
                    return OCRResult(
                        engine="paddleocr_vl",
                        variant=variant,
                        status="error",
                        error=f"PADDLEOCR_VL_CMD trả mã {completed.returncode}: {stderr or stdout}",
                        elapsed_sec=time.perf_counter() - start,
                    )
                text = file_text or stdout
                raw_payload = {"stdout": stdout, "stderr": stderr, "generated": raw_files, "device": device, "worker": "subprocess"}
                return OCRResult(
                    engine="paddleocr_vl",
                    variant=variant,
                    status="ok" if text else "skipped",
                    text=text,
                    boxes=_extract_layout_boxes(raw_payload),
                    elapsed_sec=time.perf_counter() - start,
                    raw=raw_payload,
                )
        except Exception as exc:
            return OCRResult(
                engine="paddleocr_vl",
                variant=variant,
                status="error",
                error=f"Lỗi khi chạy PADDLEOCR_VL_CMD: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )

    # 2) API Python chính thức mới: from paddleocr import PaddleOCRVL.
    try:
        global _PADDLEOCR_VL
        install_modelscope_stub()
        from paddleocr import PaddleOCRVL  # type: ignore

        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
        }
        device = resolve_paddle_device(PADDLEOCR_VL_DEVICE)
        if device:
            kwargs["device"] = device
        if PADDLEOCR_VL_ENGINE:
            kwargs["vl_rec_backend"] = PADDLEOCR_VL_ENGINE
        if _PADDLEOCR_VL is None:
            _PADDLEOCR_VL = PaddleOCRVL(**kwargs)
        pipeline = _PADDLEOCR_VL
        output = pipeline.predict(str(image_path))
        pages = list(output)
        chunks: list[str] = []
        raw: list[Any] = []
        with tempfile.TemporaryDirectory(prefix="paddleocr_vl_api_") as tmp:
            out_dir = Path(tmp) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            for res in pages:
                raw.append(str(res))
                for method in ["save_to_markdown", "save_to_json"]:
                    if hasattr(res, method):
                        try:
                            getattr(res, method)(save_path=out_dir)
                        except Exception:
                            pass
            file_text, raw_files = _read_generated_output(out_dir)
        if file_text:
            chunks.append(file_text)
        else:
            chunks.extend([str(x) for x in raw if str(x).strip()])
        text = "\n".join(chunks).strip()
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="ok" if text else "skipped",
            text=text,
            boxes=_extract_layout_boxes({"pages": raw, "generated": raw_files}),
            elapsed_sec=time.perf_counter() - start,
            raw={"api": "PaddleOCRVL", "pages": raw, "device": device, "worker": "subprocess"},
        )
    except ModuleNotFoundError:
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="skipped",
            error="Chưa cài PaddleOCR-VL/PaddleOCR 3.x. Chạy scripts/setup_latest_windows.ps1 -Full hoặc cấu hình PADDLEOCR_VL_CMD.",
            elapsed_sec=time.perf_counter() - start,
        )
    except Exception as exc:
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="error",
            error=f"PaddleOCR-VL lỗi: {exc}",
            elapsed_sec=time.perf_counter() - start,
        )
