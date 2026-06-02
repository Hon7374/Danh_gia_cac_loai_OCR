from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from app.config import ROOT_DIR
from .base import BaseOCREngine, OCRBox, OCRResult


class EasyOCREngine(BaseOCREngine):
    name = "easyocr"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="easyocr_worker_") as tmp:
                out_json = Path(tmp) / "result.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.ocr_engines.easyocr_worker",
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
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error=f"EasyOCR worker lỗi: {detail}",
                        elapsed_sec=time.perf_counter() - start,
                    )
                if not out_json.exists():
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error="EasyOCR worker không tạo result.json",
                        elapsed_sec=time.perf_counter() - start,
                    )
                data = json.loads(out_json.read_text(encoding="utf-8"))
                boxes = [
                    OCRBox(text=b.get("text", ""), confidence=b.get("confidence"), bbox=b.get("bbox"))
                    for b in data.get("boxes") or []
                    if isinstance(b, dict)
                ]
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status=data.get("status") or "ok",
                    text=data.get("text") or "",
                    boxes=boxes,
                    elapsed_sec=time.perf_counter() - start,
                    error=data.get("error") or "",
                    raw=data.get("raw"),
                )
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error="EasyOCR quá thời gian 30 phút.",
                elapsed_sec=time.perf_counter() - start,
            )
        except ModuleNotFoundError:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="skipped",
                error="Chưa cài easyocr. Chạy: pip install -r requirements-optional-ocr.txt",
                elapsed_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"EasyOCR lỗi: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )
