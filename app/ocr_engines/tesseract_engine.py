from __future__ import annotations

import shutil
import time
import os
from pathlib import Path

from PIL import Image

from app.config import ROOT_DIR, TESSERACT_CMD, TESSERACT_LANG, TESSERACT_TESSDATA_DIR
from .base import BaseOCREngine, OCRBox, OCRResult


def _find_tesseract_binary() -> str | None:
    """Tự dò Tesseract binary để người dùng không phải sửa .env ngay từ đầu."""
    candidates: list[str] = []
    if TESSERACT_CMD:
        candidates.append(TESSERACT_CMD)
    path_hit = shutil.which("tesseract")
    if path_hit:
        candidates.append(path_hit)
    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\%USERNAME%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
    )
    for item in candidates:
        expanded = os.path.expandvars(item)
        if expanded and Path(expanded).exists():
            return expanded
    return None


def _find_tessdata_dir(binary: str | None) -> str | None:
    candidates: list[Path] = []
    if TESSERACT_TESSDATA_DIR:
        candidates.append(Path(TESSERACT_TESSDATA_DIR))
    candidates.append(ROOT_DIR / "models" / "tessdata")
    if binary:
        candidates.append(Path(binary).resolve().parent / "tessdata")
    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/local/share/tessdata"),
        ]
    )
    for item in candidates:
        expanded = Path(os.path.expandvars(str(item))).expanduser()
        if expanded.exists() and expanded.is_dir():
            return str(expanded)
    return None


class TesseractEngine(BaseOCREngine):
    name = "tesseract"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            import pytesseract
            from pytesseract import Output

            binary = _find_tesseract_binary()
            if not binary:
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status="skipped",
                    error=(
                        "Chưa tìm thấy Tesseract binary. Chạy scripts/setup_latest_windows.ps1 "
                        "hoặc cài UB Mannheim Tesseract rồi mở lại app. Nếu đã cài, đặt "
                        "TESSERACT_CMD trong .env."
                    ),
                    elapsed_sec=time.perf_counter() - start,
                )
            pytesseract.pytesseract.tesseract_cmd = binary
            tessdata_dir = _find_tessdata_dir(binary)
            config = f"--tessdata-dir {Path(tessdata_dir).as_posix()}" if tessdata_dir else ""

            image = Image.open(image_path).convert("RGB")
            try:
                data = pytesseract.image_to_data(image, lang=TESSERACT_LANG, output_type=Output.DICT, config=config)
            except Exception as exc:
                # Nếu máy chưa có vie.traineddata, fallback eng để demo không chết.
                if "vie" in TESSERACT_LANG:
                    data = pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT, config=config)
                else:
                    raise exc

            boxes: list[OCRBox] = []
            words: list[str] = []
            n = len(data.get("text", []))
            for i in range(n):
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                try:
                    conf = float(data.get("conf", [None] * n)[i])
                    if conf < 0:
                        conf = None
                except Exception:
                    conf = None
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                boxes.append(OCRBox(text=text, confidence=conf, bbox=[int(x), int(y), int(x + w), int(y + h)]))
                words.append(text)
            text = " ".join(words).strip()
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="ok",
                text=text,
                boxes=boxes,
                elapsed_sec=time.perf_counter() - start,
                raw={"tesseract_binary": binary, "tessdata_dir": tessdata_dir, "lang_requested": TESSERACT_LANG},
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"Tesseract lỗi hoặc chưa cài đủ language data. Chi tiết: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )
