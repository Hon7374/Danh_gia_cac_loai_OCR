from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from app.config import CACHE_ROOT, OCR_DEVICE, TORCH_DEVICE
from app.ocr_engines.base import OCRBox, OCRResult


def _read_image_unicode(image_path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    return image


def _torch_gpu_enabled() -> tuple[bool, str]:
    requested = (TORCH_DEVICE if TORCH_DEVICE != "auto" else OCR_DEVICE).strip().lower()
    if requested in {"cpu", "none", "off", "false", "0"}:
        return False, "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return True, "cuda:0"
    except Exception:
        pass
    return False, "cpu"


def run(image_path: Path, variant: str) -> OCRResult:
    start = time.perf_counter()
    try:
        import easyocr

        gpu, device = _torch_gpu_enabled()
        easyocr_root = CACHE_ROOT / "easyocr"
        model_dir = easyocr_root / "model"
        user_network_dir = easyocr_root / "user_network"
        model_dir.mkdir(parents=True, exist_ok=True)
        user_network_dir.mkdir(parents=True, exist_ok=True)
        reader = easyocr.Reader(
            ["vi", "en"],
            gpu=gpu,
            model_storage_directory=str(model_dir),
            user_network_directory=str(user_network_dir),
        )
        image = _read_image_unicode(image_path)
        results = reader.readtext(image, detail=1, paragraph=False)
        boxes: list[OCRBox] = []
        texts: list[str] = []
        for item in results:
            poly, text, conf = item
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            text = str(text).strip()
            if text:
                boxes.append(OCRBox(text=text, confidence=float(conf), bbox=bbox))
                texts.append(text)
        return OCRResult(
            engine="easyocr",
            variant=variant,
            status="ok",
            text="\n".join(texts),
            boxes=boxes,
            elapsed_sec=time.perf_counter() - start,
            raw={
                "device": device,
                "gpu": gpu,
                "worker": "subprocess",
                "model_storage_directory": str(model_dir),
            },
        )
    except ModuleNotFoundError:
        return OCRResult(
            engine="easyocr",
            variant=variant,
            status="skipped",
            error="Chưa cài easyocr. Chạy: pip install -r requirements-optional-ocr.txt",
            elapsed_sec=time.perf_counter() - start,
        )
    except Exception as exc:
        return OCRResult(
            engine="easyocr",
            variant=variant,
            status="error",
            error=f"EasyOCR lỗi: {exc}\n{traceback.format_exc()}",
            elapsed_sec=time.perf_counter() - start,
        )


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python -m app.ocr_engines.easyocr_worker <image_path> <variant> <out_json>", file=sys.stderr)
        return 2
    result = run(Path(sys.argv[1]), sys.argv[2])
    out_json = Path(sys.argv[3])
    out_json.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
