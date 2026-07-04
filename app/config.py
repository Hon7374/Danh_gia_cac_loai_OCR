from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env_int(name: str, default: int, min_value: int = 1, max_value: int = 32) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))

CACHE_ROOT = ROOT_DIR.parent / "cache"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("PADDLE_HOME", str(CACHE_ROOT / "paddle"))
os.environ.setdefault("PADDLEOCR_HOME", str(CACHE_ROOT / "paddleocr"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE_ROOT / "paddlex"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")

JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

STORAGE_DIR = Path(os.getenv("DOCUMENT_STORAGE_DIR", str(ROOT_DIR.parent / "storage"))).expanduser()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Tesseract binary có thể để trống; engine sẽ tự dò theo PATH/common Windows paths.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "vie+eng").strip() or "vie+eng"
TESSERACT_TESSDATA_DIR = os.getenv("TESSERACT_TESSDATA_DIR", "").strip()

# GLM-OCR/Z.ai: chấp nhận cả tên biến mới ZAI_API_KEY và tên cũ GLM_OCR_API_KEY.
GLM_OCR_API_KEY = (os.getenv("GLM_OCR_API_KEY") or os.getenv("ZAI_API_KEY") or "").strip()
GLM_OCR_ENDPOINT = os.getenv(
    "GLM_OCR_ENDPOINT",
    "https://api.z.ai/api/paas/v4/layout_parsing",
).strip()

# Device selection. "auto" uses GPU when the installed package supports it, otherwise CPU.
OCR_DEVICE = os.getenv("OCR_DEVICE", "auto").strip().lower() or "auto"
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "auto").strip().lower() or "auto"
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "auto").strip().lower() or "auto"

# PaddleOCR-VL: có thể chạy qua command mới chính thức: paddleocr doc_parser -i "{input}" --save_path "{output_dir}"
PADDLEOCR_VL_CMD = os.getenv("PADDLEOCR_VL_CMD", "").strip()
PADDLEOCR_VL_DEVICE = os.getenv("PADDLEOCR_VL_DEVICE", "auto").strip().lower() or "auto"
PADDLEOCR_VL_ENGINE = os.getenv("PADDLEOCR_VL_ENGINE", "").strip()

_DEFAULT_LAYOUTLMV3_MODEL_DIR = ROOT_DIR / "models" / "layoutlmv3-congvan-token-classification"
LAYOUTLMV3_MODEL_DIR = os.getenv(
    "LAYOUTLMV3_MODEL_DIR",
    str(_DEFAULT_LAYOUTLMV3_MODEL_DIR) if _DEFAULT_LAYOUTLMV3_MODEL_DIR.exists() else "",
).strip()
LAYOUTLMV3_MODEL_NAME = os.getenv(
    "LAYOUTLMV3_MODEL_NAME",
    "",
).strip()
LAYOUTLMV3_PROCESSOR_NAME = os.getenv("LAYOUTLMV3_PROCESSOR_NAME", "microsoft/layoutlmv3-base").strip()
LAYOUTLMV3_MAX_WORDS = _env_int("LAYOUTLMV3_MAX_WORDS", 128, 64, 1024)

# Performance knobs. Defaults are tuned for local CPU runs.
OCR_PDF_DPI = int(os.getenv("OCR_PDF_DPI", "180") or "180")
OCR_MAX_IMAGE_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "2200") or "2200")
OCR_OPENCV_WORKERS = _env_int("OCR_OPENCV_WORKERS", 4, 1, 8)
OCR_TESSERACT_WORKERS = _env_int("OCR_TESSERACT_WORKERS", 4, 1, 8)
OCR_GPU_WORKERS = _env_int("OCR_GPU_WORKERS", 1, 1, 2)

# VietOCR refinement recognizes every detected text crop on CPU and is very slow.
# Keep it off for benchmark/full-document runs; set to 1 only for quality experiments.
PADDLE_VIETOCR_REFINE = os.getenv("PADDLE_VIETOCR_REFINE", "0").strip().lower() in {"1", "true", "yes", "on"}

_DEFAULT_VIETOCR_MODEL_DIR = ROOT_DIR / "models" / "vietocr-congvan"
VIETOCR_MODEL_PROFILE = os.getenv("VIETOCR_MODEL_PROFILE", "pretrained").strip().lower() or "pretrained"
_USE_LOCAL_VIETOCR_DEFAULT = VIETOCR_MODEL_PROFILE in {"finetuned", "local", "congvan", "custom"}
VIETOCR_CONFIG_PATH = os.getenv(
    "VIETOCR_CONFIG_PATH",
    str(_DEFAULT_VIETOCR_MODEL_DIR / "config.yml")
    if _USE_LOCAL_VIETOCR_DEFAULT and (_DEFAULT_VIETOCR_MODEL_DIR / "config.yml").exists()
    else "",
).strip()
VIETOCR_WEIGHTS_PATH = os.getenv(
    "VIETOCR_WEIGHTS_PATH",
    str(_DEFAULT_VIETOCR_MODEL_DIR / "transformerocr.pth")
    if _USE_LOCAL_VIETOCR_DEFAULT and (_DEFAULT_VIETOCR_MODEL_DIR / "transformerocr.pth").exists()
    else "",
).strip()
VIETOCR_BATCH_SIZE = _env_int("VIETOCR_BATCH_SIZE", 48, 1, 256)
PADDLE_VIETOCR_REFINE_TIMEOUT_SEC = _env_int("PADDLE_VIETOCR_REFINE_TIMEOUT_SEC", 90, 10, 1800)
PADDLE_VIETOCR_MAX_BOXES = _env_int("PADDLE_VIETOCR_MAX_BOXES", 80, 1, 1000)
