from __future__ import annotations

import importlib.metadata as md
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

PACKAGES = [
    ("fastapi", "FastAPI web app"),
    ("opencv-python", "OpenCV preprocessing"),
    ("pytesseract", "Python wrapper for Tesseract"),
    ("easyocr", "EasyOCR"),
    ("paddleocr", "PaddleOCR/PaddleOCR-VL"),
    ("paddlex", "PaddleX pipeline extras"),
    ("paddlepaddle", "PaddlePaddle runtime"),
    ("vietocr", "VietOCR"),
    ("torch", "PyTorch"),
    ("transformers", "LayoutLMv3/Transformers"),
    ("onnxruntime", "LayoutLMv3 ONNX runtime"),
    ("einops", "PaddleOCR-VL tensor ops"),
]


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "NOT INSTALLED"


def find_tessdata_dir(tess: str | None) -> str | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("TESSERACT_TESSDATA_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(ROOT / "models" / "tessdata")
    if tess:
        candidates.append(Path(tess).resolve().parent / "tessdata")
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


def main() -> int:
    print("\n=== OCR demo environment check ===")
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    print(f"Project: {ROOT}")

    tess = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    common = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    if not tess:
        for c in common:
            if Path(os.path.expandvars(c)).exists():
                tess = os.path.expandvars(c)
                break
    if tess:
        print(f"Tesseract binary: FOUND -> {tess}")
        tessdata_dir = find_tessdata_dir(tess)
        print(f"Tesseract tessdata: {tessdata_dir or 'NOT FOUND'}")
        try:
            out = subprocess.run([tess, "--version"], text=True, capture_output=True, timeout=10)
            print(out.stdout.splitlines()[0] if out.stdout else "tesseract --version OK")
            lang_cmd = [tess, "--list-langs"]
            if tessdata_dir:
                lang_cmd.extend(["--tessdata-dir", tessdata_dir])
            langs = subprocess.run(lang_cmd, text=True, capture_output=True, timeout=10)
            print("Languages:", ", ".join([x.strip() for x in langs.stdout.splitlines()[1:]]))
        except Exception as exc:
            print(f"Cannot execute tesseract: {exc}")
    else:
        print("Tesseract binary: NOT FOUND")
        print("Fix: Windows chạy scripts/setup_latest_windows.ps1; Linux chạy scripts/setup_latest_linux.sh")

    print("\nPython packages:")
    for pkg, desc in PACKAGES:
        print(f"- {pkg:18s} {version(pkg):15s}  # {desc}")

    print("\nAPI/model env:")
    for key in [
        "TESSERACT_TESSDATA_DIR",
        "GLM_OCR_API_KEY",
        "ZAI_API_KEY",
        "GLM_OCR_ENDPOINT",
        "PADDLEOCR_VL_CMD",
        "LAYOUTLMV3_MODEL_DIR",
        "LAYOUTLMV3_MODEL_NAME",
        "LAYOUTLMV3_PROCESSOR_NAME",
    ]:
        val = os.environ.get(key, "")
        if "KEY" in key and val:
            val = val[:6] + "..." + val[-4:]
        print(f"- {key}={val or '(empty)'}")
    print("==================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
