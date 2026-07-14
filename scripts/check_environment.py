from __future__ import annotations

import hashlib
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
    (("fastapi",), "FastAPI web app"),
    (("opencv-python-headless", "opencv-python"), "OpenCV preprocessing"),
    (("pytesseract",), "Python wrapper for Tesseract"),
    (("easyocr",), "EasyOCR"),
    (("paddleocr",), "PaddleOCR/PaddleOCR-VL"),
    (("paddlex",), "PaddleX pipeline extras"),
    (("paddlepaddle-gpu", "paddlepaddle"), "PaddlePaddle runtime"),
    (("vietocr",), "VietOCR"),
    (("torch",), "PyTorch"),
    (("transformers",), "LayoutLMv3/Transformers"),
    (("onnxruntime-gpu", "onnxruntime"), "LayoutLMv3 ONNX runtime"),
    (("einops",), "PaddleOCR-VL tensor ops"),
    (("matplotlib",), "Benchmark report charts"),
    (("python-docx",), "Editable Word/report export"),
]


def version(packages: tuple[str, ...]) -> str:
    for pkg in packages:
        try:
            return f"{md.version(pkg)} ({pkg})"
        except md.PackageNotFoundError:
            continue
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
    for packages, desc in PACKAGES:
        label = "/".join(packages)
        print(f"- {label:32s} {version(packages):34s}  # {desc}")

    print("\nAPI/model env:")
    for key in [
        "TESSERACT_TESSDATA_DIR",
        "OCR_TEMP_DIR",
        "PADDLE_VIETOCR_REFINE",
        "VIETOCR_MODEL_PROFILE",
        "VIETOCR_CONFIG_PATH",
        "VIETOCR_WEIGHTS_PATH",
        "PADDLEOCR_VL_CMD",
        "LAYOUTLMV3_MODEL_DIR",
        "LAYOUTLMV3_MODEL_NAME",
        "LAYOUTLMV3_PROCESSOR_NAME",
    ]:
        val = os.environ.get(key, "")
        if "KEY" in key and val:
            val = val[:6] + "..." + val[-4:]
        print(f"- {key}={val or '(empty)'}")
    weights_value = os.environ.get("VIETOCR_WEIGHTS_PATH", "").strip()
    if weights_value:
        weights_path = Path(weights_value).expanduser()
        if not weights_path.is_absolute():
            weights_path = ROOT / weights_path
        weights_path = weights_path.resolve()
        if weights_path.exists() and weights_path.is_file():
            digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            print(f"- VietOCR weights: FOUND ({weights_path.stat().st_size} bytes, sha256={digest})")
        else:
            print(f"- VietOCR weights: NOT FOUND -> {weights_path}")
    print("==================================\n")
    print("Note: this command checks installation metadata and the Tesseract executable;")
    print("model inference is verified by running an OCR job through the application.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
