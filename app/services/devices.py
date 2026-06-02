from __future__ import annotations

from functools import lru_cache
import subprocess
import sys

from app.config import OCR_DEVICE, PADDLE_DEVICE, TORCH_DEVICE


CPU_VALUES = {"cpu", "none", "off", "false", "0"}
AUTO_VALUES = {"", "auto"}
GPU_VALUES = {"gpu", "cuda", "cuda:0", "gpu:0", "true", "1", "on", "yes"}


def _normalized(value: str | None, fallback: str = "auto") -> str:
    return (value or fallback).strip().lower()


def _prefers_gpu(value: str) -> bool:
    return value in GPU_VALUES or value.startswith("cuda") or value.startswith("gpu")


@lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    code = "import torch; print('1' if torch.cuda.is_available() else '0')"
    return _probe_bool(code)


@lru_cache(maxsize=1)
def paddle_cuda_available() -> bool:
    code = "import paddle; print('1' if paddle.is_compiled_with_cuda() else '0')"
    return _probe_bool(code)


def _probe_bool(code: str) -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip().splitlines()[-1:] == ["1"]
    except Exception:
        return False


def resolve_torch_device(requested: str | None = None) -> str:
    value = _normalized(requested, TORCH_DEVICE)
    if value in AUTO_VALUES:
        value = _normalized(OCR_DEVICE)
    if value in CPU_VALUES:
        return "cpu"
    if _prefers_gpu(value) and torch_cuda_available():
        return "cuda:0" if value in GPU_VALUES else value.replace("gpu", "cuda", 1)
    if value in AUTO_VALUES and torch_cuda_available():
        return "cuda:0"
    return "cpu"


def resolve_paddle_device(requested: str | None = None) -> str:
    value = _normalized(requested, PADDLE_DEVICE)
    if value in AUTO_VALUES:
        value = _normalized(OCR_DEVICE)
    if value in CPU_VALUES:
        return "cpu"
    if _prefers_gpu(value) and paddle_cuda_available():
        return "gpu:0" if value in GPU_VALUES else value.replace("cuda", "gpu", 1)
    if value in AUTO_VALUES and paddle_cuda_available():
        return "gpu:0"
    return "cpu"


def easyocr_gpu_enabled() -> bool:
    return resolve_torch_device().startswith("cuda")


def device_summary() -> dict[str, object]:
    return {
        "ocr_device": OCR_DEVICE,
        "torch_cuda_available": torch_cuda_available(),
        "torch_device": resolve_torch_device(),
        "paddle_cuda_available": paddle_cuda_available(),
        "paddle_device": resolve_paddle_device(),
    }
