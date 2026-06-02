from __future__ import annotations

import sys
import types
import os


def install_modelscope_stub() -> None:
    """Keep PaddleOCR workers from importing Torch through modelscope."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")

    if "modelscope" in sys.modules:
        return

    modelscope = types.ModuleType("modelscope")
    hub = types.ModuleType("modelscope.hub")
    errors = types.ModuleType("modelscope.hub.errors")

    class NotExistError(Exception):
        pass

    def snapshot_download(*_args, **_kwargs):
        raise RuntimeError(
            "ModelScope download is disabled in Paddle OCR workers to avoid "
            "Torch/Paddle CUDA DLL conflicts. Use cached models or another "
            "PaddleX model source."
        )

    errors.NotExistError = NotExistError
    hub.errors = errors
    modelscope.hub = hub
    modelscope.snapshot_download = snapshot_download

    sys.modules["modelscope"] = modelscope
    sys.modules["modelscope.hub"] = hub
    sys.modules["modelscope.hub.errors"] = errors
