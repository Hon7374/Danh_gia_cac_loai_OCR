from __future__ import annotations

from app.ocr_engines.paddle_runtime import install_modelscope_stub


def main() -> int:
    """Run PaddleOCR CLI without importing ModelScope/Torch into the Paddle process."""
    install_modelscope_stub()
    from paddleocr.__main__ import console_entry

    result = console_entry()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
