from __future__ import annotations

import json
import sys
from pathlib import Path

from app.ocr_engines.paddle_vietocr_engine import run_paddle_vietocr_direct


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python -m app.ocr_engines.paddle_vietocr_worker <image_path> <variant> <out_json>", file=sys.stderr)
        return 2
    result = run_paddle_vietocr_direct(Path(sys.argv[1]), sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
