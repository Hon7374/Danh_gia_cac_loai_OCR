from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from app.ocr_engines.base import OCRBox
from app.ocr_engines import paddle_vietocr_engine as refine_engine


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: python -m app.ocr_engines.vietocr_refine_worker <image_path> <boxes_json> <out_json>",
            file=sys.stderr,
        )
        return 2
    image_path = Path(sys.argv[1])
    boxes_json = Path(sys.argv[2])
    out_json = Path(sys.argv[3])
    try:
        raw_boxes = json.loads(boxes_json.read_text(encoding="utf-8"))
        boxes = [
            OCRBox(
                text=b.get("text", ""),
                confidence=b.get("confidence"),
                bbox=b.get("bbox"),
                label=b.get("label"),
                polygon=b.get("polygon"),
            )
            for b in raw_boxes
            if isinstance(b, dict)
        ]
        refined = refine_engine._try_vietocr_recognize_local(image_path, boxes)
        if refined is None:
            payload = {
                "status": "error",
                "error": "VietOCR local recognizer unavailable",
                "model_info": refine_engine._VIETOCR_MODEL_INFO,
                "boxes": [],
            }
        else:
            payload = {
                "status": "ok",
                "error": "",
                "model_info": refine_engine._VIETOCR_MODEL_INFO,
                "boxes": [
                    {
                        "text": b.text,
                        "confidence": b.confidence,
                        "bbox": b.bbox,
                        "label": b.label,
                        "polygon": b.polygon,
                    }
                    for b in refined
                ],
            }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "error": f"{exc}\n{traceback.format_exc()}",
            "model_info": refine_engine._VIETOCR_MODEL_INFO,
            "boxes": [],
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
