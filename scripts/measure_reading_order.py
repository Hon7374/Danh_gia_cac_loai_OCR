from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import _read_ground_truth_file
from app.services.metrics import cer, wer
from app.services.reading_order import order_boxes_xy_cut


def _metric_percent(value: float | None) -> float | None:
    return round(value * 100.0, 6) if value is not None else None


def _find_row(report: dict[str, Any], engine: str, variant: str) -> dict[str, Any]:
    for row in report.get("results") or []:
        if row.get("engine") == engine and row.get("variant") == variant:
            return row
    raise RuntimeError(f"Missing result row: {engine}/{variant}")


def measure(job_dir: Path, engine: str, variant: str) -> dict[str, Any]:
    report_path = job_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = _find_row(report, engine, variant)
    boxes = [box for box in row.get("boxes") or [] if isinstance(box, dict)]
    raw_images = list(report.get("raw_images") or [])
    if not raw_images and report.get("raw_image"):
        raw_images = [report["raw_image"]]

    page_texts: list[str] = []
    page_diagnostics: list[dict[str, Any]] = []
    reordered_count = 0
    for page_no, relative_image in enumerate(raw_images, start=1):
        page_boxes = [box for box in boxes if int(box.get("page") or 1) == page_no]
        image_path = job_dir / Path(str(relative_image))
        with Image.open(image_path) as image:
            ordered, diagnostics = order_boxes_xy_cut(page_boxes, image.width, image.height)
        if len(ordered) != len(page_boxes):
            raise RuntimeError(f"Box count changed on page {page_no}")
        reordered_count += int(diagnostics.get("changed_positions") or 0)
        page_diagnostics.append({"page": page_no, **diagnostics})
        page_texts.append("\n".join(str(box.get("text") or "") for box in ordered))

    reordered_text = "\n\n".join(text.strip() for text in page_texts if text.strip())
    truth = ""
    truth_meta = report.get("ground_truth_file") or {}
    relative_truth = truth_meta.get("relative_path")
    if relative_truth:
        truth_path = job_dir / Path(str(relative_truth))
        truth, _, _ = _read_ground_truth_file(truth_path)

    before_cer = cer(str(row.get("text") or ""), truth) if truth else None
    before_wer = wer(str(row.get("text") or ""), truth) if truth else None
    after_cer = cer(reordered_text, truth) if truth else None
    after_wer = wer(reordered_text, truth) if truth else None
    return {
        "job_id": report.get("job_id") or job_dir.name,
        "engine": engine,
        "variant": variant,
        "page_count": len(raw_images),
        "box_count": len(boxes),
        "reordered_positions": reordered_count,
        "text_length_before": len(str(row.get("text") or "")),
        "text_length_after": len(reordered_text),
        "cer_before_pct": _metric_percent(before_cer),
        "cer_after_pct": _metric_percent(after_cer),
        "cer_improvement_points": (
            round((before_cer - after_cer) * 100.0, 6)
            if before_cer is not None and after_cer is not None
            else None
        ),
        "wer_before_pct": _metric_percent(before_wer),
        "wer_after_pct": _metric_percent(after_wer),
        "wer_improvement_points": (
            round((before_wer - after_wer) * 100.0, 6)
            if before_wer is not None and after_wer is not None
            else None
        ),
        "pages": page_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure geometry-only XY-cut reading order without modifying a job report."
    )
    parser.add_argument("job_id")
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs")
    parser.add_argument("--engine", default="paddle_vietocr")
    parser.add_argument("--variant", default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = measure(args.jobs_dir.resolve() / args.job_id, args.engine, args.variant)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
