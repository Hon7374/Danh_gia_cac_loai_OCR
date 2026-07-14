from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import JOBS_DIR
from app.main import (
    _apply_ground_truth_metrics,
    _is_active_demo_result,
    _read_ground_truth_file,
    _refresh_document_archive,
    build_comparison_summary,
)
from app.ocr_engines.base import OCRBox
from app.ocr_engines.paddleocr_vl_engine import _sanitize_vl_hallucinations
from app.services.ocr_quality import analyze_ocr_text_quality
from app.services.storage import save_json, save_results_csv
from scripts.refresh_scanned_jobs_with_finetuned_vietocr import (
    choose_best_engine,
    preserve_result_history,
    row_metric_snapshot,
)


def _ground_truth(job_dir: Path, report: dict[str, Any]) -> str:
    metadata = report.get("ground_truth_file") or {}
    relative_path = metadata.get("relative_path")
    if not relative_path:
        return ""
    path = job_dir / relative_path
    if not path.exists():
        return ""
    text, warning, reader = _read_ground_truth_file(path)
    metadata["text_length"] = len(text)
    metadata["warning"] = warning
    metadata["reader"] = reader
    report["ground_truth_text_length"] = len(text)
    return text


def _sanitize_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    old_boxes = [box for box in (row.get("boxes") or []) if isinstance(box, dict)]
    ocr_boxes = [
        OCRBox(
            text=str(box.get("text") or ""),
            confidence=box.get("confidence"),
            bbox=box.get("bbox"),
            label=box.get("label"),
            polygon=box.get("polygon"),
        )
        for box in old_boxes
    ]
    cleaned_text, kept_ocr_boxes, diagnostics = _sanitize_vl_hallucinations(
        str(row.get("text") or ""),
        ocr_boxes,
    )
    kept_ids = {id(box) for box in kept_ocr_boxes}
    kept_boxes = [
        copy.deepcopy(raw_box)
        for raw_box, ocr_box in zip(old_boxes, ocr_boxes)
        if id(ocr_box) in kept_ids
    ]

    candidate = copy.deepcopy(row)
    candidate["text"] = cleaned_text
    candidate["boxes"] = kept_boxes
    candidate["quality_guard"] = analyze_ocr_text_quality(cleaned_text)
    raw = candidate.setdefault("raw", {})
    raw["paddleocr_vl_hallucination_guard"] = diagnostics
    raw.setdefault("repairs", []).append(
        {
            "type": "paddleocr_vl_hallucination_guard",
            "timestamp_unix": time.time(),
            "removed_box_count": diagnostics.get("removed_box_count"),
            "removed_text_characters": diagnostics.get("removed_text_characters"),
            "policy": diagnostics.get("policy"),
        }
    )
    return candidate, diagnostics


def sanitize_job(job_id: str, variants: set[str] | None = None) -> dict[str, Any]:
    safe_job_id = Path(job_id).name
    if safe_job_id != job_id:
        raise ValueError("Invalid job id")
    job_dir = JOBS_DIR / safe_job_id
    report_path = job_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth = _ground_truth(job_dir, report)
    results = list(report.get("results") or [])
    attempts: list[dict[str, Any]] = []
    changed = False

    for index, row in enumerate(results):
        if row.get("engine") != "paddleocr_vl":
            continue
        variant = str(row.get("variant") or "raw")
        if variants and variant not in variants:
            continue
        candidate, diagnostics = _sanitize_row(row)
        attempt = {
            "engine": "paddleocr_vl",
            "variant": variant,
            "old": row_metric_snapshot(row),
            "removed_box_count": diagnostics.get("removed_box_count"),
            "removed_text_characters": diagnostics.get("removed_text_characters"),
            "changed": candidate.get("text") != row.get("text") or candidate.get("boxes") != row.get("boxes"),
        }
        if not attempt["changed"]:
            attempts.append(attempt)
            continue
        if not candidate.get("text"):
            attempt["error"] = "Hallucination guard produced empty text; candidate rejected."
            attempts.append(attempt)
            continue
        if truth:
            _apply_ground_truth_metrics([candidate], truth)
        attempt["new"] = row_metric_snapshot(candidate)
        preserve_result_history(
            report,
            row,
            replacement_mode="paddleocr_vl_hallucination_guard",
            reason="replaced_by_paddleocr_vl_hallucination_guard",
        )
        results[index] = candidate
        attempts.append(attempt)
        changed = True

    if truth:
        _apply_ground_truth_metrics(results, truth)
    report["results"] = results
    active_results = [row for row in results if _is_active_demo_result(row)]
    report["comparison_summary"] = build_comparison_summary(active_results)
    report["best_engine"] = choose_best_engine(report)
    report.setdefault("refresh_history", []).append(
        {
            "type": "paddleocr_vl_hallucination_guard",
            "timestamp_unix": time.time(),
            "ground_truth_recomputed": bool(truth),
            "rows": attempts,
        }
    )

    # Persist the complete previous row before archive synchronization.  The
    # archive service then versions the old TXT/JSON/DOCX under 09_history.
    save_json(report_path, report)
    if changed:
        _refresh_document_archive(job_dir, report)
        save_json(report_path, report)
    save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
    save_results_csv(job_dir / "benchmark_results.csv", results)
    return {
        "job_id": safe_job_id,
        "changed": changed,
        "best_engine": report.get("best_engine"),
        "rows": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove audited PaddleOCR-VL runaway numbered hallucinations from saved jobs."
    )
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--variant", action="append", choices=["raw", "opencv_preprocessed"])
    args = parser.parse_args()
    variants = set(args.variant) if args.variant else None
    output = [sanitize_job(job_id, variants=variants) for job_id in args.job_id]
    for item in output:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
