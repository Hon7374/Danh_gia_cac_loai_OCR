from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import JOBS_DIR
from app.main import (
    _apply_ground_truth_metrics,
    _is_active_demo_result,
    _read_ground_truth_file,
    _refresh_document_archive,
    _refresh_layout_postprocess,
    _run_engine_on_pages,
    build_comparison_summary,
)
from app.ocr_engines import ENGINE_REGISTRY
from app.services.ocr_quality import ocr_selection_score
from app.services.preprocess import preprocess_image
from app.services.storage import save_json, save_results_csv
from scripts.refresh_scanned_jobs_with_finetuned_vietocr import (
    preserve_result_history,
    row_metric_snapshot,
    should_accept_candidate,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _relative(job_dir: Path, path: Path) -> str:
    return str(path.relative_to(job_dir))


def _ground_truth_text(job_dir: Path, report: dict) -> str:
    meta = report.get("ground_truth_file") or {}
    rel_path = meta.get("relative_path")
    if not rel_path:
        return ""
    gt_path = job_dir / rel_path
    if not gt_path.exists():
        return ""
    text, warning, reader = _read_ground_truth_file(gt_path)
    report["ground_truth_text_length"] = len(text)
    meta["text_length"] = len(text)
    meta["warning"] = warning
    meta["reader"] = reader
    return text


def _summarize_steps(preprocessed_results: list[tuple[Path, list[str]]]) -> tuple[list[str], list[dict]]:
    steps = ["adaptive_safe_preprocess"]
    steps_by_page = []
    for idx, (_, page_steps) in enumerate(preprocessed_results, start=1):
        steps_by_page.append({"page": idx, "steps": page_steps})
        for step in page_steps:
            if step not in steps:
                steps.append(step)
    return steps, steps_by_page


def refresh_job(job_id: str, engines: set[str] | None = None, refresh_layout: bool = True) -> dict:
    safe_job_id = Path(job_id).name
    job_dir = JOBS_DIR / safe_job_id
    report_path = job_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found for job {safe_job_id}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_images = [job_dir / rel for rel in (report.get("raw_images") or [])]
    if not raw_images and report.get("raw_image"):
        raw_images = [job_dir / report["raw_image"]]
    if not raw_images:
        raise ValueError(f"job {safe_job_id} has no raw images")

    image_dir = job_dir / "images"
    preprocessed_results = [preprocess_image(raw_page, image_dir) for raw_page in raw_images]
    preprocessed_images = [item[0] for item in preprocessed_results]
    steps, steps_by_page = _summarize_steps(preprocessed_results)
    report["preprocessed_images"] = [_relative(job_dir, path) for path in preprocessed_images]
    report["preprocessed_image"] = _relative(job_dir, preprocessed_images[0])
    report["opencv_steps"] = steps
    report["opencv_steps_by_page"] = steps_by_page

    results = report.get("results") or []
    available_opencv_engines = {
        row.get("engine")
        for row in results
        if row.get("variant") == "opencv_preprocessed" and row.get("engine") in ENGINE_REGISTRY
    }
    target_engines = engines or set(available_opencv_engines)
    truth = _ground_truth_text(job_dir, report)

    changed_rows = []
    attempts = []
    for idx, row in enumerate(list(results)):
        engine_key = row.get("engine")
        if row.get("variant") != "opencv_preprocessed" or engine_key not in target_engines:
            continue
        engine_cls = ENGINE_REGISTRY.get(engine_key)
        if not engine_cls:
            continue
        try:
            evaluated_old = dict(row)
            if truth:
                _apply_ground_truth_metrics([evaluated_old], truth)
            new_row = _run_engine_on_pages(engine_cls, engine_key, "opencv_preprocessed", preprocessed_images)
            if truth:
                _apply_ground_truth_metrics([new_row], truth)
            accepted, decision = should_accept_candidate(evaluated_old, new_row, bool(truth))
        except Exception as exc:
            attempts.append(
                {
                    "engine": engine_key,
                    "variant": "opencv_preprocessed",
                    "accepted": False,
                    "decision": f"Candidate refresh failed: {exc}",
                    "old": row_metric_snapshot(row),
                    "new": None,
                }
            )
            continue

        attempt = {
            "engine": engine_key,
            "variant": "opencv_preprocessed",
            "accepted": accepted,
            "decision": decision,
            "old": row_metric_snapshot(evaluated_old),
            "new": row_metric_snapshot(new_row),
        }
        attempts.append(attempt)
        if not accepted:
            continue

        preserve_result_history(
            report,
            row,
            replacement_mode="opencv_safe_preprocessing_refresh",
            reason="replaced_by_opencv_safe_preprocessing_refresh",
        )
        results[idx] = new_row
        changed_rows.append(engine_key)

    if truth:
        _apply_ground_truth_metrics(results, truth)

    report["results"] = results
    active_results = [row for row in results if _is_active_demo_result(row)]
    report["comparison_summary"] = build_comparison_summary(active_results)

    ok_rows = [row for row in active_results if row.get("status") == "ok" and row.get("text")]
    max_text_len = max([len(row.get("text") or "") for row in ok_rows] or [1])
    if ok_rows:
        best = max(ok_rows, key=lambda row: ocr_selection_score(row, max_text_len))
        report["best_engine"] = {"engine": best.get("engine"), "variant": best.get("variant")}

    report.setdefault("refresh_history", []).append(
        {
            "type": "opencv_safe_preprocessing_refresh",
            "timestamp_unix": time.time(),
            "ground_truth_recomputed": bool(truth),
            "rows": attempts,
        }
    )

    if refresh_layout:
        _refresh_layout_postprocess(job_dir, report)

    save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
    save_results_csv(job_dir / "benchmark_results.csv", results)
    # Persist the accepted row and its complete result_history entry before the
    # archive sync. If archive repair fails, a later report view can safely
    # retry it without losing the replacement audit trail.
    save_json(report_path, report)
    _refresh_document_archive(job_dir, report)
    save_json(report_path, report)

    return {
        "job_id": safe_job_id,
        "engines": changed_rows,
        "attempts": attempts,
        "opencv_steps": steps,
        "best_engine": report.get("best_engine"),
        "summary": report.get("comparison_summary"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate safe OpenCV images and refresh opencv OCR rows.")
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--engine", action="append", choices=sorted(ENGINE_REGISTRY.keys()))
    parser.add_argument("--skip-layout", action="store_true")
    args = parser.parse_args()

    engines = set(args.engine) if args.engine else None
    all_results = []
    for job_id in args.job_id:
        result = refresh_job(job_id, engines=engines, refresh_layout=not args.skip_layout)
        all_results.append(result)
        print(f"{result['job_id']}: refreshed {', '.join(result['engines']) or 'no OCR rows'}")
        for row in (result["summary"] or {}).get("rows") or []:
            if row.get("variant") == "opencv_preprocessed":
                print(
                    f"  {row.get('engine')} opencv: "
                    f"CER={row.get('cer_pct')} WER={row.get('wer_pct')} "
                    f"time={row.get('elapsed_sec')}s"
                )

    out_path = JOBS_DIR / "_opencv_safe_refresh_summary.json"
    save_json(out_path, all_results)
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
