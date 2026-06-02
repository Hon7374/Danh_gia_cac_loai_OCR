from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import JOBS_DIR
from app.main import (
    _apply_ground_truth_metrics,
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

    changed_rows = []
    for idx, row in enumerate(list(results)):
        engine_key = row.get("engine")
        if row.get("variant") != "opencv_preprocessed" or engine_key not in target_engines:
            continue
        engine_cls = ENGINE_REGISTRY.get(engine_key)
        if not engine_cls:
            continue
        new_row = _run_engine_on_pages(engine_cls, engine_key, "opencv_preprocessed", preprocessed_images)
        results[idx] = new_row
        changed_rows.append(engine_key)

    truth = _ground_truth_text(job_dir, report)
    if truth:
        _apply_ground_truth_metrics(results, truth)

    report["results"] = results
    report["comparison_summary"] = build_comparison_summary(results)

    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("text")]
    max_text_len = max([len(row.get("text") or "") for row in ok_rows] or [1])
    if ok_rows:
        best = max(ok_rows, key=lambda row: ocr_selection_score(row, max_text_len))
        report["best_engine"] = {"engine": best.get("engine"), "variant": best.get("variant")}

    if refresh_layout:
        _refresh_layout_postprocess(job_dir, report)
    _refresh_document_archive(job_dir, report)

    save_json(report_path, report)
    save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
    save_results_csv(job_dir / "benchmark_results.csv", results)

    return {
        "job_id": safe_job_id,
        "engines": changed_rows,
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
