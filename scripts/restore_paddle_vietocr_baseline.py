from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import JOBS_DIR
from app.main import _aggregate_page_rows, _apply_ground_truth_metrics, _read_ground_truth_file, build_comparison_summary
from app.services.ocr_quality import ocr_selection_score
from app.services.storage import save_json, save_results_csv
from scripts.refresh_scanned_jobs_with_finetuned_vietocr import variant_images


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ground_truth_text(job_dir: Path, report: dict[str, Any]) -> str:
    meta = report.get("ground_truth_file")
    if not isinstance(meta, dict) or not meta.get("relative_path"):
        return ""
    path = job_dir / meta["relative_path"]
    if not path.exists():
        return ""
    try:
        text, warning, reader = _read_ground_truth_file(path)
        meta["text_length"] = len(text)
        meta["reader"] = meta.get("reader") or reader
        if warning and not meta.get("warning"):
            meta["warning"] = warning
        report["ground_truth_text_length"] = len(text)
        return text
    except Exception as exc:
        meta["warning"] = f"Không đọc lại được ground-truth: {exc}"
        return ""


def run_worker_unrefined(image_path: Path, variant: str, page_no: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["PADDLE_VIETOCR_REFINE"] = "0"
    with tempfile.TemporaryDirectory(prefix="paddle_baseline_restore_") as tmp:
        out_json = Path(tmp) / "result.json"
        start = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "app.ocr_engines.paddle_vietocr_worker",
                str(image_path),
                variant,
                str(out_json),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        elapsed = time.perf_counter() - start
        if completed.returncode != 0 or not out_json.exists():
            detail = (completed.stderr or completed.stdout or "").strip()
            return {
                "engine": "paddle_vietocr",
                "variant": variant,
                "status": "error",
                "text": "",
                "boxes": [],
                "elapsed_sec": elapsed,
                "error": detail or "PaddleOCR baseline worker failed",
                "page": page_no,
                "image": str(image_path),
            }
        row = json.loads(out_json.read_text(encoding="utf-8"))
        row["page"] = page_no
        row["image"] = str(image_path)
        return row


def restore_variant(job_dir: Path, report: dict[str, Any], variant: str) -> dict[str, Any]:
    page_rows = []
    start = time.perf_counter()
    for page_no, rel in enumerate(variant_images(report, variant), start=1):
        page_rows.append(run_worker_unrefined(job_dir / rel, variant, page_no))
        print(f"  {job_dir.name} {variant} page {page_no}/{len(variant_images(report, variant))}")
    row = _aggregate_page_rows("paddle_vietocr", variant, page_rows, elapsed_sec=time.perf_counter() - start)
    row["raw"]["restore"] = {
        "mode": "paddleocr_baseline_no_vietocr_refine",
        "restored_at_unix": time.time(),
        "note": "Restored PaddleOCR baseline recognition with PADDLE_VIETOCR_REFINE=0.",
    }
    return row


def choose_best(report: dict[str, Any]) -> dict[str, str] | None:
    rows = [row for row in report.get("results") or [] if row.get("status") == "ok" and row.get("text")]
    if not rows:
        return None
    best = max(rows, key=ocr_selection_score)
    return {"engine": best.get("engine"), "variant": best.get("variant")}


def restore_job(job_dir: Path, variants: set[str]) -> dict[str, Any]:
    report_path = job_dir / "report.json"
    report = load_json(report_path)
    results = list(report.get("results") or [])
    changed = False
    restored = []
    for idx, row in enumerate(results):
        if row.get("engine") != "paddle_vietocr":
            continue
        variant = row.get("variant") or "raw"
        if variant not in variants:
            continue
        if not variant_images(report, variant):
            restored.append({"variant": variant, "status": "skipped", "reason": "Không có ảnh scan."})
            continue
        print(f"restore {job_dir.name} / {variant}")
        new_row = restore_variant(job_dir, report, variant)
        results[idx] = new_row
        changed = True
        restored.append({"variant": variant, "status": new_row.get("status"), "text_len": len(new_row.get("text") or "")})
    if changed:
        report["results"] = results
        gt = ground_truth_text(job_dir, report)
        if gt:
            _apply_ground_truth_metrics(results, gt)
        report["comparison_summary"] = build_comparison_summary(results)
        report["best_engine"] = choose_best(report)
        report.setdefault("refresh_history", []).append(
            {"type": "paddle_vietocr_baseline_restore", "timestamp_unix": time.time(), "rows": restored}
        )
        save_json(report_path, report)
        save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
        save_results_csv(job_dir / "benchmark_results.csv", results)
    return {"job_id": job_dir.name, "changed": changed, "rows": restored, "best_engine": report.get("best_engine")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore paddle_vietocr baseline rows with VietOCR refinement disabled.")
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--variant", action="append", choices=["raw", "opencv_preprocessed"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = set(args.variant or ["raw", "opencv_preprocessed"])
    summary = []
    for job_id in args.job_id:
        summary.append(restore_job(args.jobs_dir / job_id, variants))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
