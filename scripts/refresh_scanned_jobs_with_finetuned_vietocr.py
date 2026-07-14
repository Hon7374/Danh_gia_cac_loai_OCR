from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
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
from app.main import (
    _aggregate_page_rows,
    _apply_ground_truth_metrics,
    _is_active_demo_result,
    _read_ground_truth_file,
    build_comparison_summary,
)
from app.ocr_engines import ENGINE_REGISTRY
from app.ocr_engines.base import OCRBox
from app.ocr_engines import paddle_vietocr_engine as refine_engine
from app.services.ocr_quality import ocr_selection_score
from app.services.storage import save_json, save_results_csv


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_ground_truth(job_dir: Path, report: dict[str, Any]) -> str:
    meta = report.get("ground_truth_file")
    if not isinstance(meta, dict) or not meta.get("relative_path"):
        return ""
    gt_path = job_dir / meta["relative_path"]
    if not gt_path.exists():
        return ""
    try:
        text, warning, reader = _read_ground_truth_file(gt_path)
        meta["text_length"] = len(text)
        meta["reader"] = meta.get("reader") or reader
        if warning and not meta.get("warning"):
            meta["warning"] = warning
        report["ground_truth_text_length"] = len(text)
        return text
    except Exception as exc:
        meta["warning"] = f"Không đọc lại được ground-truth: {exc}"
        return ""


def variant_images(report: dict[str, Any], variant: str) -> list[str]:
    if variant == "opencv_preprocessed":
        images = list(report.get("preprocessed_images") or [])
        if images:
            return images
        single = report.get("preprocessed_image")
        return [single] if single else []
    images = list(report.get("raw_images") or [])
    if images:
        return images
    single = report.get("raw_image")
    return [single] if single else []


def boxes_for_page(row: dict[str, Any]) -> dict[int, list[OCRBox]]:
    pages: dict[int, list[OCRBox]] = defaultdict(list)
    for raw_box in row.get("boxes") or []:
        if not isinstance(raw_box, dict):
            continue
        text = str(raw_box.get("text") or "").strip()
        bbox = raw_box.get("bbox")
        if not text or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            clean_bbox = [int(v) for v in bbox]
            page = int(raw_box.get("page") or 1)
        except Exception:
            continue
        pages[page].append(
            OCRBox(
                text=text,
                confidence=raw_box.get("confidence"),
                bbox=clean_bbox,
                label=raw_box.get("label"),
                polygon=raw_box.get("polygon"),
            )
        )
    return pages


def row_metric_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "text_len": len(row.get("text") or ""),
        "elapsed_sec": row.get("elapsed_sec"),
        "cer": row.get("cer"),
        "wer": row.get("wer"),
        "quality_guard": row.get("quality_guard"),
    }


def has_reliable_page_numbers(row: dict[str, Any], image_count: int) -> bool:
    if image_count <= 1:
        return True
    pages = []
    for box in row.get("boxes") or []:
        if isinstance(box, dict) and box.get("page") is not None:
            try:
                pages.append(int(box.get("page")))
            except Exception:
                pass
    if not pages:
        return False
    return len(set(pages)) > 1 and max(pages) > 1


def error_score(row: dict[str, Any]) -> float | None:
    cer_value = row.get("cer")
    wer_value = row.get("wer")
    if isinstance(cer_value, (int, float)) and isinstance(wer_value, (int, float)):
        return 0.65 * float(cer_value) + 0.35 * float(wer_value)
    if isinstance(cer_value, (int, float)):
        return float(cer_value)
    if isinstance(wer_value, (int, float)):
        return float(wer_value)
    return None


def should_accept_candidate(old_row: dict[str, Any], candidate: dict[str, Any], has_ground_truth: bool) -> tuple[bool, str]:
    if candidate.get("status") != "ok" or not candidate.get("text"):
        return False, "Kết quả fine-tune không OK hoặc rỗng."
    if has_ground_truth:
        old_score = error_score(old_row)
        new_score = error_score(candidate)
        if new_score is None:
            return False, "Không tính được CER/WER cho kết quả fine-tune."
        if old_score is not None and new_score > old_score + 0.002:
            return False, f"CER/WER xấu hơn bản cũ ({old_score:.4f} -> {new_score:.4f})."
        return True, "CER/WER tốt hơn hoặc tương đương bản cũ."
    old_guard = old_row.get("quality_guard") or {}
    new_guard = candidate.get("quality_guard") or {}
    old_quality = float(old_guard.get("vietnamese_quality_score") or 0)
    new_quality = float(new_guard.get("vietnamese_quality_score") or 0)
    if new_guard.get("severe_diacritic_loss"):
        return False, "Kết quả fine-tune vẫn bị cảnh báo rụng dấu nghiêm trọng."
    if new_quality + 1 < old_quality:
        return False, f"Điểm guard tiếng Việt giảm ({old_quality:.2f} -> {new_quality:.2f})."
    return True, "Vietnamese quality guard tốt hơn hoặc tương đương."


def refine_existing_boxes(job_dir: Path, report: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = row.get("variant") or "raw"
    images = variant_images(report, variant)
    if not images:
        refreshed_row = dict(row)
        return refreshed_row, {
            "variant": variant,
            "status": "skipped",
            "reason": "Không tìm thấy ảnh scan tương ứng trong report.",
            "old": row_metric_snapshot(row),
            "new": row_metric_snapshot(row),
            "refreshed_pages": 0,
            "failed_pages": 0,
        }
    if not has_reliable_page_numbers(row, len(images)):
        refreshed_row = dict(row)
        return refreshed_row, {
            "variant": variant,
            "status": "skipped",
            "reason": "Box cũ không có page number đáng tin cậy cho file nhiều trang; không refine để tránh làm sai đánh giá.",
            "old": row_metric_snapshot(row),
            "new": row_metric_snapshot(row),
            "refreshed_pages": 0,
            "failed_pages": 0,
        }
    grouped_boxes = boxes_for_page(row)
    page_rows: list[dict[str, Any]] = []
    refresh_start = time.perf_counter()
    refreshed_pages = 0
    failed_pages = 0
    hybrid_total_boxes = 0
    hybrid_accepted = 0
    hybrid_fallback = 0
    hybrid_reason_counts: dict[str, int] = defaultdict(int)

    for page_no, rel_image in enumerate(images, start=1):
        image_path = job_dir / rel_image
        source_boxes = grouped_boxes.get(page_no) or []
        page_start = time.perf_counter()
        if not image_path.exists():
            failed_pages += 1
            page_rows.append(
                {
                    "page": page_no,
                    "status": "error",
                    "text": "",
                    "boxes": [],
                    "elapsed_sec": time.perf_counter() - page_start,
                    "error": f"Không tìm thấy ảnh trang: {rel_image}",
                }
            )
            continue
        if not source_boxes:
            failed_pages += 1
            page_rows.append(
                {
                    "page": page_no,
                    "status": "skipped",
                    "text": "",
                    "boxes": [],
                    "elapsed_sec": time.perf_counter() - page_start,
                    "error": "Không có box PaddleOCR cũ để refine.",
                }
            )
            continue

        vietocr_candidates = refine_engine._try_vietocr_recognize_local(image_path, source_boxes)
        elapsed = time.perf_counter() - page_start
        if vietocr_candidates is None:
            failed_pages += 1
            page_rows.append(
                {
                    "page": page_no,
                    "status": "error",
                    "text": "",
                    "boxes": [],
                    "elapsed_sec": elapsed,
                    "error": "VietOCR fine-tuned recognizer không trả kết quả.",
                }
            )
            continue

        refined, hybrid_stats = refine_engine._apply_hybrid_refinement(
            source_boxes,
            vietocr_candidates,
            model_info=dict(refine_engine._VIETOCR_MODEL_INFO),
        )
        refreshed_pages += 1
        hybrid_total_boxes += int(hybrid_stats.get("total_paddle_boxes") or 0)
        hybrid_accepted += int(hybrid_stats.get("accepted_refinements") or 0)
        hybrid_fallback += int(hybrid_stats.get("fallback_to_paddle") or 0)
        for reason, count in (hybrid_stats.get("fallback_reason_counts") or {}).items():
            hybrid_reason_counts[str(reason)] += int(count or 0)
        refine_note = (
            f"VietOCR hybrid accepted {hybrid_stats.get('accepted_refinements', 0)}/"
            f"{hybrid_stats.get('total_paddle_boxes', len(source_boxes))} crops; "
            f"Paddle fallback for {hybrid_stats.get('fallback_to_paddle', 0)} crops"
        )
        boxes = [
            {
                "text": box.text,
                "confidence": box.confidence,
                "bbox": box.bbox,
                "label": box.label,
                "polygon": box.polygon,
                "page": page_no,
            }
            for box in refined
        ]
        page_rows.append(
            {
                "page": page_no,
                "status": "ok",
                "text": "\n".join(box["text"] for box in boxes if box.get("text")),
                "boxes": boxes,
                "elapsed_sec": elapsed,
                "error": "",
                "raw": {
                    "note": "PaddleOCR geometry with guarded VietOCR recognition and per-crop Paddle fallback.",
                    "refine": refine_note,
                    "vietocr_model": dict(refine_engine._VIETOCR_MODEL_INFO),
                    "hybrid_refinement": hybrid_stats,
                },
            }
        )

    refreshed_row = _aggregate_page_rows(
        "paddle_vietocr",
        variant,
        page_rows,
        elapsed_sec=time.perf_counter() - refresh_start,
    )
    refreshed_row["previous_eval"] = row_metric_snapshot(row)
    refreshed_row["raw"]["engine_display_name"] = "PaddleOCR + VietOCR hybrid"
    refreshed_row["raw"]["note"] = (
        "Existing PaddleOCR detections were recognized with guarded VietOCR; "
        "every rejected/invalid VietOCR candidate retained the original Paddle text."
    )
    refreshed_row["raw"]["refine"] = (
        f"VietOCR hybrid accepted {hybrid_accepted}/{hybrid_total_boxes} crops; "
        f"Paddle fallback for {hybrid_fallback} crops"
    )
    refreshed_row["raw"]["hybrid_refinement"] = {
        "policy": getattr(refine_engine, "_HYBRID_POLICY_VERSION", "paddle-vietocr-hybrid"),
        "status": "completed" if refreshed_pages else "unavailable",
        "total_paddle_boxes": hybrid_total_boxes,
        "accepted_refinements": hybrid_accepted,
        "fallback_to_paddle": hybrid_fallback,
        "fallback_reason_counts": dict(hybrid_reason_counts),
        "paddle_geometry_preserved": True,
        "model": dict(refine_engine._VIETOCR_MODEL_INFO),
    }
    refreshed_row["raw"]["refresh"] = {
        "mode": "existing_paddle_boxes_guarded_vietocr_hybrid",
        "refreshed_at_unix": time.time(),
        "refreshed_pages": refreshed_pages,
        "failed_pages": failed_pages,
        "vietocr_model": dict(refine_engine._VIETOCR_MODEL_INFO),
        "note": (
            "Reused old PaddleOCR detections and applied guarded VietOCR recognition per crop; "
            "unsafe candidates fell back to the original Paddle text. No new detection was run."
        ),
    }
    status = {
        "variant": variant,
        "mode": "existing_boxes_hybrid",
        "old": refreshed_row["previous_eval"],
        "new": row_metric_snapshot(refreshed_row),
        "refreshed_pages": refreshed_pages,
        "failed_pages": failed_pages,
        "hybrid_accepted": hybrid_accepted,
        "hybrid_fallback": hybrid_fallback,
    }
    return refreshed_row, status


def rerun_detection(job_dir: Path, report: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = row.get("variant") or "raw"
    images = [job_dir / rel for rel in variant_images(report, variant)]
    if not images:
        refreshed_row = dict(row)
        return refreshed_row, {
            "variant": variant,
            "status": "skipped",
            "reason": "Không tìm thấy ảnh scan tương ứng trong report.",
            "old": row_metric_snapshot(row),
            "new": row_metric_snapshot(row),
            "refreshed_pages": 0,
            "failed_pages": 0,
        }
    engine_cls = ENGINE_REGISTRY["paddle_vietocr"]
    from app.main import _run_engine_on_pages

    refreshed_row = _run_engine_on_pages(engine_cls, "paddle_vietocr", variant, images)
    refreshed_row["previous_eval"] = row_metric_snapshot(row)
    refreshed_raw = refreshed_row.setdefault("raw", {})
    refreshed_raw["engine_display_name"] = "PaddleOCR + VietOCR hybrid"
    refreshed_raw["note"] = (
        "PaddleOCR detection/recognition reran from the page image, followed by guarded "
        "VietOCR recognition with per-crop Paddle fallback."
    )
    refreshed_raw["refresh"] = {
        "mode": "rerun_paddle_detection_and_guarded_vietocr_hybrid",
        "refreshed_at_unix": time.time(),
        "vietocr_model": refreshed_raw.get("vietocr_model"),
        "note": (
            "Reran full PaddleOCR detection/recognition and applied the guarded VietOCR hybrid; "
            "invalid or runaway VietOCR crops retained Paddle text."
        ),
    }
    return refreshed_row, {
        "variant": variant,
        "mode": "rerun_detection_hybrid",
        "old": refreshed_row["previous_eval"],
        "new": row_metric_snapshot(refreshed_row),
        "refreshed_pages": refreshed_row.get("raw", {}).get("ok_pages"),
        "failed_pages": (refreshed_row.get("raw", {}).get("error_pages") or 0) + (refreshed_row.get("raw", {}).get("skipped_pages") or 0),
    }


def choose_best_engine(report: dict[str, Any]) -> dict[str, str] | None:
    ok_rows = [
        row
        for row in report.get("results") or []
        if _is_active_demo_result(row) and row.get("status") == "ok" and row.get("text")
    ]
    if not ok_rows:
        return None
    best = max(ok_rows, key=ocr_selection_score)
    return {"engine": best.get("engine"), "variant": best.get("variant")}


def preserve_result_history(
    report: dict[str, Any],
    row: dict[str, Any],
    *,
    replacement_mode: str,
    reason: str = "replaced_by_vietocr_refresh",
) -> None:
    """Archive the complete JSON row immediately before replacing it."""
    history = report.get("result_history")
    if not isinstance(history, list):
        history = [] if history is None else [copy.deepcopy(history)]
        report["result_history"] = history
    history_index = len(history)
    history.append(copy.deepcopy(row))
    metadata = report.get("result_history_metadata")
    if not isinstance(metadata, list):
        metadata = [] if metadata is None else [copy.deepcopy(metadata)]
        report["result_history_metadata"] = metadata
    metadata.append(
        {
            "archived_at_unix": time.time(),
            "reason": reason,
            "replacement_mode": replacement_mode,
            "result_history_index": history_index,
            "engine": row.get("engine"),
            "variant": row.get("variant"),
        }
    )


def refresh_report(job_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report_path = job_dir / "report.json"
    report = load_json(report_path)
    result_rows = list(report.get("results") or [])
    statuses: list[dict[str, Any]] = []
    changed = False
    ground_truth = get_ground_truth(job_dir, report)

    for idx, row in enumerate(result_rows):
        if row.get("engine") != "paddle_vietocr":
            continue
        variant = row.get("variant") or "raw"
        if args.variant and variant not in set(args.variant):
            continue
        if bool(getattr(args, "rerun_detection", False)):
            refreshed_row, status = rerun_detection(job_dir, report, row)
        elif row.get("status") == "ok" and row.get("boxes"):
            refreshed_row, status = refine_existing_boxes(job_dir, report, row)
        elif args.rerun_missing_detection:
            refreshed_row, status = rerun_detection(job_dir, report, row)
        else:
            statuses.append(
                {
                    "variant": variant,
                    "status": "skipped",
                    "reason": "Không có box cũ hoặc row không OK. Dùng --rerun-missing-detection nếu muốn chạy lại detect.",
                    "old": row_metric_snapshot(row),
                }
            )
            continue
        if status.get("status") == "skipped":
            statuses.append(status)
            continue
        if ground_truth:
            _apply_ground_truth_metrics([refreshed_row], ground_truth)
            status["new"] = row_metric_snapshot(refreshed_row)
        accepted, reason = should_accept_candidate(row, refreshed_row, bool(ground_truth))
        status["accepted"] = accepted
        status["decision"] = reason
        if not accepted:
            statuses.append(status)
            continue
        preserve_result_history(
            report,
            row,
            replacement_mode=str(status.get("mode") or "vietocr_refresh"),
        )
        result_rows[idx] = refreshed_row
        statuses.append(status)
        changed = True

    if not changed:
        return {"job_id": job_dir.name, "changed": False, "rows": statuses}

    report["results"] = result_rows
    if ground_truth:
        _apply_ground_truth_metrics(result_rows, ground_truth)
        refreshed_by_variant = {
            row.get("variant"): row
            for row in result_rows
            if row.get("engine") == "paddle_vietocr"
        }
        for status in statuses:
            variant = status.get("variant")
            if variant in refreshed_by_variant and status.get("new") is not None:
                status["new"] = row_metric_snapshot(refreshed_by_variant[variant])
    active_result_rows = [row for row in result_rows if _is_active_demo_result(row)]
    report["comparison_summary"] = build_comparison_summary(active_result_rows)
    report["best_engine"] = choose_best_engine(report)
    report.setdefault("refresh_history", []).append(
        {
            "type": "vietocr_finetune_refresh",
            "timestamp_unix": time.time(),
            "ground_truth_recomputed": bool(ground_truth),
            "rows": statuses,
        }
    )
    save_json(report_path, report)
    save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
    save_results_csv(job_dir / "benchmark_results.csv", result_rows)
    return {
        "job_id": job_dir.name,
        "changed": True,
        "best_engine": report.get("best_engine"),
        "has_ground_truth": bool(ground_truth),
        "rows": statuses,
    }


def report_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.jobs_dir.glob("*/report.json"))
    if args.job_id:
        wanted = set(args.job_id)
        paths = [path for path in paths if path.parent.name in wanted]
    if args.only_with_paddle_vietocr:
        filtered = []
        for path in paths:
            try:
                report = load_json(path)
            except Exception:
                continue
            if any(row.get("engine") == "paddle_vietocr" for row in report.get("results") or []):
                filtered.append(path)
        paths = filtered
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh scanned OCR jobs with the guarded PaddleOCR + VietOCR hybrid "
            "and recompute evaluation metrics."
        )
    )
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--job-id", action="append", help="Refresh only this job id. Repeat for multiple jobs.")
    parser.add_argument("--variant", action="append", choices=["raw", "opencv_preprocessed"], help="Refresh only one variant. Repeat for both.")
    parser.add_argument(
        "--only-with-paddle-vietocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limit refresh to jobs that already contain Paddle+VietOCR (use --no-only-with-paddle-vietocr for all jobs).",
    )
    parser.add_argument("--rerun-missing-detection", action="store_true", help="Run PaddleOCR detection again for skipped/error rows that have no boxes.")
    parser.add_argument(
        "--rerun-detection",
        action="store_true",
        help=(
            "Force full PaddleOCR detection plus guarded VietOCR hybrid recognition, "
            "even when the existing row already has boxes."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-path", type=Path, default=ROOT / "jobs" / "_vietocr_finetune_refresh_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = report_paths(args)
    if args.limit:
        paths = paths[: args.limit]
    summary = {
        "started_at_unix": time.time(),
        "jobs_total": len(paths),
        "jobs": [],
    }
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] refresh {path.parent.name}")
        try:
            status = refresh_report(path.parent, args)
        except Exception as exc:
            status = {"job_id": path.parent.name, "changed": False, "error": str(exc)}
        summary["jobs"].append(status)
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if status.get("changed"):
            for row in status.get("rows") or []:
                old = row.get("old") or {}
                new = row.get("new") or {}
                print(
                    f"  {row.get('variant')}: CER {old.get('cer')} -> {new.get('cer')}, "
                    f"WER {old.get('wer')} -> {new.get('wer')}, pages {row.get('refreshed_pages')}"
                )
        else:
            print(f"  skipped/no change: {status.get('error') or status.get('rows')}")
    summary["finished_at_unix"] = time.time()
    summary["changed_jobs"] = sum(1 for item in summary["jobs"] if item.get("changed"))
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"changed_jobs": summary["changed_jobs"], "summary_path": str(args.summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
