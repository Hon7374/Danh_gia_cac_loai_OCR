from __future__ import annotations

import os
import re
import subprocess
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
import json
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    JOBS_DIR,
    OCR_GPU_WORKERS,
    OCR_OPENCV_WORKERS,
    OCR_TESSERACT_WORKERS,
    ROOT_DIR,
    STORAGE_DIR,
)
from app.ocr_engines import ENGINE_REGISTRY
from app.services.metrics import cer, wer
from app.services.ocr_quality import analyze_ocr_text_quality, ocr_selection_score
from app.services.pdf_utils import ensure_images
from app.services.preprocess import preprocess_image
from app.services.storage import new_job_dir, save_json, save_results_csv
from app.services.layoutlmv3_postprocess import layoutlmv3_postprocess, stabilize_layout_fields_from_rows
from app.services.devices import device_summary
from app.services.field_extract import extract_fields_rule_based
from app.services.document_archive import archive_scan, ensure_archive_word_outputs, sync_archive_metadata

app = FastAPI(title="OCR Full Benchmark Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT_DIR / "app" / "templates")

GROUND_TRUTH_METRIC_VERSION = 2
GROUND_TRUTH_LENGTH_RATIO_LIMIT = 3
GROUND_TRUTH_LENGTH_ABSOLUTE_LIMIT = 6000
TEXT_INPUT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".doc", ".docx"}


def _append_warning(existing: str | None, extra: str | None) -> str | None:
    extra = (extra or "").strip()
    if not extra:
        return existing
    existing = (existing or "").strip()
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}"


def _is_text_input_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_INPUT_SUFFIXES


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response: Response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def _avg_confidence(row: dict) -> float | None:
    vals = []
    for b in row.get("boxes") or []:
        c = b.get("confidence")
        if isinstance(c, (int, float)):
            vals.append(float(c))
    if not vals:
        return None
    # Một số engine trả confidence 0-1, một số trả 0-100. Chuẩn hóa về %.
    avg = sum(vals) / len(vals)
    return avg * 100 if avg <= 1.0 else avg


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100, 2)


def _round(value: float | None, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)




def _decode_uploaded_text(data: bytes) -> str:
    """Đọc file text chuẩn từ upload.

    Hỗ trợ TXT/MD/CSV/JSON theo các encoding phổ biến. Nếu là DOCX thì đọc nội dung
    word/document.xml để không cần cài thêm python-docx.
    """
    for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _normalize_ground_truth_text(text: str) -> str:
    return (
        text.replace("\ufeff", "")
        .replace("\x07", "\n")
        .replace("\x0b", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _extract_docx_text(path: Path) -> str:
    texts: list[str] = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        with zf.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
    for p in root.findall(".//w:p", ns):
        parts = [t.text or "" for t in p.findall(".//w:t", ns)]
        line = "".join(parts).strip()
        if line:
            texts.append(line)
    return "\n".join(texts)


def _ps_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _extract_doc_text(path: Path) -> str:
    extracted_path = path.with_name(f"{path.stem}_extracted.txt")
    if extracted_path.exists() and extracted_path.stat().st_mtime >= path.stat().st_mtime:
        return extracted_path.read_text(encoding="utf-8-sig", errors="ignore")

    script = f"""
$ErrorActionPreference = 'Stop'
$docPath = {_ps_literal(str(path.resolve()))}
$outPath = {_ps_literal(str(extracted_path.resolve()))}
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {{
  $doc = $word.Documents.Open($docPath, $false, $true)
  try {{
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($outPath, $doc.Content.Text, $utf8)
  }} finally {{
    $doc.Close([ref]$false)
  }}
}} finally {{
  $word.Quit()
}}
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "Microsoft Word COM extraction failed")
    return extracted_path.read_text(encoding="utf-8-sig", errors="ignore")


def _read_ground_truth_file(path: Path, data: bytes | None = None) -> tuple[str, str | None, str]:
    suffix = path.suffix.lower()
    warning = None
    reader = "text"
    if suffix == ".docx":
        text = _extract_docx_text(path)
        reader = "docx"
    elif suffix == ".doc":
        try:
            text = _extract_doc_text(path)
            reader = "microsoft_word_doc"
            warning = "Da doc file .doc bang Microsoft Word. CER/WER se so sanh voi text OCR cua toan bo file da chay."
        except Exception as exc:
            text = ""
            reader = "unsupported_doc"
            warning = f"Chua doc duoc file .doc nhi phan cu: {exc}. Hay chuyen sang .docx hoac .txt de tinh CER/WER."
    else:
        text = _decode_uploaded_text(data if data is not None else path.read_bytes())
    return _normalize_ground_truth_text(text), warning, reader


def _truth_for_prediction(pred_text: str, truth_text: str) -> tuple[str, bool, str | None]:
    truth = truth_text.strip()
    pred = pred_text.strip()
    if not truth or not pred:
        return truth, False, None
    pred_len = len(pred)
    truth_len = len(truth)
    if pred_len > max(GROUND_TRUTH_LENGTH_ABSOLUTE_LIMIT, truth_len * GROUND_TRUTH_LENGTH_RATIO_LIMIT):
        warning = (
            f"Bo qua CER/WER: OCR co {pred_len} ky tu, text chuan chi {truth_len} ky tu. "
            "Ground truth khong khop cung pham vi tai lieu/trang OCR."
        )
        return "", False, warning
    if truth_len <= max(GROUND_TRUTH_LENGTH_ABSOLUTE_LIMIT, pred_len * GROUND_TRUTH_LENGTH_RATIO_LIMIT):
        return truth, False, None
    compare_len = max(1000, int(len(pred) * 1.35))
    compare_len = min(compare_len, len(truth))
    return truth[:compare_len], True, None


def _apply_ground_truth_metrics(result_rows: list[dict], ground_truth: str) -> dict:
    stats = {
        "used_slice": False,
        "skipped_mismatch": False,
        "warning": None,
    }
    for row in result_rows:
        text = row.get("text") or ""
        row["ground_truth_metric_version"] = GROUND_TRUTH_METRIC_VERSION
        row.pop("ground_truth_metric_warning", None)
        if ground_truth.strip() and text:
            compare_truth, sliced, warning = _truth_for_prediction(text, ground_truth)
            row["ground_truth_compare_length"] = len(compare_truth)
            if warning:
                row["cer"] = None
                row["wer"] = None
                row["ground_truth_metric_warning"] = warning
                stats["skipped_mismatch"] = True
                stats["warning"] = stats["warning"] or (
                    "Bo qua CER/WER vi text chuan khong khop voi pham vi tai lieu OCR. "
                    "Hay upload ground truth cua dung file hoac dung cac trang da OCR."
                )
                continue
            row["cer"] = cer(text, compare_truth)
            row["wer"] = wer(text, compare_truth)
            stats["used_slice"] = bool(stats["used_slice"] or sliced)
        else:
            row["cer"] = None
            row["wer"] = None
            row["ground_truth_compare_length"] = 0
    return stats


def _aggregate_page_rows(engine_key: str, variant_name: str, page_rows: list[dict], elapsed_sec: float | None = None) -> dict:
    ok_rows = [row for row in page_rows if row.get("status") == "ok" and row.get("text")]
    error_rows = [row for row in page_rows if row.get("status") == "error"]
    skipped_rows = [row for row in page_rows if row.get("status") == "skipped"]

    if ok_rows:
        status = "ok"
    elif skipped_rows and not error_rows:
        status = "skipped"
    else:
        status = "error"

    text = "\n\n".join((row.get("text") or "").strip() for row in page_rows if (row.get("text") or "").strip())
    boxes = []
    for row in page_rows:
        page_no = row.get("page")
        for box in row.get("boxes") or []:
            if isinstance(box, dict):
                page_box = dict(box)
                page_box.setdefault("page", page_no)
                boxes.append(page_box)
            else:
                boxes.append(box)

    page_errors = [
        f"Trang {row.get('page')}: {row.get('error')}"
        for row in page_rows
        if row.get("status") in {"error", "skipped"} and row.get("error")
    ]
    error = "; ".join(page_errors)
    if status == "ok" and page_errors:
        error = f"Một số trang không chạy được: {error}"

    page_elapsed_total = sum(float(row.get("elapsed_sec") or 0) for row in page_rows)
    elapsed = float(elapsed_sec if elapsed_sec is not None else page_elapsed_total)
    quality_guard = analyze_ocr_text_quality(text)
    return {
        "engine": engine_key,
        "variant": variant_name,
        "status": status,
        "text": text,
        "boxes": boxes,
        "elapsed_sec": elapsed,
        "error": error,
        "quality_guard": quality_guard,
        "raw": {
            "page_count": len(page_rows),
            "ok_pages": len(ok_rows),
            "error_pages": len(error_rows),
            "skipped_pages": len(skipped_rows),
            "first_page_text": page_rows[0].get("text") if page_rows else "",
            "page_elapsed_total": page_elapsed_total,
            "wall_elapsed_sec": elapsed,
            "page_results": [
                {
                    "page": row.get("page"),
                    "status": row.get("status"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "text_len": len(row.get("text") or ""),
                    "boxes": len(row.get("boxes") or []),
                    "error": row.get("error") or "",
                    "note": (row.get("raw") or {}).get("note"),
                    "refine": (row.get("raw") or {}).get("refine"),
                    "paddle_device": (row.get("raw") or {}).get("paddle_device"),
                    "vietocr_device": (row.get("raw") or {}).get("vietocr_device"),
                    "vietocr_model": (row.get("raw") or {}).get("vietocr_model"),
                }
                for row in page_rows
            ],
        },
    }


GPU_ENGINE_KEYS = {"easyocr", "paddle_vietocr", "paddleocr_vl"}


def _workers_for_engine(engine_key: str) -> int:
    if engine_key == "tesseract":
        return OCR_TESSERACT_WORKERS
    if engine_key in GPU_ENGINE_KEYS:
        return OCR_GPU_WORKERS
    return 1


def _run_single_page(engine_cls, page_no: int, img_path: Path, variant_name: str) -> dict:
    engine = engine_cls()
    row = engine.run(img_path, variant=variant_name).to_dict()
    row["page"] = page_no
    row["image"] = str(img_path)
    return row


def _run_engine_on_pages(engine_cls, engine_key: str, variant_name: str, image_paths: list[Path]) -> dict:
    start = time.perf_counter()
    workers = min(max(1, _workers_for_engine(engine_key)), max(1, len(image_paths)))
    items = list(enumerate(image_paths, start=1))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"ocr_{engine_key}") as executor:
            page_rows = list(
                executor.map(
                    lambda item: _run_single_page(engine_cls, item[0], item[1], variant_name),
                    items,
                )
            )
    else:
        page_rows = [_run_single_page(engine_cls, page_no, img_path, variant_name) for page_no, img_path in items]
    return _aggregate_page_rows(engine_key, variant_name, page_rows, time.perf_counter() - start)


def _text_input_result_row(text: str, elapsed_sec: float, warning: str | None = None) -> dict:
    status = "ok" if text.strip() else "error"
    row = {
        "engine": "text_input",
        "variant": "direct_text",
        "status": status,
        "elapsed_sec": elapsed_sec,
        "text": text,
        "boxes": [],
        "cer": None,
        "wer": None,
        "ground_truth_compare_length": 0,
        "raw": {
            "page_count": 1,
            "first_page_text": text,
            "note": "Text source uploaded directly; OCR image step skipped.",
        },
        "quality_guard": analyze_ocr_text_quality(text),
    }
    if warning:
        row["warning"] = warning
    if status != "ok":
        row["error"] = warning or "File text khong co noi dung de nhan dien."
    return row


def _text_input_postprocess(text: str, reader: str, warning: str | None = None) -> dict:
    fields = extract_fields_rule_based(text).to_dict()
    fields["confidence_note"] = "text_input_rule"
    note = "File đầu vào là text/Word nên hệ thống bỏ qua OCR ảnh và trích xuất trường trực tiếp từ nội dung text."
    if warning:
        note = f"{note} Cảnh báo đọc file: {warning}"
    return {
        "mode": "text_input_direct",
        "extractor": "text_input",
        "note": note,
        "model_configured": False,
        "model_ran": False,
        "reader": reader,
        "fields_source": "text_input_rule",
        "fields": fields,
        "labels_preview_rows": [],
    }


def read_ground_truth_upload(upload: UploadFile | None, job_dir: Path) -> tuple[str, dict | None]:
    """Lưu và đọc file ground truth upload.

    Trả về (text, metadata). Nếu không upload file thì trả về text rỗng.
    """
    if not upload or not upload.filename:
        return "", None

    gt_dir = job_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    safe_name = Path(upload.filename).name
    gt_path = gt_dir / safe_name
    data = upload.file.read()
    gt_path.write_bytes(data)

    text, warning, reader = _read_ground_truth_file(gt_path, data=data)
    return text, {
        "filename": safe_name,
        "relative_path": str(gt_path.relative_to(job_dir)),
        "size_bytes": len(data),
        "text_length": len(text),
        "warning": warning,
        "reader": reader,
    }

    suffix = gt_path.suffix.lower()
    warning = None
    if suffix == ".docx":
        text = _extract_docx_text(gt_path)
    elif suffix == ".doc":
        text = ""
        warning = "File .doc nhị phân cũ chưa được đọc trực tiếp. Hãy chuyển sang .docx hoặc xuất .txt để tính CER/WER."
    else:
        text = _decode_uploaded_text(data)

    # Chuẩn hóa nhẹ để so CER/WER ổn hơn, không sửa nội dung có nghĩa.
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text, {
        "filename": safe_name,
        "relative_path": str(gt_path.relative_to(job_dir)),
        "size_bytes": len(data),
        "text_length": len(text),
        "warning": warning,
    }


def _ocr_display_label(row: dict, quality_guard: dict | None = None) -> str:
    engine = row.get("engine") or ""
    variant = row.get("variant") or ""
    if engine == "text_input":
        return "text input / direct_text"
    if engine != "paddle_vietocr":
        return f"{engine} / {variant}"

    raw = row.get("raw") or {}
    page_results = raw.get("page_results") or []
    refine_notes = " ".join(
        str(item.get("refine") or "")
        for item in page_results
        if isinstance(item, dict)
    ).lower()
    if not refine_notes:
        refine_notes = str(raw.get("refine") or "").lower()

    guard = quality_guard or row.get("quality_guard") or analyze_ocr_text_quality(row.get("text") or "")
    if "refined each crop" in refine_notes:
        engine_label = "paddle+vietocr"
    elif "skipped" in refine_notes or "unavailable" in refine_notes or "timeout" in refine_notes:
        engine_label = "paddleocr (vietocr fallback)"
    elif guard.get("diacritic_loss") or guard.get("severe_diacritic_loss"):
        engine_label = "paddleocr (mất dấu)"
    else:
        engine_label = "paddleocr"
    return f"{engine_label} / {variant}"


def build_comparison_summary(result_rows: list[dict]) -> dict:
    """Tạo dữ liệu dashboard biểu đồ để hiển thị ngay trên giao diện.

    Không dùng thư viện chart bên ngoài để demo chạy offline ổn định trong Docker/local.
    Biểu đồ được render bằng HTML/CSS thuần ở template.
    """
    rows = []
    for r in result_rows:
        cer_pct = _pct(r.get("cer"))
        wer_pct = _pct(r.get("wer"))
        elapsed = _round(r.get("elapsed_sec") or 0, 3)
        text_len = len(r.get("text") or "")
        avg_conf = _round(_avg_confidence(r), 2)
        quality_guard = r.get("quality_guard") or analyze_ocr_text_quality(r.get("text") or "")
        label = _ocr_display_label(r, quality_guard)
        quality_score = None
        quality_source = None
        if cer_pct is not None and wer_pct is not None:
            # CER quan trọng hơn WER vì tiếng Việt hay sai dấu/ký tự.
            quality_score = max(0.0, round(100 - (0.65 * cer_pct + 0.35 * wer_pct), 2))
            quality_source = "CER/WER"
        elif cer_pct is not None:
            quality_score = max(0.0, round(100 - cer_pct, 2))
            quality_source = "CER"
        elif wer_pct is not None:
            quality_score = max(0.0, round(100 - wer_pct, 2))
            quality_source = "WER"

        rows.append({
            "engine": r.get("engine"),
            "variant": r.get("variant"),
            "label": label,
            "status": r.get("status"),
            "cer_pct": cer_pct,
            "wer_pct": wer_pct,
            "elapsed_sec": elapsed,
            "text_len": text_len,
            "avg_confidence": avg_conf,
            "quality_score": quality_score,
            "quality_source": quality_source,
            "quality_guard": quality_guard,
            "selection_score": ocr_selection_score(r),
            "ground_truth_metric_warning": r.get("ground_truth_metric_warning"),
            "error": r.get("error"),
        })

    ok_rows = [x for x in rows if x["status"] == "ok"]
    skipped_rows = [x for x in rows if x["status"] == "skipped"]
    error_rows = [x for x in rows if x["status"] == "error"]

    def min_by(key: str):
        candidates = [x for x in ok_rows if x.get(key) is not None]
        return min(candidates, key=lambda x: x[key]) if candidates else None

    def max_by(key: str):
        candidates = [x for x in ok_rows if x.get(key) is not None]
        return max(candidates, key=lambda x: x[key]) if candidates else None

    max_time = max([x["elapsed_sec"] or 0 for x in rows] or [0])
    max_text = max([x["text_len"] or 0 for x in rows] or [0])

    for x in rows:
        x["cer_bar"] = 0 if x["cer_pct"] is None else min(100, max(0, x["cer_pct"]))
        x["wer_bar"] = 0 if x["wer_pct"] is None else min(100, max(0, x["wer_pct"]))
        x["time_bar"] = 0 if not max_time else round((x["elapsed_sec"] or 0) / max_time * 100, 2)
        x["text_bar"] = 0 if not max_text else round((x["text_len"] or 0) / max_text * 100, 2)
        if x["quality_score"] is None and x["status"] == "ok" and x["text_len"] > 0:
            length_score = 0 if not max_text else round((x["text_len"] or 0) / max_text * 100, 2)
            vi_score = float((x.get("quality_guard") or {}).get("vietnamese_quality_score") or 0)
            if x["avg_confidence"] is not None:
                x["quality_score"] = round((0.45 * x["avg_confidence"]) + (0.20 * length_score) + (0.35 * vi_score), 2)
                x["quality_source"] = "confidence + text length + Vietnamese guard"
            else:
                x["quality_score"] = round((0.55 * length_score) + (0.45 * vi_score), 2)
                x["quality_source"] = "relative text length + Vietnamese guard"
            penalty = float((x.get("quality_guard") or {}).get("quality_penalty") or 0)
            if penalty:
                x["quality_score"] = max(0.0, round(x["quality_score"] - penalty, 2))
                if (x.get("quality_guard") or {}).get("severe_diacritic_loss"):
                    x["quality_score"] = min(x["quality_score"], 60.0)
                elif (x.get("quality_guard") or {}).get("diacritic_loss"):
                    x["quality_score"] = min(x["quality_score"], 78.0)
        x["quality_bar"] = 0 if x["quality_score"] is None else min(100, max(0, x["quality_score"]))
        x["confidence_bar"] = 0 if x["avg_confidence"] is None else min(100, max(0, x["avg_confidence"]))

    # So sánh tác động tiền xử lý OpenCV trên cùng engine.
    preprocessing_effect = []
    by_engine = {}
    for x in rows:
        by_engine.setdefault(x["engine"], {})[x["variant"]] = x
    for engine, variants in by_engine.items():
        raw = variants.get("raw")
        pre = variants.get("opencv_preprocessed")
        if not raw or not pre:
            continue
        cer_delta = None
        wer_delta = None
        time_delta = None
        if raw.get("cer_pct") is not None and pre.get("cer_pct") is not None:
            cer_delta = round(pre["cer_pct"] - raw["cer_pct"], 2)
        if raw.get("wer_pct") is not None and pre.get("wer_pct") is not None:
            wer_delta = round(pre["wer_pct"] - raw["wer_pct"], 2)
        if raw.get("elapsed_sec") is not None and pre.get("elapsed_sec") is not None:
            time_delta = round(pre["elapsed_sec"] - raw["elapsed_sec"], 3)
        raw_score = raw.get("quality_score")
        if raw_score is None:
            raw_score = raw.get("selection_score")
        pre_score = pre.get("quality_score")
        if pre_score is None:
            pre_score = pre.get("selection_score")
        if pre.get("status") != "ok":
            recommended_variant = "raw"
        elif raw.get("status") != "ok":
            recommended_variant = "opencv_preprocessed"
        elif pre_score is not None and raw_score is not None and pre_score > raw_score + 0.1:
            recommended_variant = "opencv_preprocessed"
        else:
            recommended_variant = "raw"
        preprocessing_effect.append({
            "engine": engine,
            "raw_status": raw.get("status"),
            "pre_status": pre.get("status"),
            "cer_delta": cer_delta,
            "wer_delta": wer_delta,
            "time_delta": time_delta,
            "text_len_delta": (pre.get("text_len") or 0) - (raw.get("text_len") or 0),
            "recommended_variant": recommended_variant,
            "recommendation_note": (
                "Use OpenCV for this engine."
                if recommended_variant == "opencv_preprocessed"
                else "Use raw for this engine; OpenCV is only a fallback."
            ),
        })

    engine_status = []
    for engine, variants in by_engine.items():
        values = list(variants.values())
        engine_status.append({
            "engine": engine,
            "ok": sum(1 for x in values if x["status"] == "ok"),
            "skipped": sum(1 for x in values if x["status"] == "skipped"),
            "error": sum(1 for x in values if x["status"] == "error"),
            "total": len(values),
        })

    return {
        "total_runs": len(rows),
        "ok_runs": len(ok_rows),
        "error_runs": len(error_rows),
        "skipped_runs": len(skipped_rows),
        "issue_runs": len(error_rows) + len(skipped_rows),
        "has_ground_truth": any(x.get("cer_pct") is not None or x.get("wer_pct") is not None for x in rows),
        "best_cer": min_by("cer_pct"),
        "best_wer": min_by("wer_pct"),
        "fastest": min_by("elapsed_sec"),
        "best_quality": max_by("quality_score"),
        "best_confidence": max_by("avg_confidence"),
        "rows": rows,
        "preprocessing_effect": preprocessing_effect,
        "engine_status": engine_status,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "engines": [
                {
                    "key": "tesseract",
                    "label": "Tesseract OCR",
                    "hint": "Nhanh, nen dung de test truoc.",
                    "checked": True,
                    "badge": "nhanh",
                },
                {
                    "key": "easyocr",
                    "label": "EasyOCR",
                    "hint": "Cham hon Tesseract, chi bat khi can doi chieu.",
                    "checked": False,
                    "badge": "vua",
                },
                {
                    "key": "paddle_vietocr",
                    "label": "PaddleOCR (+ VietOCR thử nghiệm)",
                    "hint": "De rung dau voi tieng Viet; VietOCR refine co timeout va rat cham.",
                    "checked": False,
                    "badge": "thu nghiem",
                },
                {
                    "key": "paddleocr_vl",
                    "label": "PaddleOCR-VL",
                    "hint": "Rat cham tren CPU, chi dung khi can minh hoa layout/VL.",
                    "checked": False,
                    "badge": "rat cham",
                },
            ],
            "sample_pdf": "/download_sample/sample_cong_van_scan.pdf",
            "sample_img": "/download_sample/sample_cong_van_scan.png",
            "sample_truth": "/download_sample/ground_truth_text/sample_cong_van.txt",
            "device_summary": device_summary(),
        },
    )


@app.get("/download_sample/{file_path:path}")
def download_sample(file_path: str):
    p = ROOT_DIR / "demo_samples" / file_path
    if not p.exists() or not p.is_file():
        return HTMLResponse("Không tìm thấy file mẫu", status_code=404)
    return FileResponse(p)


@app.get("/jobs/{job_id}/{filename:path}")
def download_job_file(job_id: str, filename: str):
    p = JOBS_DIR / job_id / filename
    if not p.exists() or not p.is_file():
        return HTMLResponse("Không tìm thấy file", status_code=404)
    return FileResponse(p)


@app.get("/storage/{filename:path}")
def download_storage_file(filename: str):
    root = STORAGE_DIR.resolve()
    p = (root / filename).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return HTMLResponse("Không tìm thấy file lưu trữ", status_code=404)
    if not p.exists() or not p.is_file():
        return HTMLResponse("Không tìm thấy file lưu trữ", status_code=404)
    return FileResponse(p)


def _format_storage_size(size_bytes: int | None) -> str:
    if not isinstance(size_bytes, int):
        return "-"
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size_bytes} B"


def _storage_url(rel_path: str | None) -> str:
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    return "/storage/" + quote(rel, safe="/")


def _storage_abs(rel_path: str | None) -> str:
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    if not rel:
        return ""
    return str((STORAGE_DIR / rel).resolve())


def _decorate_storage_file(file_info: dict | None) -> dict | None:
    if not isinstance(file_info, dict):
        return None
    decorated = dict(file_info)
    rel_path = decorated.get("path")
    if rel_path:
        decorated["url"] = _storage_url(rel_path)
        decorated["absolute_path"] = _storage_abs(rel_path)
        p = STORAGE_DIR / str(rel_path)
        if p.exists() and p.is_file():
            decorated["size_bytes"] = p.stat().st_size
    decorated["size_label"] = _format_storage_size(decorated.get("size_bytes"))
    sha = decorated.get("sha256") or ""
    decorated["sha256_short"] = sha[:12] if sha else "-"
    return decorated


def _decorate_archive_output(output: dict) -> dict:
    decorated = dict(output)
    text_path = decorated.get("text_path")
    json_path = decorated.get("json_path")
    word_path = decorated.get("word_path")
    if text_path:
        decorated["text_url"] = _storage_url(text_path)
        decorated["text_absolute_path"] = _storage_abs(text_path)
    if json_path:
        decorated["json_url"] = _storage_url(json_path)
        decorated["json_absolute_path"] = _storage_abs(json_path)
    if word_path:
        word_file = decorated.get("word_file") if isinstance(decorated.get("word_file"), dict) else {"path": word_path}
        decorated["word_file"] = _decorate_storage_file(word_file)
        decorated["word_url"] = _storage_url(word_path)
        decorated["word_absolute_path"] = _storage_abs(word_path)
        decorated["word_filename"] = decorated["word_file"]["filename"] if decorated.get("word_file") else Path(str(word_path)).name
        decorated["word_size_label"] = decorated["word_file"]["size_label"] if decorated.get("word_file") else "-"
    return decorated


def _decorate_archive_exports(exports: dict | None) -> dict:
    if not isinstance(exports, dict):
        return {}
    decorated = {}
    for key, value in exports.items():
        if isinstance(value, str) and value:
            decorated[key] = {
                "path": value,
                "url": _storage_url(value),
                "absolute_path": _storage_abs(value),
            }
        else:
            decorated[key] = value
    return decorated


def _hydrate_document_archive(report: dict) -> bool:
    archive = report.get("document_archive")
    if not isinstance(archive, dict):
        return False

    manifest_rel = archive.get("manifest_path")
    if not manifest_rel:
        return False
    manifest_file = STORAGE_DIR / str(manifest_rel)
    if not manifest_file.exists() or not manifest_file.is_file():
        return False

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if ensure_archive_word_outputs(manifest_file):
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

    hydrated = dict(archive)
    hydrated["manifest"] = manifest
    hydrated["document_id"] = manifest.get("document_id") or hydrated.get("document_id")
    hydrated["status"] = manifest.get("status") or hydrated.get("status")
    hydrated["created_at"] = manifest.get("created_at") or hydrated.get("created_at")
    hydrated["source_filename"] = manifest.get("source_filename") or hydrated.get("source_filename")
    hydrated["page_count"] = manifest.get("page_count") or hydrated.get("page_count")
    hydrated["storage_root"] = manifest.get("storage_root") or hydrated.get("storage_root")
    hydrated["storage_root_absolute"] = _storage_abs(hydrated.get("storage_root"))
    hydrated["manifest_url"] = _storage_url(manifest_rel)
    hydrated["index_url"] = _storage_url(hydrated.get("index_path"))

    hydrated["original_file"] = _decorate_storage_file(manifest.get("original_file"))
    hydrated["ground_truth"] = _decorate_storage_file(manifest.get("ground_truth"))

    raw_pages = [_decorate_storage_file(item) for item in ((manifest.get("pages") or {}).get("raw") or [])]
    raw_pages = [item for item in raw_pages if item]
    opencv_pages = [_decorate_storage_file(item) for item in ((manifest.get("pages") or {}).get("opencv_preprocessed") or [])]
    opencv_pages = [item for item in opencv_pages if item]
    opencv_by_page = {item.get("page"): item for item in opencv_pages}
    hydrated["pages"] = {
        "raw": raw_pages,
        "opencv_preprocessed": opencv_pages,
    }
    hydrated["preview_pages"] = [
        {
            "page": raw.get("page"),
            "raw": raw,
            "opencv": opencv_by_page.get(raw.get("page")),
        }
        for raw in raw_pages
    ]

    outputs = [_decorate_archive_output(item) for item in (manifest.get("ocr_outputs") or [])]
    hydrated["ocr_outputs"] = outputs
    hydrated["ocr_output_count"] = len(outputs)
    hydrated["exports"] = _decorate_archive_exports(manifest.get("exports"))
    hydrated["workflow_manifest_path"] = manifest.get("workflow_manifest_path") or hydrated.get("workflow_manifest_path")
    hydrated["workflow_manifest_url"] = _storage_url(hydrated.get("workflow_manifest_path"))
    hydrated["extracted_fields_path"] = manifest.get("extracted_fields_path") or hydrated.get("extracted_fields_path")
    hydrated["extracted_fields_url"] = _storage_url(hydrated.get("extracted_fields_path"))

    changed = hydrated != archive
    report["document_archive"] = hydrated
    return changed


def _manifest_rel_path(path: Path) -> str:
    return path.relative_to(STORAGE_DIR).as_posix()


def _find_document_manifest(document_id: str) -> Path | None:
    safe_id = Path(document_id).name
    index_path = STORAGE_DIR / "documents" / "index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("document_id") == safe_id and row.get("manifest_path"):
                candidate = STORAGE_DIR / str(row["manifest_path"])
                if candidate.exists() and candidate.is_file():
                    return candidate

    matches = list((STORAGE_DIR / "documents").glob(f"*/*/{safe_id}/manifest.json"))
    return matches[0] if matches else None


def _archive_from_manifest_path(manifest_path: Path) -> dict | None:
    if not manifest_path.exists() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    job_id = manifest.get("job_id")
    report_path = JOBS_DIR / str(job_id) / "report.json" if job_id else None
    if report_path and report_path.exists():
        try:
            latest_report = json.loads(report_path.read_text(encoding="utf-8"))
            sync_archive_metadata(manifest_path, latest_report)
        except (OSError, json.JSONDecodeError):
            pass
    report = {
        "document_archive": {
            "manifest_path": _manifest_rel_path(manifest_path),
            "index_path": "documents/index.jsonl",
        }
    }
    _hydrate_document_archive(report)
    archive = report["document_archive"]
    return archive if archive.get("manifest") else None


def _read_document_archive_rows() -> list[dict]:
    index_path = STORAGE_DIR / "documents" / "index.jsonl"
    rows: list[dict] = []
    seen: set[str] = set()

    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            manifest_rel = item.get("manifest_path")
            manifest_path = STORAGE_DIR / str(manifest_rel) if manifest_rel else None
            archive = _archive_from_manifest_path(manifest_path) if manifest_path else None
            if archive:
                item = {**item, **_document_row_from_archive(archive)}
            else:
                item["detail_url"] = f"/documents/{item.get('document_id', '')}"
                item["storage_root_absolute"] = _storage_abs(item.get("storage_root"))
            if item.get("document_id"):
                seen.add(item["document_id"])
                rows.append(item)

    for manifest_path in (STORAGE_DIR / "documents").glob("*/*/*/manifest.json"):
        archive = _archive_from_manifest_path(manifest_path)
        if not archive or archive.get("document_id") in seen:
            continue
        rows.append(_document_row_from_archive(archive))

    rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return rows


def _document_row_from_archive(archive: dict) -> dict:
    manifest = archive.get("manifest") or {}
    fields = {}
    extracted_path = archive.get("extracted_fields_path")
    if extracted_path:
        try:
            fields = json.loads((STORAGE_DIR / extracted_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            fields = {}
    workflow = {}
    workflow_path = archive.get("workflow_manifest_path")
    if workflow_path:
        try:
            workflow = json.loads((STORAGE_DIR / workflow_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            workflow = {}
    original = archive.get("original_file") or {}
    return {
        "document_id": archive.get("document_id"),
        "job_id": manifest.get("job_id"),
        "created_at": archive.get("created_at"),
        "source_filename": archive.get("source_filename"),
        "page_count": archive.get("page_count"),
        "status": archive.get("status"),
        "storage_root": archive.get("storage_root"),
        "storage_root_absolute": archive.get("storage_root_absolute"),
        "document_number": workflow.get("document_number") or fields.get("so_ky_hieu") or "",
        "issued_date": workflow.get("issued_date") or fields.get("ngay_ban_hanh") or "",
        "document_type": workflow.get("document_type") or fields.get("loai_van_ban") or "",
        "issuing_agency": workflow.get("issuing_agency") or fields.get("co_quan_ban_hanh") or "",
        "subject": workflow.get("subject") or fields.get("trich_yeu") or "",
        "ocr_output_count": archive.get("ocr_output_count") or 0,
        "original_size_label": original.get("size_label") or "-",
        "detail_url": f"/documents/{archive.get('document_id')}",
        "manifest_url": archive.get("manifest_url"),
    }


def _document_archive_stats(rows: list[dict]) -> dict:
    total_pages = sum(int(row.get("page_count") or 0) for row in rows)
    total_outputs = sum(int(row.get("ocr_output_count") or 0) for row in rows)
    complete = sum(1 for row in rows if row.get("status") == "scanned")
    return {
        "total_documents": len(rows),
        "total_pages": total_pages,
        "total_outputs": total_outputs,
        "complete_documents": complete,
        "storage_root": str((STORAGE_DIR / "documents").resolve()),
    }


def _read_storage_json(rel_path: str | None) -> dict:
    if not rel_path:
        return {}
    try:
        data = json.loads((STORAGE_DIR / rel_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _select_postprocess_row(report: dict) -> dict | None:
    rows = [
        row
        for row in (report.get("results") or [])
        if row.get("engine") != "glm_ocr" and row.get("status") == "ok" and row.get("text")
    ]
    if not rows:
        return None

    max_text_len = max(len(row.get("text") or "") for row in rows)
    return sorted(rows, key=lambda row: ocr_selection_score(row, max_text_len), reverse=True)[0]


def _layout_image_rel(report: dict, row: dict) -> str | None:
    key = "preprocessed_image" if row.get("variant") == "opencv_preprocessed" else "raw_image"
    return report.get(key)


def _layout_boxes_from_row(row: dict) -> list:
    from app.ocr_engines.base import OCRBox

    raw_boxes = [box for box in (row.get("boxes") or []) if isinstance(box, dict)]
    has_page_numbers = any(box.get("page") is not None for box in raw_boxes)
    if has_page_numbers:
        raw_boxes = [box for box in raw_boxes if int(box.get("page") or 1) == 1]
    return [
        OCRBox(text=box.get("text", ""), confidence=box.get("confidence"), bbox=box.get("bbox"))
        for box in raw_boxes
        if box.get("text") and box.get("bbox")
    ]


def _annotate_layout_source(layout_result: dict | None, row: dict | None) -> dict | None:
    if not layout_result or not row:
        return layout_result
    guard = row.get("quality_guard") or {}
    layout_result["ocr_source"] = {
        "engine": row.get("engine"),
        "variant": row.get("variant"),
        "page": 1,
        "note": "Token LayoutLMv3 lấy từ OCR/bbox trang 1 của engine được chọn.",
    }
    layout_result["ocr_source_quality"] = {
        "diacritic_loss": guard.get("diacritic_loss"),
        "severe_diacritic_loss": guard.get("severe_diacritic_loss"),
        "vietnamese_quality_score": guard.get("vietnamese_quality_score"),
    }
    layout_result["labels_preview_note"] = (
        f"Token preview lấy từ {row.get('engine')} / {row.get('variant')} trang 1. "
        "Nếu OCR nguồn rụng chữ thì token LayoutLMv3 cũng rụng theo; field cuối vẫn được hậu kiểm bằng best-OCR/rule."
    )
    return layout_result


def _layout_postprocess_needs_refresh(report: dict) -> bool:
    if report.get("input_mode") == "text":
        return False
    layout_result = report.get("layoutlmv3_postprocess")
    if not layout_result:
        return True
    best = _select_postprocess_row(report)
    if not best:
        return False
    source = layout_result.get("ocr_source") or {}
    if source.get("engine") != best.get("engine") or source.get("variant") != best.get("variant"):
        return True
    if layout_result.get("model_ran") and layout_result.get("labels_preview") and not layout_result.get("labels_preview_rows"):
        return True
    return False


def _merge_postprocess_fields_from_rows(layout_result: dict | None, rows: list[dict]) -> dict | None:
    return stabilize_layout_fields_from_rows(layout_result, rows)

    if not layout_result or not isinstance(layout_result.get("fields"), dict):
        return layout_result

    # Header fields are often better in classic OCR rows than in OCR-VL markdown,
    # because OCR-VL may intentionally ignore header blocks.
    field_priority = {"easyocr": 0, "paddleocr_vl": 1, "tesseract": 2, "paddle_vietocr": 3}
    ok_rows = [
        row
        for row in rows
        if row.get("engine") != "glm_ocr" and row.get("status") == "ok" and row.get("text")
    ]
    ok_rows = sorted(
        ok_rows,
        key=lambda row: (
            field_priority.get(row.get("engine"), 99),
            0 if row.get("variant") == "raw" else 1,
        ),
    )

    candidates: list[dict] = []
    for row in ok_rows:
        try:
            source_text = ((row.get("raw") or {}).get("first_page_text") or row.get("text", ""))
            fields = extract_fields_rule_based(source_text).to_dict()
        except Exception:
            continue
        fields["_engine"] = row.get("engine")
        fields["_variant"] = row.get("variant")
        candidates.append(fields)

    if not candidates:
        return layout_result

    fields = dict(layout_result["fields"])

    def key_text(value: str) -> str:
        value = unicodedata.normalize("NFD", str(value or ""))
        value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
        return re.sub(r"\s+", " ", value).lower().strip()

    def source_bonus(candidate: dict) -> float:
        engine_bonus = {"easyocr": 70, "paddleocr_vl": 55, "tesseract": 35, "paddle_vietocr": 10}
        variant_bonus = 8 if candidate.get("_variant") == "raw" else 0
        return engine_bonus.get(candidate.get("_engine"), 0) + variant_bonus

    def field_score(key: str, value: str, candidate: dict | None = None) -> float:
        value = (value or "").strip()
        if not value:
            return -1
        text_key = key_text(value)
        bonus = source_bonus(candidate or {})
        letters = [ch for ch in value if ch.isalpha()]
        vietnamese = sum(1 for ch in value if ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
        diacritic_score = (80 * vietnamese / len(letters)) if letters else 0

        if key == "so_ky_hieu":
            if not re.search(r"\d{1,5}/\d{4}/", value):
                return -1
            return 200 + bonus + (45 if "Đ" in value.upper() else 0) - (20 if "ND-" in value.upper() else 0)

        if key == "ngay_ban_hanh":
            return 200 + bonus if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value) else 20 + bonus

        if key == "co_quan_ban_hanh":
            if len(value) > 120 or re.search(r"\d{2,}/\d{4}", value) or text_key.startswith(("so:", "so ")):
                return -1
            score = min(len(value), 80) + bonus + diacritic_score
            if any(x in text_key for x in ("chinh phu", "ngan hang nha nuoc", "quoc hoi", "bo ", "uy ban", "ubnd")):
                score += 180
            if any(x in text_key for x in ("cong hoa xa hoi", "doc lap", "hanh phuc")):
                score -= 320
            if text_key in {"viet nam", "ha noi"}:
                score -= 200
            return score

        if key == "loai_van_ban":
            if len(value) > 40:
                return -1
            return 120 + bonus + diacritic_score

        if key == "trich_yeu":
            if len(value) < 18 or len(value) > 360:
                return -1
            score = min(len(value), 180) + bonus + diacritic_score
            if text_key.startswith(("dieu ", "chuong ", "can cu", "theo de nghi", "noi nhan")):
                score -= 500
            if re.match(r"^\d+\.", text_key):
                score -= 350
            if any(marker in text_key for marker in (" chuong ii", " dieu 4", " dieu 5", " phu luc", "noi nhan")):
                score -= 350
            if re.search(r"\b\d+\.", value) or any(
                marker in text_key
                for marker in ("giao nhan giay to", "theo quy dinh cua", "co quan, to chuc", "ca nhan")
            ):
                score -= 360
            if any(candidate_type in text_key for candidate_type in ("quy dinh", "ve viec", "phan cap", "sua doi", "bo sung")):
                score += 160
            return score

        if key in {"noi_gui", "noi_nhan"} and len(value) > 220:
            return -1
        return len(value) + bonus + diacritic_score

    def best_candidate_value(key: str) -> str:
        scored = [
            ((candidate.get(key) or "").strip(), field_score(key, (candidate.get(key) or "").strip(), candidate))
            for candidate in candidates
        ]
        scored = [(value, score) for value, score in scored if score >= 0]
        if not scored:
            return ""
        return max(scored, key=lambda item: item[1])[0]

    for key in ("so_ky_hieu", "ngay_ban_hanh", "co_quan_ban_hanh"):
        best = best_candidate_value(key)
        if best and field_score(key, best) >= field_score(key, (fields.get(key) or "").strip()):
            fields[key] = best

    doc_type_values = [(candidate.get("loai_van_ban") or "").strip() for candidate in candidates]
    doc_type_values = [value for value in doc_type_values if field_score("loai_van_ban", value) >= 0]
    if doc_type_values:
        counts = Counter(doc_type_values)
        fields["loai_van_ban"] = max(
            doc_type_values,
            key=lambda value: (counts[value], field_score("loai_van_ban", value)),
        )

    for key in ("trich_yeu", "noi_gui", "noi_nhan"):
        if key == "trich_yeu":
            best = best_candidate_value(key)
            if best and field_score(key, best) > field_score(key, (fields.get(key) or "").strip()):
                fields[key] = best
            continue
        if key in {"noi_gui", "noi_nhan"} and fields.get(key) and len(str(fields.get(key))) > 220:
            fields[key] = ""
        if fields.get(key):
            continue
        for candidate in candidates:
            value = (candidate.get(key) or "").strip()
            if value and field_score(key, value, candidate) >= 0:
                fields[key] = value
                break

    fields["confidence_note"] = "layoutlmv3+best_ocr_rule"
    layout_result["fields"] = fields
    layout_result["fields_source"] = "layoutlmv3+best_ocr_rule"
    return layout_result


def _fill_fields_from_upload_name(layout_result: dict | None, uploaded_file: str | None) -> dict | None:
    if not layout_result or not isinstance(layout_result.get("fields"), dict) or not uploaded_file:
        return layout_result
    fields = layout_result["fields"]
    if fields.get("so_ky_hieu"):
        return layout_result
    stem = Path(uploaded_file).stem
    match = re.search(r"\b(\d{1,5})[-_/](\d{4})[-_/]([A-Za-zĐđ]+(?:[-_][A-Za-zĐđ0-9]+)+)\b", stem)
    if match:
        suffix = match.group(3).replace("_", "-").upper()
        fields["so_ky_hieu"] = f"{match.group(1)}/{match.group(2)}/{suffix}"
    return layout_result


def _refresh_layout_postprocess(job_dir: Path, report: dict) -> bool:
    if report.get("input_mode") == "text":
        return False
    best = _select_postprocess_row(report)
    if not best:
        return False

    image_rel = _layout_image_rel(report, best)
    if not image_rel:
        return False
    image_path = job_dir / Path(image_rel)

    boxes = _layout_boxes_from_row(best)
    postprocess_text = ((best.get("raw") or {}).get("first_page_text") or best.get("text", ""))
    layout_result = layoutlmv3_postprocess(image_path, postprocess_text, boxes)
    layout_result = _merge_postprocess_fields_from_rows(layout_result, report.get("results") or [])
    layout_result = _fill_fields_from_upload_name(layout_result, report.get("uploaded_file"))
    layout_result = _annotate_layout_source(layout_result, best)
    changed = layout_result != report.get("layoutlmv3_postprocess")
    report["layoutlmv3_postprocess"] = layout_result
    report["best_engine"] = {"engine": best.get("engine"), "variant": best.get("variant")}
    return changed


def _refresh_document_archive(job_dir: Path, report: dict) -> bool:
    current = report.get("document_archive") or {}
    manifest_path = current.get("manifest_path")
    if manifest_path and (STORAGE_DIR / manifest_path).exists():
        sync_archive_metadata(STORAGE_DIR / manifest_path, report)
        return _hydrate_document_archive(report)
    if not report.get("results"):
        return False
    report["document_archive"] = archive_scan(job_dir, report)
    _hydrate_document_archive(report)
    return True


def render_result(request: Request, job_id: str, job_dir: Path, report: dict):
    result_rows = [row for row in (report.get("results") or []) if row.get("engine") != "glm_ocr"]
    report["results"] = result_rows
    _hydrate_document_archive(report)
    layout_result = report.get("layoutlmv3_postprocess")
    comparison_summary = build_comparison_summary(result_rows)
    report["comparison_summary"] = comparison_summary
    raw_image = Path(report.get("raw_image") or "")
    pre_img = Path(report.get("preprocessed_image") or "")
    raw_image_url = f"/jobs/{job_id}/{raw_image.as_posix()}" if str(raw_image) not in {"", "."} else ""
    pre_image_url = f"/jobs/{job_id}/{pre_img.as_posix()}" if str(pre_img) not in {"", "."} else ""

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "job_id": job_id,
            "report": report,
            "results": result_rows,
            "layout_result": layout_result,
            "summary": comparison_summary,
            "raw_image_url": raw_image_url,
            "pre_image_url": pre_image_url,
            "csv_url": f"/jobs/{job_id}/benchmark_results.csv",
            "json_url": f"/jobs/{job_id}/report.json",
            "summary_url": f"/jobs/{job_id}/comparison_summary.json",
        },
    )


@app.get("/documents", response_class=HTMLResponse)
def document_archive_home(request: Request):
    rows = _read_document_archive_rows()
    return templates.TemplateResponse(
        "documents.html",
        {
            "request": request,
            "documents": rows,
            "stats": _document_archive_stats(rows),
        },
    )


@app.get("/documents/{document_id}", response_class=HTMLResponse)
def document_archive_detail(request: Request, document_id: str):
    manifest_path = _find_document_manifest(document_id)
    if not manifest_path:
        return HTMLResponse("Không tìm thấy hồ sơ lưu trữ", status_code=404)
    archive = _archive_from_manifest_path(manifest_path)
    if not archive:
        return HTMLResponse("Không đọc được manifest hồ sơ", status_code=500)
    row = _document_row_from_archive(archive)
    fields = _read_storage_json(archive.get("extracted_fields_path"))
    workflow = _read_storage_json(archive.get("workflow_manifest_path"))
    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "archive": archive,
            "document": row,
            "fields": fields,
            "workflow": workflow,
            "storage_root": str((STORAGE_DIR / "documents").resolve()),
        },
    )


def _refresh_report_ground_truth(job_dir: Path, report: dict) -> bool:
    meta = report.get("ground_truth_file")
    if not meta or not meta.get("relative_path"):
        return False
    metric_rows = [row for row in (report.get("results") or []) if row.get("status") == "ok" and row.get("text")]
    has_saved_text = bool(report.get("ground_truth_text_length") or meta.get("text_length"))
    metrics_current = bool(metric_rows) and all(
        row.get("ground_truth_metric_version") == GROUND_TRUTH_METRIC_VERSION
        and (
            (row.get("cer") is not None and row.get("wer") is not None)
            or bool(row.get("ground_truth_metric_warning"))
        )
        for row in metric_rows
    )
    if has_saved_text and metrics_current:
        report["comparison_summary"] = build_comparison_summary(report.get("results") or [])
        return False

    gt_path = job_dir / meta["relative_path"]
    if not gt_path.exists():
        return False

    text, warning, reader = _read_ground_truth_file(gt_path)
    report["ground_truth_text_length"] = len(text)
    meta["text_length"] = len(text)
    meta["warning"] = warning
    meta["reader"] = reader

    metric_stats = _apply_ground_truth_metrics(report.get("results") or [], text)
    if metric_stats.get("used_slice"):
        extra = " Da dung phan dau file text chuan de khop voi do dai text OCR."
        meta["warning"] = _append_warning(meta.get("warning"), extra)
    if metric_stats.get("skipped_mismatch"):
        meta["warning"] = _append_warning(meta.get("warning"), metric_stats.get("warning"))
    report["comparison_summary"] = build_comparison_summary(report.get("results") or [])
    return bool(text)


@app.get("/reports/{job_id}", response_class=HTMLResponse)
def view_report(request: Request, job_id: str):
    safe_job_id = Path(job_id).name
    job_dir = JOBS_DIR / safe_job_id
    report_path = job_dir / "report.json"
    if not report_path.exists() or not report_path.is_file():
        return HTMLResponse("Không tìm thấy report", status_code=404)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    changed = _refresh_report_ground_truth(job_dir, report)
    refresh_layout = (request.query_params.get("refresh_layout") or "").strip().lower() in {"1", "true", "yes"}
    if refresh_layout or _layout_postprocess_needs_refresh(report):
        changed = _refresh_layout_postprocess(job_dir, report) or changed
    if changed:
        save_json(report_path, report)
        save_json(job_dir / "comparison_summary.json", report["comparison_summary"])
        save_results_csv(job_dir / "benchmark_results.csv", report.get("results") or [])
    archive_changed = _refresh_document_archive(job_dir, report)
    if archive_changed:
        save_json(report_path, report)
    return render_result(request, safe_job_id, job_dir, report)


@app.post("/run", response_class=HTMLResponse)
def run_demo(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF/ảnh công văn")],
    engines: Annotated[list[str] | None, Form()] = None,
    compare_raw_preprocessed: Annotated[str | None, Form()] = None,
    ground_truth_file: Annotated[UploadFile | None, File(description="File text chuẩn .txt/.md/.csv/.json/.docx")] = None,
    ground_truth: Annotated[str, Form()] = "",
):
    selected = engines or ["tesseract"]

    job_id, job_dir = new_job_dir()

    upload_dir = job_dir / "upload"
    upload_dir.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "input.bin").name
    uploaded_path = upload_dir / safe_name
    with uploaded_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ưu tiên file ground truth upload. Ô nhập text chỉ là fallback nếu cần dán nhanh.
    uploaded_ground_truth, ground_truth_meta = read_ground_truth_upload(ground_truth_file, job_dir)
    if uploaded_ground_truth:
        ground_truth = uploaded_ground_truth

    if _is_text_input_file(uploaded_path):
        start = time.perf_counter()
        text, source_warning, reader = _read_ground_truth_file(uploaded_path)
        result_rows = [_text_input_result_row(text, time.perf_counter() - start, source_warning)]
        ground_truth_metric_stats = _apply_ground_truth_metrics(result_rows, ground_truth)
        if ground_truth_meta and ground_truth_metric_stats.get("skipped_mismatch"):
            ground_truth_meta["warning"] = _append_warning(
                ground_truth_meta.get("warning"),
                ground_truth_metric_stats.get("warning"),
            )

        layout_result = _text_input_postprocess(text, reader, source_warning)
        layout_result = _merge_postprocess_fields_from_rows(layout_result, result_rows)
        layout_result = _fill_fields_from_upload_name(layout_result, str(uploaded_path.relative_to(job_dir)))
        comparison_summary = build_comparison_summary(result_rows)
        report = {
            "job_id": job_id,
            "input_mode": "text",
            "uploaded_file": str(uploaded_path.relative_to(job_dir)),
            "uploaded_text": {
                "filename": safe_name,
                "reader": reader,
                "text_length": len(text),
                "warning": source_warning,
            },
            "page_count": 1,
            "raw_images": [],
            "preprocessed_images": [],
            "raw_image": "",
            "preprocessed_image": "",
            "opencv_steps": ["text_input_no_ocr"],
            "opencv_steps_by_page": [],
            "runtime": {
                "opencv_workers": 0,
                "tesseract_workers": 0,
                "gpu_workers": 0,
            },
            "ground_truth_file": ground_truth_meta,
            "ground_truth_text_length": len(ground_truth.strip()),
            "results": result_rows,
            "comparison_summary": comparison_summary,
            "layoutlmv3_postprocess": layout_result,
            "best_engine": {"engine": "text_input", "variant": "direct_text"},
        }
        save_json(job_dir / "comparison_summary.json", comparison_summary)
        save_results_csv(job_dir / "benchmark_results.csv", result_rows)
        report["document_archive"] = archive_scan(job_dir, report)
        _hydrate_document_archive(report)
        save_json(job_dir / "report.json", report)
        return RedirectResponse(url=f"/reports/{job_id}", status_code=303)

    image_dir = job_dir / "images"
    raw_images = ensure_images(uploaded_path, image_dir)
    opencv_workers = min(max(1, OCR_OPENCV_WORKERS), max(1, len(raw_images)))
    if opencv_workers > 1:
        with ThreadPoolExecutor(max_workers=opencv_workers, thread_name_prefix="opencv") as executor:
            preprocessed_results = list(executor.map(lambda raw_page: preprocess_image(raw_page, image_dir), raw_images))
    else:
        preprocessed_results = [preprocess_image(raw_page, image_dir) for raw_page in raw_images]
    preprocessed_images = [item[0] for item in preprocessed_results]
    steps_by_page = [
        {"page": idx, "steps": item[1]}
        for idx, item in enumerate(preprocessed_results, start=1)
    ]
    steps = ["adaptive_safe_preprocess"]
    for _, page_steps in preprocessed_results:
        for step in page_steps:
            if step not in steps:
                steps.append(step)
    raw_image = raw_images[0]
    pre_img = preprocessed_images[0]

    variants: list[tuple[str, list[Path]]] = []
    if compare_raw_preprocessed:
        variants.append(("raw", raw_images))
    variants.append(("opencv_preprocessed", preprocessed_images))

    result_rows = []
    for engine_key in selected:
        engine_cls = ENGINE_REGISTRY.get(engine_key)
        if not engine_cls:
            continue
        for variant_name, image_paths in variants:
            result_rows.append(_run_engine_on_pages(engine_cls, engine_key, variant_name, image_paths))

    ground_truth_metric_stats = _apply_ground_truth_metrics(result_rows, ground_truth)
    if ground_truth_meta and ground_truth_metric_stats.get("used_slice"):
        extra = " Da dung phan dau file text chuan de khop voi do dai text OCR."
        ground_truth_meta["warning"] = _append_warning(ground_truth_meta.get("warning"), extra)
    if ground_truth_meta and ground_truth_metric_stats.get("skipped_mismatch"):
        ground_truth_meta["warning"] = _append_warning(
            ground_truth_meta.get("warning"),
            ground_truth_metric_stats.get("warning"),
        )

    # Chọn kết quả tốt nhất để hậu xử lý: ưu tiên CER thấp nếu có ground truth.
    # Khi không có ground truth, dùng quality guard để tránh chọn output rụng dấu
    # dù confidence nội bộ cao.
    ok_rows = [r for r in result_rows if r.get("status") == "ok" and r.get("text")]
    max_text_len = max([len(r.get("text") or "") for r in ok_rows] or [1])
    best = sorted(ok_rows, key=lambda r: ocr_selection_score(r, max_text_len), reverse=True)[0] if ok_rows else None

    layout_result = None
    if best:
        # LayoutLMv3 cần bbox, nên dùng image biến thể tương ứng.
        best_img = pre_img if best.get("variant") == "opencv_preprocessed" else raw_image
        boxes = _layout_boxes_from_row(best)
        postprocess_text = ((best.get("raw") or {}).get("first_page_text") or best.get("text", ""))
        layout_result = layoutlmv3_postprocess(best_img, postprocess_text, boxes)
        layout_result = _merge_postprocess_fields_from_rows(layout_result, result_rows)
        layout_result = _fill_fields_from_upload_name(layout_result, str(uploaded_path.relative_to(job_dir)))
        layout_result = _annotate_layout_source(layout_result, best)

    comparison_summary = build_comparison_summary(result_rows)

    report = {
        "job_id": job_id,
        "uploaded_file": str(uploaded_path.relative_to(job_dir)),
        "page_count": len(raw_images),
        "raw_images": [str(path.relative_to(job_dir)) for path in raw_images],
        "preprocessed_images": [str(path.relative_to(job_dir)) for path in preprocessed_images],
        "raw_image": str(raw_image.relative_to(job_dir)),
        "preprocessed_image": str(pre_img.relative_to(job_dir)),
        "opencv_steps": steps,
        "opencv_steps_by_page": steps_by_page,
        "runtime": {
            "opencv_workers": OCR_OPENCV_WORKERS,
            "tesseract_workers": OCR_TESSERACT_WORKERS,
            "gpu_workers": OCR_GPU_WORKERS,
        },
        "ground_truth_file": ground_truth_meta,
        "ground_truth_text_length": len(ground_truth.strip()),
        "results": result_rows,
        "comparison_summary": comparison_summary,
        "layoutlmv3_postprocess": layout_result,
        "best_engine": {"engine": best.get("engine"), "variant": best.get("variant")} if best else None,
    }
    save_json(job_dir / "comparison_summary.json", comparison_summary)
    save_results_csv(job_dir / "benchmark_results.csv", result_rows)
    report["document_archive"] = archive_scan(job_dir, report)
    _hydrate_document_archive(report)
    save_json(job_dir / "report.json", report)

    return RedirectResponse(url=f"/reports/{job_id}", status_code=303)
