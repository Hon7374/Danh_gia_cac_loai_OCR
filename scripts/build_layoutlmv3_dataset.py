from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import OCR_MAX_IMAGE_SIDE, OCR_PDF_DPI
from app.ocr_engines.tesseract_engine import TesseractEngine
from app.services.field_extract import extract_fields_rule_based


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
SKIP_EXTS = {".doc", ".docx"}
FIELD_ORDER = [
    "so_ky_hieu",
    "ngay_ban_hanh",
    "loai_van_ban",
    "co_quan_ban_hanh",
    "trich_yeu",
    "noi_gui",
    "noi_nhan",
]


def _ascii_slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:90] or "document"


def _word_key(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _resize_for_ocr(img: Image.Image) -> Image.Image:
    max_side = max(0, int(OCR_MAX_IMAGE_SIDE or 0))
    if not max_side:
        return img
    width, height = img.size
    current_max = max(width, height)
    if current_max <= max_side:
        return img
    scale = max_side / current_max
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _save_page(img: Image.Image, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    img = _resize_for_ocr(img.convert("RGB"))
    img.save(out, optimize=True)
    return out


def render_pages(source: Path, out_dir: Path, max_pages: int, dpi: int) -> list[Path]:
    suffix = source.suffix.lower()
    slug = _ascii_slug(source.stem)
    if suffix in IMAGE_EXTS:
        out = out_dir / f"{slug}_page001.png"
        with Image.open(source) as img:
            return [_save_page(img, out)]
    if suffix in PDF_EXTS:
        pages: list[Path] = []
        zoom = dpi / 72
        with fitz.open(source) as doc:
            for page_index in range(min(max_pages, len(doc))):
                page = doc[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                out = out_dir / f"{slug}_page{page_index + 1:03d}.png"
                pages.append(_save_page(img, out))
        return pages
    return []


def find_span(words: list[str], value: str, max_window: int = 80) -> tuple[int, int] | None:
    target = _word_key(value)
    if not target:
        return None
    keys = [_word_key(word) for word in words]
    for start in range(len(keys)):
        if not keys[start]:
            continue
        combined = ""
        for end in range(start, min(len(keys), start + max_window)):
            combined += keys[end]
            if combined == target:
                return start, end + 1
            if len(target) >= 10 and target in combined:
                return start, end + 1
            if len(combined) > len(target) + 30 and target not in combined:
                break
    return None


def find_date_span(words: list[str], value: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value or "")
    if not match:
        return find_span(words, value, max_window=12)
    day, month, year = str(int(match.group(1))), str(int(match.group(2))), match.group(3)
    keys = [_word_key(word) for word in words]
    for start, key in enumerate(keys):
        if not key.startswith("ngay"):
            continue
        window = keys[start : start + 10]
        joined = " ".join(window)
        has_markers = any(marker in joined for marker in ("thang", "nam"))
        has_day = day in window or f"{int(day):02d}" in window
        has_month = month in window or f"{int(month):02d}" in window
        has_year = year in window or any(year in key for key in window)
        if has_markers and has_day and has_month and has_year:
            end = start + next((idx + 1 for idx, key in reversed(list(enumerate(window))) if year in key), len(window))
            return start, min(end, len(words))
    return None


def find_serial_span(words: list[str]) -> tuple[int, int] | None:
    serial_re = re.compile(r"^\d{1,5}/\d{4}/[a-z0-9d-]+$")
    for idx, word in enumerate(words[:120]):
        key = _word_key(word.replace("/", " "))
        slash_value = word.upper().replace(" ", "")
        if re.search(r"\d{1,5}/\d{4}/", slash_value):
            return idx, idx + 1
        if serial_re.match(key):
            return idx, idx + 1
    return None


def find_doc_type_span(words: list[str]) -> tuple[int, int, str] | None:
    keys = [_word_key(word) for word in words]
    patterns = [
        (("nghi", "dinh"), "loai_van_ban"),
        (("quyet", "dinh"), "loai_van_ban"),
        (("thong", "tu"), "loai_van_ban"),
        (("cong", "van"), "loai_van_ban"),
        (("nghidinh",), "loai_van_ban"),
        (("quyetdinh",), "loai_van_ban"),
    ]
    for idx in range(min(len(keys), 160)):
        for parts, field in patterns:
            if len(parts) == 1 and keys[idx] == parts[0]:
                return idx, idx + 1, field
            if len(parts) == 2 and idx + 1 < len(keys) and keys[idx] == parts[0] and keys[idx + 1] == parts[1]:
                return idx, idx + 2, field
    return None


def find_title_span(words: list[str], doc_type_span: tuple[int, int, str] | None) -> tuple[int, int] | None:
    if not doc_type_span:
        return None
    start = doc_type_span[1]
    keys = [_word_key(word) for word in words]
    stop_prefixes = ("cancu", "chuong", "dieu", "theo", "noinhan", "quyetdinh")
    end = min(len(words), start + 80)
    for idx in range(start, min(len(words), start + 90)):
        key = keys[idx]
        if idx > start + 4 and any(key.startswith(prefix) for prefix in stop_prefixes):
            end = idx
            break
    while start < end and not keys[start]:
        start += 1
    while end > start and not keys[end - 1]:
        end -= 1
    if end - start >= 3:
        return start, end
    return None


def apply_label(labels: list[str], span: tuple[int, int], field: str) -> bool:
    start, end = span
    if start < 0 or end <= start or end > len(labels):
        return False
    indexes = [idx for idx in range(start, end) if labels[idx] == "O"]
    if not indexes:
        return False
    for offset, idx in enumerate(indexes):
        labels[idx] = f"{'B' if offset == 0 else 'I'}-{field}"
    return True


def label_words(words: list[str], fields: dict[str, str]) -> tuple[list[str], dict[str, bool]]:
    labels = ["O"] * len(words)
    matched: dict[str, bool] = {}
    for field in FIELD_ORDER:
        value = str(fields.get(field) or "").strip()
        if not value:
            matched[field] = False
            continue
        span = find_date_span(words, value) if field == "ngay_ban_hanh" else find_span(words, value)
        matched[field] = bool(span and apply_label(labels, span, field))

    if not matched.get("so_ky_hieu"):
        span = find_serial_span(words)
        matched["so_ky_hieu"] = bool(span and apply_label(labels, span, "so_ky_hieu"))

    doc_type_span = find_doc_type_span(words)
    if not matched.get("loai_van_ban") and doc_type_span:
        matched["loai_van_ban"] = apply_label(labels, (doc_type_span[0], doc_type_span[1]), "loai_van_ban")

    if not matched.get("trich_yeu"):
        span = find_title_span(words, doc_type_span)
        matched["trich_yeu"] = bool(span and apply_label(labels, span, "trich_yeu"))
    return labels, matched


def build_record(image_path: Path, dataset_dir: Path, engine: TesseractEngine) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result = engine.run(image_path, variant="layoutlmv3_dataset")
    status: dict[str, Any] = {"image": str(image_path), "ocr_status": result.status, "error": result.error}
    if result.status != "ok" or not result.boxes:
        return None, status
    filtered = [box for box in result.boxes if box.text and box.bbox]
    words = [box.text for box in filtered]
    boxes = [box.bbox for box in filtered]
    fields = extract_fields_rule_based(result.text).to_dict()
    labels, matched = label_words(words, fields)
    record = {
        "image": image_path.relative_to(dataset_dir).as_posix(),
        "words": words,
        "boxes": boxes,
        "labels": labels,
    }
    status.update(
        {
            "word_count": len(words),
            "labeled_count": sum(label != "O" for label in labels),
            "fields": fields,
            "matched_fields": matched,
        }
    )
    return record, status


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weak LayoutLMv3 JSONL dataset from PDF/image files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_template/layoutlmv3_training"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=int(OCR_PDF_DPI or 180))
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--clean-rendered", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    source_dir = (args.source_dir or dataset_dir / "pages").resolve()
    rendered_dir = dataset_dir / "rendered_pages"
    output_jsonl = args.output_jsonl or dataset_dir / "train.jsonl"
    eval_jsonl = args.eval_jsonl or dataset_dir / "eval.jsonl"
    manifest_path = dataset_dir / "build_manifest.json"

    if args.clean_rendered and rendered_dir.exists():
        shutil.rmtree(rendered_dir)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    engine = TesseractEngine()
    sources = [path for path in sorted(source_dir.iterdir()) if path.is_file() and not path.name.startswith(".")]

    for source in sources:
        suffix = source.suffix.lower()
        if suffix in SKIP_EXTS:
            skipped.append({"file": str(source), "reason": "DOC/DOCX cần convert sang PDF hoặc ảnh để có bbox layout"})
            continue
        if suffix not in IMAGE_EXTS | PDF_EXTS:
            skipped.append({"file": str(source), "reason": f"Định dạng không hỗ trợ: {suffix}"})
            continue
        try:
            for image_path in render_pages(source, rendered_dir, max(1, args.max_pages), args.dpi):
                record, status = build_record(image_path, dataset_dir, engine)
                status["source_file"] = str(source)
                processed.append(status)
                if record:
                    records.append(record)
        except Exception as exc:
            skipped.append({"file": str(source), "reason": str(exc)})

    eval_records: list[dict[str, Any]] = []
    train_records = records
    if records and args.eval_ratio > 0:
        eval_count = max(1, int(round(len(records) * args.eval_ratio)))
        eval_records = records[-eval_count:]
        train_records = records[:-eval_count] or records

    write_jsonl(output_jsonl, train_records)
    write_jsonl(eval_jsonl, eval_records)
    manifest = {
        "dataset_dir": str(dataset_dir),
        "source_dir": str(source_dir),
        "rendered_dir": str(rendered_dir),
        "train_jsonl": str(output_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "record_count": len(records),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "processed": processed,
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("record_count", "train_count", "eval_count", "skipped")}, ensure_ascii=False, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
