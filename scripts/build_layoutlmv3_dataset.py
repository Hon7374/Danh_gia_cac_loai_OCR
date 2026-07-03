from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import unicodedata
from difflib import SequenceMatcher
from statistics import median
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
TEXT_EXTS = {".txt"}
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
DOC_TYPE_PATTERNS = [
    (("nghi", "dinh"), "NGHỊ ĐỊNH"),
    (("nghi", "quyet"), "NGHỊ QUYẾT"),
    (("quyet", "dinh"), "QUYẾT ĐỊNH"),
    (("thong", "tu"), "THÔNG TƯ"),
    (("cong", "van"), "CÔNG VĂN"),
    (("luat",), "LUẬT"),
    (("ke", "hoach"), "KẾ HOẠCH"),
    (("bao", "cao"), "BÁO CÁO"),
    (("to", "trinh"), "TỜ TRÌNH"),
    (("thong", "bao"), "THÔNG BÁO"),
    (("giay", "moi"), "GIẤY MỜI"),
]
NATIONAL_HEADER_KEYS = (
    "conghoaxahoi",
    "chunghiavietnam",
    "doclap",
    "tudo",
    "hanhphuc",
)
BODY_START_KEYS = (
    "can",
    "cancu",
    "caucu",
    "chuong",
    "dieu",
    "theode",
    "theodenghi",
    "noinhan",
    "kinhgui",
    "quyetdinh",
    "thongdoc",
    "botruong",
    "chutich",
)
ISSUER_KEYWORDS = (
    "bo",
    "chinhphu",
    "quochoi",
    "uyban",
    "ubnd",
    "nganhang",
    "congan",
    "toaan",
    "kiemsat",
    "thanhtra",
    "vanphong",
    "cuc",
    "tongcuc",
    "vien",
)


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


def _line_text(line: dict[str, Any]) -> str:
    return " ".join(str(item["word"]) for item in line["items"]).strip()


def _line_key(line: dict[str, Any]) -> str:
    return _word_key(_line_text(line))


def _token_keys(line: dict[str, Any]) -> list[str]:
    return [_word_key(str(item["word"])) for item in line["items"]]


def _has_national_header(value: str) -> bool:
    key = _word_key(value)
    return any(part in key for part in NATIONAL_HEADER_KEYS)


def _has_issuer_keyword(value: str) -> bool:
    key = _word_key(value)
    return any(keyword in key for keyword in ISSUER_KEYWORDS)


def _uppercase_ratio(value: str) -> float:
    letters = [ch for ch in value if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def _looks_like_serial_value(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or "")).upper()
    return bool(re.search(r"\d{1,5}/\d{4}/[A-ZĐ0-9_.-]+", compact))


def _looks_like_serial_line(line: dict[str, Any]) -> bool:
    key = _line_key(line)
    return key.startswith(("so", "s0", "s6")) or any(_looks_like_serial_value(str(item["word"])) for item in line["items"])


def _is_body_start_line(line: dict[str, Any]) -> bool:
    key = _line_key(line)
    return any(key.startswith(prefix) for prefix in BODY_START_KEYS)


def _date_parts(value: str) -> tuple[str, str, str] | None:
    key = _word_key(value)
    match = re.search(r"ngay(\d{1,2}).{0,8}thang(\d{1,2}).{0,8}nam(\d{4})", key)
    if match:
        return match.group(1), match.group(2), match.group(3)
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def _date_candidate_matches_expected(candidate: str, expected: str) -> bool:
    expected_parts = _date_parts(expected)
    candidate_parts = _date_parts(candidate)
    if not expected_parts or not candidate_parts:
        return True
    expected_day, expected_month, expected_year = expected_parts
    day, month, year = candidate_parts
    if year != expected_year:
        return False
    return int(day) == int(expected_day) and int(month) == int(expected_month)


def build_layout_lines(words: list[str], boxes: list[list[int]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, (word, box) in enumerate(zip(words, boxes)):
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(coord) for coord in box]
        items.append(
            {
                "idx": idx,
                "word": word,
                "box": [x0, y0, x1, y1],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "xc": (x0 + x1) / 2,
                "yc": (y0 + y1) / 2,
                "h": max(1, y1 - y0),
            }
        )
    if not items:
        return []

    height = median([item["h"] for item in items])
    threshold = max(8.0, height * 0.75)
    lines: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda entry: (entry["yc"], entry["x0"])):
        if not lines or abs(item["yc"] - lines[-1]["yc"]) > threshold:
            lines.append({"items": [item], "yc": item["yc"]})
            continue
        lines[-1]["items"].append(item)
        lines[-1]["yc"] = sum(entry["yc"] for entry in lines[-1]["items"]) / len(lines[-1]["items"])

    for line in lines:
        line["items"].sort(key=lambda entry: entry["x0"])
        line["x0"] = min(item["x0"] for item in line["items"])
        line["x1"] = max(item["x1"] for item in line["items"])
        line["y0"] = min(item["y0"] for item in line["items"])
        line["y1"] = max(item["y1"] for item in line["items"])
        line["text"] = _line_text(line)
        line["key"] = _line_key(line)
    return lines


def _match_key(path_or_name: Path | str) -> str:
    stem = Path(path_or_name).stem if not isinstance(path_or_name, Path) else path_or_name.stem
    stem = re.sub(r"\([^)]*\)", " ", stem)
    stem = re.sub(r"(?i)(?:_?level\d+|_?signed|_?daky|_?da ky|_?final|_?ban phat hanh)$", " ", stem)
    return _word_key(stem)


def _doc_type_hint(source: Path) -> str:
    key = _word_key(source.stem)
    if "nghiquyet" in key or re.search(r"(?:^|[^a-z])nq(?:cp|tw|hdnd|ubtvqh)?", source.stem.lower()):
        return "nghiquyet"
    if "nghidinh" in key or "ndcp" in key:
        return "nghidinh"
    if "quyetdinh" in key or "qdttg" in key:
        return "quyetdinh"
    if "thongtu" in key or re.search(r"(?:^|[^a-z])tt(?:[^a-z]|$)", source.stem.lower()):
        return "thongtu"
    if "luat" in key:
        return "luat"
    return ""


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_source_files(source_dir: Path, recursive: bool, limit: int = 0) -> list[Path]:
    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    sources = [
        path
        for path in iterator
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in (IMAGE_EXTS | PDF_EXTS | SKIP_EXTS)
    ]
    sources = sorted(sources, key=lambda item: item.as_posix().lower())
    return sources[:limit] if limit and limit > 0 else sources


def iter_truth_files(source_dir: Path, truth_dir: Path | None, recursive: bool) -> list[Path]:
    roots = [truth_dir.resolve()] if truth_dir else [source_dir]
    files: list[Path] = []
    for root in roots:
        iterator = root.rglob("*") if recursive else root.iterdir()
        files.extend(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in TEXT_EXTS and not path.name.startswith(".")
        )
    return sorted(set(files), key=lambda item: item.as_posix().lower())


def match_truth_file(source: Path, truth_files: list[Path], min_score: float) -> tuple[Path | None, float]:
    if not truth_files:
        return None, 0.0
    source_key = _match_key(source)
    if not source_key:
        return None, 0.0

    best_path: Path | None = None
    best_score = 0.0
    for truth in truth_files:
        truth_key = _match_key(truth)
        if not truth_key:
            continue
        same_dir_bonus = 0.05 if truth.parent == source.parent else 0.0
        if source_key == truth_key:
            score = 1.0 + same_dir_bonus
        elif source_key in truth_key or truth_key in source_key:
            shorter = min(len(source_key), len(truth_key))
            longer = max(len(source_key), len(truth_key))
            score = 0.88 + same_dir_bonus + 0.08 * (shorter / max(1, longer))
        else:
            score = SequenceMatcher(None, source_key, truth_key).ratio() + same_dir_bonus
        if score > best_score:
            best_path = truth
            best_score = score
    if best_path and best_score >= min_score:
        return best_path, best_score
    return None, best_score


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
            page_limit = len(doc) if max_pages <= 0 else min(max_pages, len(doc))
            for page_index in range(page_limit):
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
        (("nghi", "quyet"), "loai_van_ban"),
        (("quyet", "dinh"), "loai_van_ban"),
        (("thong", "tu"), "loai_van_ban"),
        (("cong", "van"), "loai_van_ban"),
        (("luat",), "loai_van_ban"),
        (("nghidinh",), "loai_van_ban"),
        (("nghiquyet",), "loai_van_ban"),
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


def apply_indexes(labels: list[str], indexes: list[int], field: str) -> bool:
    indexes = sorted(idx for idx in indexes if 0 <= idx < len(labels) and labels[idx] == "O")
    if not indexes:
        return False
    for offset, idx in enumerate(indexes):
        labels[idx] = f"{'B' if offset == 0 else 'I'}-{field}"
    return True


def _line_index_span(line: dict[str, Any], start_pos: int = 0, end_pos: int | None = None) -> list[int]:
    items = line["items"][start_pos:end_pos]
    return [int(item["idx"]) for item in items]


def _find_phrase_in_keys(keys: list[str], phrase: tuple[str, ...]) -> tuple[int, int] | None:
    for start in range(0, len(keys) - len(phrase) + 1):
        if tuple(keys[start : start + len(phrase)]) == phrase:
            return start, start + len(phrase)
    return None


def _find_serial_indexes(lines: list[dict[str, Any]], stop_line: int | None = None) -> tuple[list[int], int | None]:
    serial_re = re.compile(r"\d{1,5}/\d{4}/[A-ZĐ0-9_.-]+", flags=re.I)
    limit = min(stop_line if stop_line is not None and stop_line > 0 else 40, 40, len(lines))
    for line_idx, line in enumerate(lines[:limit]):
        line_key = _line_key(line)
        if _is_body_start_line(line):
            break
        if not (line_key.startswith(("so", "s0", "s6")) or any("/" in str(item["word"]) for item in line["items"])):
            continue
        indexes: list[int] = []
        for pos, item in enumerate(line["items"]):
            value = re.sub(r"\s+", "", str(item["word"]))
            if serial_re.search(value):
                indexes.append(int(item["idx"]))
                return indexes, line_idx
            key = _word_key(value)
            if key in {"so", "s0", "s6"}:
                continue
            if pos > 0 and any(ch.isdigit() for ch in value) and ("/" in value or "-" in value):
                indexes.append(int(item["idx"]))
        if indexes:
            return indexes, line_idx
    return [], None


def _find_date_indexes(lines: list[dict[str, Any]], expected_date: str = "") -> tuple[list[int], int | None]:
    for line_idx, line in enumerate(lines[:12]):
        if _is_body_start_line(line):
            break
        keys = _token_keys(line)
        if "ngay" not in keys and not any(re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", str(item["word"])) for item in line["items"]):
            continue
        if _line_key(line).startswith(("luat", "nghidinh", "thongtuso")):
            continue
        start = next((idx for idx, key in enumerate(keys) if key == "ngay"), None)
        if start is not None:
            end = None
            for pos in range(start, min(len(keys), start + 12)):
                if re.fullmatch(r"\d{4}", keys[pos]) or (keys[pos].isdigit() and len(keys[pos]) == 4):
                    end = pos + 1
                    break
            if end and any(key == "thang" for key in keys[start:end]) and any(key == "nam" for key in keys[start:end]):
                candidate = " ".join(str(item["word"]) for item in line["items"][start:end])
                if _date_candidate_matches_expected(candidate, expected_date):
                    return _line_index_span(line, start, end), line_idx
                continue
        for pos, item in enumerate(line["items"]):
            if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", str(item["word"])):
                if _date_candidate_matches_expected(str(item["word"]), expected_date):
                    return [int(item["idx"])], line_idx
    return [], None


def _find_issuer_indexes(lines: list[dict[str, Any]], stop_line: int | None) -> list[int]:
    limit = min(stop_line if stop_line is not None and stop_line > 0 else 10, len(lines))
    indexes: list[int] = []
    for line in lines[:limit]:
        text = _line_text(line)
        key = _line_key(line)
        if not key or _looks_like_serial_line(line):
            continue
        if any(part in key for part in NATIONAL_HEADER_KEYS) and not _has_issuer_keyword(text):
            continue

        items = line["items"]
        keys = _token_keys(line)
        national_start = None
        for pos in range(len(keys) - 1):
            if keys[pos] == "cong" and keys[pos + 1] == "hoa":
                national_start = pos
                break
        candidate_items = items[:national_start] if national_start is not None else items
        candidate_text = " ".join(str(item["word"]) for item in candidate_items).strip()
        candidate_key = _word_key(candidate_text)
        if not candidate_key:
            continue
        if _has_national_header(candidate_text):
            continue
        if _has_issuer_keyword(candidate_text) or _uppercase_ratio(candidate_text) >= 0.55:
            indexes.extend(int(item["idx"]) for item in candidate_items)
    return indexes[:36]


def _find_doc_type_indexes(
    lines: list[dict[str, Any]],
    serial_line: int | None,
    date_line: int | None,
    hint: str = "",
) -> tuple[list[int], int | None, int | None]:
    start_line = 0
    if serial_line is not None:
        start_line = max(start_line, serial_line + 1)
    if date_line is not None:
        start_line = max(start_line, date_line + 1)

    best: tuple[float, list[int], int, int] | None = None
    for line_idx, line in enumerate(lines[:80]):
        if line_idx < start_line:
            continue
        if _is_body_start_line(line):
            break
        if _has_national_header(_line_text(line)) or _looks_like_serial_line(line):
            continue
        keys = _token_keys(line)
        for phrase, _label in DOC_TYPE_PATTERNS:
            phrase_key = "".join(phrase)
            if hint and phrase_key != hint:
                continue
            found = _find_phrase_in_keys(keys, phrase)
            if not found:
                continue
            start, end = found
            before = "".join(keys[:start])
            if before and (len(before) > 8 or any(ch.isdigit() for ch in before)):
                continue
            line_word_count = sum(1 for key in keys if key)
            if phrase == ("luat",) and (start > 1 or line_word_count > 6):
                continue
            if phrase in {( "ke", "hoach"), ("bao", "cao"), ("thong", "bao")} and start > 1 and line_word_count > len(phrase) + 4:
                continue
            score = 100 - line_idx
            if start == 0:
                score += 30
            if line_word_count <= len(phrase) + 4:
                score += 20
            if _uppercase_ratio(_line_text(line)) >= 0.55:
                score += 10
            indexes = _line_index_span(line, start, end)
            if not best or score > best[0]:
                best = (score, indexes, line_idx, end)
    if not best:
        return [], None, None
    return best[1], best[2], best[3]


def _find_subject_indexes(
    lines: list[dict[str, Any]],
    doc_type_line: int | None,
    doc_type_end_pos: int | None,
) -> list[int]:
    if doc_type_line is None:
        for line in lines[:90]:
            key = _line_key(line)
            if any(marker in key for marker in ("vv", "veviec", "trichyeu")):
                keys = _token_keys(line)
                start = 0
                for pos, token_key in enumerate(keys):
                    if token_key in {"vv", "veviec", "trichyeu"}:
                        start = min(pos + 1, len(keys))
                        break
                indexes = _line_index_span(line, start)
                return indexes or (_line_index_span(line) if len(keys) <= 24 else [])
        return []

    indexes: list[int] = []
    title_line_count = 0
    for line_idx in range(doc_type_line, min(len(lines), doc_type_line + 10)):
        line = lines[line_idx]
        keys = _token_keys(line)
        if not keys:
            continue
        start = doc_type_end_pos or 0 if line_idx == doc_type_line else 0
        line_key_after_start = _word_key(" ".join(str(item["word"]) for item in line["items"][start:]))
        if not line_key_after_start:
            continue
        if line_idx > doc_type_line and _is_body_start_line(line):
            break
        if title_line_count and (
            "luat" in line_key_after_start[:80]
            or "nghidinh" in line_key_after_start[:80]
            or "nghidink" in line_key_after_start[:80]
        ):
            break
        if any(line_key_after_start.startswith(prefix) for prefix in BODY_START_KEYS):
            break
        if _has_national_header(line_key_after_start) or _looks_like_serial_line(line):
            continue
        if line_idx == doc_type_line and start >= len(line["items"]):
            continue
        if len(line_key_after_start) <= 4:
            continue
        indexes.extend(_line_index_span(line, start))
        title_line_count += 1
        if len(indexes) >= 64 or title_line_count >= 3:
            break
    return indexes[:64]


def _find_recipient_indexes(lines: list[dict[str, Any]]) -> list[int]:
    for line_idx, line in enumerate(lines[:140]):
        key = _line_key(line)
        if "kinhgui" not in key and not key.startswith("noinhan"):
            continue
        keys = _token_keys(line)
        start = 0
        for pos in range(len(keys)):
            if keys[pos] in {"kinh", "gui", "kinhgui", "noinhan", "noi", "nhan"}:
                start = pos + 1
        indexes = _line_index_span(line, start)
        if indexes:
            return indexes[:40]
        if line_idx + 1 < len(lines):
            return _line_index_span(lines[line_idx + 1])[:40]
    return []


def _safe_exact_label_value(field: str, value: str) -> bool:
    value = str(value or "").strip()
    if not value:
        return False
    key = _word_key(value)
    if field == "co_quan_ban_hanh":
        return 4 <= len(key) <= 80 and not _has_national_header(value)
    if field == "loai_van_ban":
        return any(key == "".join(pattern) for pattern, _label in DOC_TYPE_PATTERNS)
    if field == "trich_yeu":
        if len(value) < 10 or len(value) > 320:
            return False
        if "ngay" in key[:60] and "thang" in key[:90] and "nam" in key[:110]:
            return False
        if any(key.startswith(prefix) for prefix in BODY_START_KEYS):
            return False
    if field in {"noi_gui", "noi_nhan"}:
        return len(value) <= 220
    return True


def fields_from_labels(words: list[str], labels: list[str]) -> dict[str, str]:
    fields: dict[str, list[str]] = {}
    active = ""
    for word, label in zip(words, labels):
        if label == "O":
            active = ""
            continue
        prefix, _, field = label.partition("-")
        if prefix == "B" or field != active:
            fields.setdefault(field, [])
        fields.setdefault(field, []).append(word)
        active = field
    return {field: " ".join(tokens).strip() for field, tokens in fields.items() if tokens}


def label_words(words: list[str], boxes: list[list[int]], fields: dict[str, str]) -> tuple[list[str], dict[str, bool], dict[str, str]]:
    labels = ["O"] * len(words)
    matched: dict[str, bool] = {}
    lines = build_layout_lines(words, boxes)

    date_indexes, date_line = _find_date_indexes(lines, str(fields.get("ngay_ban_hanh") or ""))
    doc_type_indexes, doc_type_line, doc_type_end_pos = _find_doc_type_indexes(
        lines,
        None,
        date_line,
        str(fields.get("_doc_type_hint") or ""),
    )
    serial_indexes, serial_line = _find_serial_indexes(lines, doc_type_line or date_line)

    header_boundaries = [line for line in (serial_line, date_line, doc_type_line) if line is not None and line > 0]
    issuer_stop_line = min(header_boundaries) if header_boundaries else None

    issuer_indexes = _find_issuer_indexes(lines, issuer_stop_line)
    matched["co_quan_ban_hanh"] = apply_indexes(labels, issuer_indexes, "co_quan_ban_hanh")

    matched["so_ky_hieu"] = apply_indexes(labels, serial_indexes, "so_ky_hieu")
    matched["ngay_ban_hanh"] = apply_indexes(labels, date_indexes, "ngay_ban_hanh")
    matched["loai_van_ban"] = apply_indexes(labels, doc_type_indexes, "loai_van_ban")

    subject_indexes = _find_subject_indexes(lines, doc_type_line, doc_type_end_pos)
    matched["trich_yeu"] = apply_indexes(labels, subject_indexes, "trich_yeu")

    recipient_indexes = _find_recipient_indexes(lines)
    matched["noi_nhan"] = apply_indexes(labels, recipient_indexes, "noi_nhan")
    matched["noi_gui"] = False

    # Exact-value matching remains a fallback for fields the structural labeler did not see.
    for field in FIELD_ORDER:
        if matched.get(field):
            continue
        value = str(fields.get(field) or "").strip()
        if not _safe_exact_label_value(field, value):
            matched.setdefault(field, False)
            continue
        span = find_date_span(words, value) if field == "ngay_ban_hanh" else find_span(words, value)
        matched[field] = bool(span and apply_label(labels, span, field))

    return labels, matched, fields_from_labels(words, labels)


def build_record(
    image_path: Path,
    dataset_dir: Path,
    engine: TesseractEngine,
    *,
    source: Path,
    truth_file: Path | None = None,
    truth_text: str = "",
    truth_match_score: float = 0.0,
    min_labeled_tokens: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result = engine.run(image_path, variant="layoutlmv3_dataset")
    status: dict[str, Any] = {
        "source_file": str(source),
        "truth_file": str(truth_file) if truth_file else "",
        "truth_match_score": round(truth_match_score, 4),
        "image": str(image_path),
        "ocr_status": result.status,
        "error": result.error,
    }
    if result.status != "ok" or not result.boxes:
        return None, status
    filtered = [box for box in result.boxes if box.text and box.bbox]
    words = [box.text for box in filtered]
    boxes = [box.bbox for box in filtered]
    ocr_fields = extract_fields_rule_based(result.text).to_dict()
    truth_fields = extract_fields_rule_based(truth_text).to_dict() if truth_text else {}
    fields = {
        field: str((truth_fields.get(field) or ocr_fields.get(field) or "")).strip()
        for field in FIELD_ORDER
    }
    fields["_doc_type_hint"] = _doc_type_hint(source)
    labels, matched, semantic_fields = label_words(words, boxes, fields)
    labeled_count = sum(label != "O" for label in labels)
    if labeled_count < min_labeled_tokens:
        status.update(
            {
                "word_count": len(words),
                "labeled_count": labeled_count,
                "fields": fields,
                "semantic_fields": semantic_fields,
                "ocr_fields": ocr_fields,
                "truth_fields": truth_fields,
                "matched_fields": matched,
                "skip_reason": "not_enough_labeled_tokens",
            }
        )
        return None, status

    record = {
        "image": image_path.relative_to(dataset_dir).as_posix(),
        "words": words,
        "boxes": boxes,
        "labels": labels,
        "source_file": str(source),
        "truth_file": str(truth_file) if truth_file else "",
    }
    status.update(
        {
            "word_count": len(words),
            "labeled_count": labeled_count,
            "fields": fields,
            "semantic_fields": semantic_fields,
            "ocr_fields": ocr_fields,
            "truth_fields": truth_fields,
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
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=int(OCR_PDF_DPI or 180))
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--truth-match-min-score", type=float, default=0.76)
    parser.add_argument("--min-labeled-tokens", type=int, default=1)
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
    sources = iter_source_files(source_dir, args.recursive, args.limit)
    truth_files = iter_truth_files(source_dir, args.truth_dir, args.recursive)

    for source in sources:
        suffix = source.suffix.lower()
        if suffix in SKIP_EXTS:
            skipped.append({"file": str(source), "reason": "DOC/DOCX cần convert sang PDF hoặc ảnh để có bbox layout"})
            continue
        if suffix not in IMAGE_EXTS | PDF_EXTS:
            skipped.append({"file": str(source), "reason": f"Định dạng không hỗ trợ: {suffix}"})
            continue
        try:
            truth_file, truth_score = match_truth_file(source, truth_files, args.truth_match_min_score)
            truth_text = _read_text(truth_file) if truth_file else ""
            for image_path in render_pages(source, rendered_dir, args.max_pages, args.dpi):
                record, status = build_record(
                    image_path,
                    dataset_dir,
                    engine,
                    source=source,
                    truth_file=truth_file,
                    truth_text=truth_text,
                    truth_match_score=truth_score,
                    min_labeled_tokens=max(0, args.min_labeled_tokens),
                )
                processed.append(status)
                if record:
                    records.append(record)
        except Exception as exc:
            skipped.append({"file": str(source), "reason": str(exc)})

    eval_records: list[dict[str, Any]] = []
    train_records = records
    if records and args.eval_ratio > 0:
        shuffled = records[:]
        random.Random(args.seed).shuffle(shuffled)
        eval_count = max(1, int(round(len(records) * args.eval_ratio)))
        eval_records = shuffled[:eval_count]
        train_records = shuffled[eval_count:] or shuffled

    write_jsonl(output_jsonl, train_records)
    write_jsonl(eval_jsonl, eval_records)
    label_counts: dict[str, int] = {}
    for record in records:
        for label in record["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    manifest = {
        "dataset_dir": str(dataset_dir),
        "source_dir": str(source_dir),
        "truth_dir": str((args.truth_dir or source_dir).resolve()),
        "rendered_dir": str(rendered_dir),
        "train_jsonl": str(output_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "source_count": len(sources),
        "truth_file_count": len(truth_files),
        "record_count": len(records),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "label_counts": dict(sorted(label_counts.items())),
        "truth_matched_count": sum(1 for item in processed if item.get("truth_file")),
        "labeled_page_count": sum(1 for item in processed if item.get("labeled_count", 0) >= args.min_labeled_tokens),
        "processed": processed,
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "source_count",
                    "truth_file_count",
                    "truth_matched_count",
                    "record_count",
                    "train_count",
                    "eval_count",
                    "label_counts",
                    "skipped",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
