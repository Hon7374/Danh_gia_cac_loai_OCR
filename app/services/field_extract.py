from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass
class ExtractedFields:
    so_ky_hieu: str = ""
    ngay_ban_hanh: str = ""
    trich_yeu: str = ""
    co_quan_ban_hanh: str = ""
    noi_gui: str = ""
    noi_nhan: str = ""
    loai_van_ban: str = ""
    confidence_note: str = "rule_based"

    def to_dict(self) -> dict:
        return asdict(self)


DOC_TYPES: tuple[tuple[str, str], ...] = (
    ("thong tu", "THÔNG TƯ"),
    ("cong van", "CÔNG VĂN"),
    ("quyet dinh", "QUYẾT ĐỊNH"),
    ("thong bao", "THÔNG BÁO"),
    ("ke hoach", "KẾ HOẠCH"),
    ("to trinh", "TỜ TRÌNH"),
    ("giay moi", "GIẤY MỜI"),
    ("bao cao", "BÁO CÁO"),
)

DOC_TYPES = (
    ("thong tu", "THÔNG TƯ"),
    ("cong van", "CÔNG VĂN"),
    ("quyet dinh", "QUYẾT ĐỊNH"),
    ("nghi dinh", "NGHỊ ĐỊNH"),
    ("ngh dinh", "NGHỊ ĐỊNH"),
    ("ngh dnh", "NGHỊ ĐỊNH"),
    ("thong bao", "THÔNG BÁO"),
    ("ke hoach", "KẾ HOẠCH"),
    ("to trinh", "TỜ TRÌNH"),
    ("giay moi", "GIẤY MỜI"),
    ("bao cao", "BÁO CÁO"),
)

TITLE_STOP_PREFIXES = (
    "can cu",
    "can c",
    "chuong",
    "dieu",
    "theo de nghi",
    "theo d nghi",
    "chinh phu ban hanh",
    "chinh ph ban hanh",
    "thong doc",
    "bo truong",
    "chu tich",
    "quyet dinh",
    "noi nhan",
    "kinh gui",
    "gio",
    "ngay",
    "thong bao",
    "tong dat",
    "tng dt",
    "giao nhan",
    "giao nhn",
)

BODY_START_PREFIXES = (
    "can cu",
    "can c",
    "chuong",
    "dieu ",
    "theo de nghi",
    "theo d nghi",
    "chinh phu ban hanh",
    "chinh ph ban hanh",
    "thong doc ban hanh",
    "bo truong ban hanh",
)

ISSUER_KEYWORDS = (
    "ngan hang",
    "bo ",
    "so ",
    "uy ban",
    "ubnd",
    "chinh phu",
    "quoc hoi",
    "cuc ",
    "tong cuc",
    "vien ",
    "cong an",
    "toa an",
    "kiem sat",
)

NATIONAL_HEADER_PARTS = (
    "cong hoa xa hoi",
    "chu nghia viet nam",
    "doc lap",
    "tu do",
    "hanh phuc",
)

SERIAL_RE = re.compile(
    r"\b\d{1,5}\s*/\s*\d{4}(?:\s*/\s*[A-ZĐa-zđ0-9_.-]+)+\b",
    flags=re.I,
)


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" \t\r\n#*•.-:")


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return value.strip(" \t\r\n#*•,.;:-")


def _key(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def _line_has_ministry_plan_name(line_key: str) -> bool:
    return "bo ke hoach" in line_key and ("dau tu" in line_key or "dau t" in line_key)


def _is_stamp_or_issuer_doc_type_context(line_key: str, needle: str) -> bool:
    if needle == "ke hoach" and _line_has_ministry_plan_name(line_key):
        return True
    if needle == "cong van" and ("cong van den" in line_key or "van den" in line_key):
        return True
    if any(part in line_key for part in ("van phong chinh phu", "cong thong tin", "cong thong tn")):
        return True
    return False


def _split_logical_lines(text: str) -> list[str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    markers = (
        r"Số\s*[:：]",
        r"Hà\s+Nội\s*,?\s*ngày",
        r"NGHỊ\s+ĐỊNH",
        r"NGHI\s+DINH",
        r"NGH\s+DINH",
        r"NGH\s+DNH",
        r"THÔNG\s+TƯ",
        r"THONG\s+TU",
        r"QUYẾT\s+ĐỊNH",
        r"QUYET\s+DINH",
        r"CÔNG\s+VĂN",
        r"CONG\s+VAN",
        r"THÔNG\s+BÁO",
        r"THONG\s+BAO",
        r"KẾ\s+HOẠCH",
        r"KE\s+HOACH",
        r"TỜ\s+TRÌNH",
        r"TO\s+TRINH",
        r"GIẤY\s+MỜI",
        r"GIAY\s+MOI",
        r"BÁO\s+CÁO",
        r"BAO\s+CAO",
        r"Căn\s+cứ",
        r"Can\s+cu",
        r"Theo\s+đề\s+nghị",
        r"Theo\s+de\s+nghi",
        r"Chương\s+[IVXLC]+",
        r"Chuong\s+[IVXLC]+",
        r"Điều\s+\d+",
        r"Dieu\s+\d+",
        r"Nơi\s+nhận",
        r"Noi\s+nhan",
    )
    marker_re = re.compile(rf"(?<!^)\s+(?=(?:{'|'.join(markers)}))", flags=re.I)
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.extend(part for part in marker_re.split(raw) if part)
    cleaned = [_clean_line(line) for line in lines if _clean_line(line)]
    merged: list[str] = []
    idx = 0
    while idx < len(cleaned):
        current = cleaned[idx]
        current_key = _key(current)
        if idx + 1 < len(cleaned):
            next_line = cleaned[idx + 1]
            next_key = _key(next_line)
            if current_key == "bo" and next_key.startswith("ke hoach") and "dau" in next_key:
                merged.append(f"{current} {next_line}")
                idx += 2
                continue
        merged.append(current)
        idx += 1
    return merged


def _looks_like_serial_header(line_key: str) -> bool:
    return bool(re.match(r"^(?:so|s0|s6|s)\b\s*[:：.-]?", line_key))


def _format_date(day: str, month: str, year: str) -> str:
    d = int(day)
    m = int(month)
    # OCR hay đọc "ngày 19" thành "ngày49" trên mẫu scan này.
    if 40 <= d <= 49:
        d -= 30
    if not (1 <= d <= 31 and 1 <= m <= 12):
        return ""
    return f"{d:02d}/{m:02d}/{year}"


def _extract_so_ky_hieu(lines: list[str]) -> str:
    for idx, line in enumerate(lines[:40]):
        line_key = _key(line)
        marker = re.search(r"\b(?:so|s0|s6|s)\b\s*[:：.-]", line_key)
        starts_with_marker = _looks_like_serial_header(line_key)
        is_legal_citation = any(x in line_key for x in ("luat ", "nghi dinh", "thong tu so"))
        if not (starts_with_marker or (marker and marker.start() <= 40)) or is_legal_citation:
            continue
        match = SERIAL_RE.search(line)
        if match:
            return re.sub(r"\s+", "", match.group(0)).upper().strip(".,;")
        number_match = re.search(r"\b(\d{1,5})\s*$", line)
        if number_match and idx + 1 < len(lines):
            next_match = re.match(r"\s*(\d{4}\s*/\s*[A-ZĐa-zđ0-9_.-]+(?:\s*/\s*[A-ZĐa-zđ0-9_.-]+)*)", lines[idx + 1])
            if next_match:
                suffix = re.sub(r"\s+", "", next_match.group(1)).upper().strip(".,;")
                return f"{number_match.group(1)}/{suffix}"

    # Fallback: prefer document-number patterns near the header and avoid legal citations.
    for line in lines[:25]:
        line_key = _key(line)
        if any(x in line_key for x in ("luat ", "nghi dinh", "thong tu so")):
            continue
        match = SERIAL_RE.search(line)
        if match:
            return re.sub(r"\s+", "", match.group(0)).upper().strip(".,;")
    return ""


def _extract_ngay_ban_hanh(lines: list[str]) -> str:
    for line in lines[:40]:
        line_key = _key(line)
        match = re.search(
            r"ngay\D{0,12}(\d{1,2})\D{0,20}thang\D{0,12}(\d{1,2})\D{0,20}nam\D{0,12}(\d{4})",
            line_key,
            flags=re.I,
        )
        if match:
            return _format_date(match.group(1), match.group(2), match.group(3))

    flat_key = "\n".join(_key(line) for line in lines[:40])
    match = re.search(
        r"ngay\D{0,12}(\d{1,2})\D{0,20}thang\D{0,12}(\d{1,2})\D{0,20}nam\D{0,12}(\d{4})",
        flat_key,
        flags=re.I,
    )
    if match:
        return _format_date(match.group(1), match.group(2), match.group(3))

    for line in lines[:15]:
        if _looks_like_serial_header(_key(line)):
            continue
        match = re.search(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})\b", line)
        if match:
            return _format_date(match.group(1), match.group(2), match.group(3))
    return ""


def _find_doc_type(lines: list[str]) -> tuple[str, int]:
    search_limit = min(len(lines), 80)
    for idx, line in enumerate(lines[:80]):
        line_key = _key(line)
        if idx > 0 and any(line_key.startswith(prefix) for prefix in BODY_START_PREFIXES):
            search_limit = idx
            break

    for idx, line in enumerate(lines[:search_limit]):
        line_key = _key(line)
        for needle, label in DOC_TYPES:
            if _is_stamp_or_issuer_doc_type_context(line_key, needle):
                continue
            if re.fullmatch(rf".{{0,12}}\b{re.escape(needle)}\b.{{0,12}}", line_key):
                return label, idx
    for idx, line in enumerate(lines[:search_limit]):
        line_key = _key(line)
        for needle, label in DOC_TYPES:
            if _is_stamp_or_issuer_doc_type_context(line_key, needle):
                continue
            match = re.search(rf"\b{re.escape(needle)}\b", line_key)
            if not match:
                continue
            before = line_key[: match.start()].strip(" :-.,;")
            if before and (len(before) > 8 or any(ch.isdigit() for ch in before)):
                continue
            if not line_key.startswith(needle) and len(line_key) > 120:
                continue
            if re.match(r"^\d+\b", before):
                continue
            return label, idx
    return "", -1


def _text_after_doc_type(line: str) -> str:
    words = line.split()
    key_words = _key(line).split()
    for needle, _label in DOC_TYPES:
        needle_words = needle.split()
        if key_words[: len(needle_words)] == needle_words:
            return " ".join(words[len(needle_words) :])
    return line


def _trim_title_stop(text: str) -> str:
    text_key = _key(text)
    stop_positions = [
        pos
        for needle in (
            "can cu",
            "can c",
            "kinh chuyen",
            "kinh chuyn",
            "kinh gui",
            "chuong i",
            "dieu 1",
            "theo de nghi",
            "theo d nghi",
            "chinh phu ban hanh",
            "chinh ph ban hanh",
            " thong bao,",
            " tng dt",
            " tong dat",
            " giao nhn",
            " giao nhan",
            " giy t",
            " h so",
            " 1.",
        )
        if (pos := text_key.find(needle)) >= 0
    ]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def _trim_subject_noise(text: str) -> str:
    text = _clean_value(_trim_title_stop(text))
    words = text.split()
    key_words = _key(text).split()
    subject_starts = (
        ("quy", "dinh"),
        ("ve", "viec"),
        ("sua", "doi"),
        ("bo", "sung"),
        ("ban", "hanh"),
    )
    for needle in subject_starts:
        n = len(needle)
        for idx in range(0, max(0, len(key_words) - n + 1)):
            if tuple(key_words[idx : idx + n]) == needle:
                return _clean_value(" ".join(words[idx:]))
    return text


def _extract_trich_yeu_from_label(lines: list[str]) -> str:
    label_patterns = (
        r"\bv/v\b\s*[:：-]?\s*(.+)?",
        r"\bve viec\b\s*[:：-]?\s*(.+)?",
        r"\btrich yeu\b\s*[:：-]?\s*(.+)?",
    )
    for idx, line in enumerate(lines[:120]):
        line_key = _key(line)
        if line_key.startswith(("can cu", "chuong ", "dieu ")):
            break
        for pattern in label_patterns:
            match = re.search(pattern, line_key, flags=re.I)
            if not match:
                continue
            tail = _clean_value(line[match.start(1) :]) if match.lastindex and match.group(1) else ""
            if tail and len(tail) >= 8:
                return tail
            if idx + 1 < len(lines):
                return _clean_value(lines[idx + 1])
    return ""


def _extract_trich_yeu_after_type(lines: list[str], type_idx: int) -> str:
    if type_idx < 0:
        return ""
    title_lines: list[str] = []
    for idx, line in enumerate(lines[type_idx : type_idx + 10]):
        if idx == 0:
            line = _trim_title_stop(_text_after_doc_type(line))
        line_key = _key(line)
        if not line_key:
            continue
        if line_key.startswith(("ngay", "gio")) or "cong van den" in line_key or "van den" in line_key:
            continue
        if any(line_key.startswith(prefix) for prefix in TITLE_STOP_PREFIXES):
            break
        if _looks_like_serial_header(line_key):
            continue
        if any(part in line_key for part in NATIONAL_HEADER_PARTS):
            continue
        if len(line_key) <= 3:
            continue
        title_lines.append(line.rstrip(";"))
        if len(" ".join(title_lines)) >= 320:
            break
    return _trim_subject_noise(" ".join(title_lines))


def _extract_co_quan_ban_hanh(lines: list[str]) -> str:
    for line in lines[:25]:
        if _line_has_ministry_plan_name(_key(line)):
            return "BỘ KẾ HOẠCH VÀ ĐẦU TƯ"

    for line in lines[:20]:
        line_key = _key(line)
        if (
            "ngan hang nha nuoc" in line_key
            or "ngan hang nha nuc" in line_key
            or "n hang nha nuoc" in line_key
            or "n hang nha nuc" in line_key
        ):
            return "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM"

    serial_idx = next((idx for idx, line in enumerate(lines[:15]) if _looks_like_serial_header(_key(line))), -1)
    doc_type_idx = _find_doc_type(lines)[1]
    stop_idx = serial_idx if serial_idx > 0 else min(8, len(lines))
    if doc_type_idx > 0:
        stop_idx = min(stop_idx, doc_type_idx)
    header_lines = lines[:stop_idx]
    selected: list[str] = []

    for line in header_lines:
        line_key = _key(line)
        if not line_key:
            continue
        if "cong hoa xa hoi" in line_key:
            prefix = re.split(r"\bC\S*NG\s+H\S*A\b|\bCONG\s+HOA\b", line, maxsplit=1, flags=re.I)[0]
            prefix_key = _key(prefix)
            if prefix_key and any(keyword in f"{prefix_key} " for keyword in ISSUER_KEYWORDS):
                return _clean_value(prefix).upper()
        if any(part in line_key for part in NATIONAL_HEADER_PARTS):
            continue
        if _looks_like_serial_header(line_key):
            continue
        has_org_keyword = any(keyword in f"{line_key} " for keyword in ISSUER_KEYWORDS)
        is_country_suffix = line_key == "viet nam" and selected and "viet nam" not in _key(" ".join(selected))
        if has_org_keyword or is_country_suffix:
            selected.append(line)

    if selected:
        return _clean_value(" ".join(selected[:3])).upper()

    for line in lines[:15]:
        line_key = _key(line)
        if "ngan hang nha nuoc viet nam" in line_key:
            return _clean_value(line).upper()

    for line in lines[:8]:
        alpha = [ch for ch in line if ch.isalpha()]
        if alpha and sum(ch.isupper() for ch in alpha) / len(alpha) >= 0.65:
            line_key = _key(line)
            if not any(part in line_key for part in NATIONAL_HEADER_PARTS) and not _looks_like_serial_header(line_key):
                return _clean_value(line).upper()
    return ""


def _extract_noi_nhan(lines: list[str]) -> str:
    type_label, type_idx = _find_doc_type(lines)
    is_normative = type_label in {"THÔNG TƯ", "NGHỊ ĐỊNH", "QUYẾT ĐỊNH"}
    if is_normative and type_idx > 0:
        lines = lines[:type_idx]
    else:
        lines = lines[:120]
    for idx, line in enumerate(lines):
        line_key = _key(line)
        if "kinh gui" in line_key or line_key.startswith("noi nhan"):
            parts = re.split(r"[:：-]", line, maxsplit=1)
            if len(parts) == 2 and _clean_value(parts[1]) and len(_clean_value(parts[1])) <= 220:
                return _clean_value(parts[1])
            if idx + 1 < len(lines):
                value = _clean_value(lines[idx + 1])
                if len(value) <= 220:
                    return value
    return ""


def extract_fields_rule_based(text: str) -> ExtractedFields:
    lines = _split_logical_lines(text)

    fields = ExtractedFields()
    if not lines:
        return fields

    fields.so_ky_hieu = _extract_so_ky_hieu(lines)
    fields.ngay_ban_hanh = _extract_ngay_ban_hanh(lines)
    fields.loai_van_ban, type_idx = _find_doc_type(lines)
    title_after_type = _extract_trich_yeu_after_type(lines, type_idx)
    fields.trich_yeu = title_after_type or _extract_trich_yeu_from_label(lines)
    fields.co_quan_ban_hanh = _extract_co_quan_ban_hanh(lines)
    fields.noi_nhan = _extract_noi_nhan(lines)

    return fields
