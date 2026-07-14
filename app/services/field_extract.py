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
    ("nghi quyet", "NGHỊ QUYẾT"),
    ("chi thi", "CHỈ THỊ"),
    ("phap lenh", "PHÁP LỆNH"),
    ("huong dan", "HƯỚNG DẪN"),
    ("quy che", "QUY CHẾ"),
    ("de an", "ĐỀ ÁN"),
    ("luat", "LUẬT"),
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
    "kinh chuyen",
    "kinh chuyn",
    "gio",
    "ngay",
    "thong bao",
    "nghi dinh so",
    "luat ",
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
    r"(?<![/／∕⁄\d])\b\d{1,6}\s*[/／∕⁄]\s*(?:\d{4}\s*[/／∕⁄]\s*)?[A-ZĐa-zđ][A-ZĐa-zđ0-9_.-]*\b",
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
            body_positions = [
                position
                for marker in (" can cu", " can c ", " dieu 1", " chuong ")
                if (position := line_key.find(marker)) >= 0
            ]
            before_body = not body_positions or match.start() < min(body_positions)
            flattened_header = bool(
                idx <= 3
                and before_body
                and match.start() <= 600
                and (
                    _looks_like_serial_header(line_key)
                    or re.search(r"\bngay\D{0,12}\d{1,2}\D{0,20}thang\b", line_key)
                    or any(part in line_key for part in NATIONAL_HEADER_PARTS)
                )
            )
            if flattened_header:
                return label, idx
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
        for index in range(0, max(0, len(key_words) - len(needle_words) + 1)):
            if key_words[index : index + len(needle_words)] == needle_words:
                return " ".join(words[index + len(needle_words) :])
    return line


def _trim_title_stop(text: str) -> str:
    text_key = _key(text)
    stop_positions: list[int] = []
    # Only cut at structural body markers. Generic phrases such as "quy định",
    # "bổ sung" or "căn cứ theo nhu cầu" can legitimately occur inside a
    # title and must never be treated as truncation anchors.
    for pattern in (
        r"(?:^|[.;])\s*(?:can cu|can c)\s+(?:luat|nghi dinh|thong tu|hien phap)\b",
        r"\bcan\s+(?!cu\s+(?:theo|nhu)\b)[a-z]{1,4}\b",
        r"(?:^|[.;])\s*(?:chuong\s+[ivx]+|dieu\s+1\b)",
        r"(?:^|[.;])\s*(?:theo de nghi|theo d nghi|kinh gui|kinh chuyen|kinh chuyn|noi nhan)\b",
        r"\s(?:tong dat|tng dt|giao nhan|giao nhn|giy t|h so)\b",
    ):
        match = re.search(pattern, text_key)
        if match:
            stop_positions.append(match.start())
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def _trim_subject_noise(text: str) -> str:
    text = _clean_value(_trim_title_stop(text))
    # Extraction callers already start at a title boundary. Searching for a
    # later verb here was destructive: e.g. "Thông tư sửa đổi, bổ sung..."
    # became only "bổ sung..." and "Bãi bỏ ... quy định mới" lost its prefix.
    return text


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _word_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _key(value))


def _canonicalize_subject(value: str, document_type: str = "") -> str:
    value = _clean_value(value)
    if not value:
        return ""
    words = value.split()
    word_keys = [_word_key(word) for word in words]

    # Frequent OCR variants Bài/Bó are not meaningful in this legal phrase.
    if len(word_keys) >= 2 and word_keys[:2] == ["bai", "bo"]:
        words[0], words[1] = "Bãi", "bỏ"

    canonical_type_words = _key(document_type).split()
    if len(canonical_type_words) == 2:
        canonical_display = document_type.lower().capitalize().split()
        for index in range(0, len(word_keys) - 2):
            if word_keys[index + 2] != "so":
                continue
            if (
                _edit_distance(word_keys[index], canonical_type_words[0]) <= 2
                and _edit_distance(word_keys[index + 1], canonical_type_words[1]) <= 2
            ):
                words[index : index + 2] = canonical_display
                word_keys[index : index + 2] = canonical_type_words
                break

    value = _clean_value(" ".join(words))
    # The enactment sentence often inserts "trong" while the official title
    # immediately below the document type omits it.
    value = re.sub(
        r"\b(quy\s+chuẩn\s+kỹ\s+thuật\s+quốc\s+gia)\s+trong\s+(lĩnh\s+vực)\b",
        r"\1 \2",
        value,
        flags=re.I,
    )
    return _clean_value(value)


def _subject_candidate_score(value: str) -> float:
    value = _clean_value(value)
    if len(value) < 18 or len(value) > 360:
        return -1
    value_key = _key(value)
    score = min(len(value), 220)
    if SERIAL_RE.search(value):
        score += 120
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", value):
        score += 40
    if value_key.startswith(("bai bo", "quy dinh", "ve viec", "sua doi", "bo sung", "ban hanh")):
        score += 100
    if value_key.startswith("bai bo mot phan"):
        score += 80
    if "bo truong" in value_key:
        score += 25
    if "linh vuc" in value_key:
        score += 25
    if value_key.startswith(("ha noi, ngay", "can cu", "dieu ", "chuong ")):
        score -= 500
    return score


def _extract_trich_yeu_from_enactment(lines: list[str], document_type: str) -> str:
    """Recover a title repeated in the normative enactment sentence."""
    markers = (("bai", "bo"), ("quy", "dinh"), ("ve", "viec"), ("sua", "doi"), ("bo", "sung"))
    candidates: list[str] = []
    for line_index, line in enumerate(lines[:180]):
        window_lines = [line]
        for following in lines[line_index + 1 : line_index + 6]:
            following_key = _key(following)
            if any(following_key.startswith(prefix) for prefix in TITLE_STOP_PREFIXES):
                break
            window_lines.append(following)
        window = " ".join(window_lines)
        words = window.split()
        key_words = _key(window).split()
        enactment_index = next(
            (
                index
                for index in range(0, max(0, len(key_words) - 1))
                if key_words[index : index + 2] == ["ban", "hanh"]
            ),
            -1,
        )
        if enactment_index < 0:
            continue
        subject_index = next(
            (
                index
                for index in range(enactment_index + 2, max(enactment_index + 2, len(key_words) - 1))
                if tuple(key_words[index : index + 2]) in markers
            ),
            -1,
        )
        if subject_index < 0 or subject_index - enactment_index > 8:
            continue
        candidate = _canonicalize_subject(_trim_title_stop(" ".join(words[subject_index:])), document_type)
        if _subject_candidate_score(candidate) >= 0:
            candidates.append(candidate)
    return max(candidates, key=_subject_candidate_score) if candidates else ""


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


def _extract_trich_yeu_after_type(lines: list[str], type_idx: int, document_type: str = "") -> str:
    if type_idx < 0:
        return ""
    title_lines: list[str] = []
    for idx, line in enumerate(lines[type_idx : type_idx + 10]):
        stop_after_line = False
        if idx == 0:
            after_type = _text_after_doc_type(line)
            line = _trim_title_stop(after_type)
            stop_after_line = _clean_value(line) != _clean_value(after_type)
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
        if stop_after_line:
            break
        if len(" ".join(title_lines)) >= 320:
            break
    return _canonicalize_subject(_trim_subject_noise(" ".join(title_lines)), document_type)


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
        alpha = [ch for ch in line if ch.isalpha()]
        is_upper_continuation = bool(
            selected
            and alpha
            and sum(ch.isupper() for ch in alpha) / len(alpha) >= 0.65
            and line_key.startswith(("va ", "truc thuoc ", "tinh ", "thanh pho ", "huyen "))
        )
        if has_org_keyword or is_country_suffix or is_upper_continuation:
            selected.append(line)

    if selected:
        value = _clean_value(" ".join(selected[:3])).upper()
        value_key = _key(value)
        if "bo nong nghiep" in value_key and "phat trien nong" in value_key:
            return "BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN"
        return value

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
    is_normative = type_label in {
        "THÔNG TƯ",
        "NGHỊ ĐỊNH",
        "QUYẾT ĐỊNH",
        "NGHỊ QUYẾT",
        "CHỈ THỊ",
        "PHÁP LỆNH",
        "LUẬT",
        "QUY CHẾ",
    }
    if is_normative and type_idx >= 0:
        # Normative documents use an explicit footer "Nơi nhận". Header stamps
        # and forwarding notes before the document type are not recipients.
        lines = lines[type_idx + 1 :]
    lines = lines[:240]
    for idx, line in enumerate(lines):
        line_key = _key(line)
        has_anchor = line_key.startswith("noi nhan") or (not is_normative and "kinh gui" in line_key)
        if has_anchor:
            parts = re.split(r"[:：-]", line, maxsplit=1)
            values = ([_clean_value(parts[1])] if len(parts) == 2 else []) + [
                _clean_value(value) for value in lines[idx + 1 : idx + 12]
            ]
            for value in values:
                value = re.sub(r"^[\s\-*•–—+]+", "", value).strip(" ;,.")
                value_key = _key(value)
                if value_key.startswith(("kt.", "tm.", "pho ", "nguoi ky")) or any(
                    marker in value_key for marker in ("thong doc", "bo truong", "chu tich")
                ):
                    break
                if not value or value_key.startswith(("luu", "nhu dieu", "dieu ")):
                    continue
                if value.count("(") != value.count(")") or value_key.endswith(" de"):
                    continue
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
    title_after_type = _extract_trich_yeu_after_type(lines, type_idx, fields.loai_van_ban)
    title_from_enactment = _extract_trich_yeu_from_enactment(lines, fields.loai_van_ban)
    title_from_label = _canonicalize_subject(_extract_trich_yeu_from_label(lines), fields.loai_van_ban)
    title_candidates = [
        value
        for value in (title_after_type, title_from_enactment, title_from_label)
        if _subject_candidate_score(value) >= 0
    ]
    fields.trich_yeu = max(title_candidates, key=_subject_candidate_score) if title_candidates else ""
    fields.co_quan_ban_hanh = _extract_co_quan_ban_hanh(lines)
    fields.noi_nhan = _extract_noi_nhan(lines)

    return fields
