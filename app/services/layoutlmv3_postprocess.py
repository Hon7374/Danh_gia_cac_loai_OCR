from __future__ import annotations

import re
import json
import math
import hashlib
import unicodedata
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image, ImageOps

from app.config import (
    LAYOUTLMV3_MAX_WORDS,
    LAYOUTLMV3_MODEL_DIR,
    LAYOUTLMV3_MODEL_NAME,
    LAYOUTLMV3_PROCESSOR_NAME,
    TORCH_DEVICE,
)
from app.ocr_engines.base import OCRBox
from .field_extract import DOC_TYPES, extract_fields_rule_based
from .ocr_quality import ocr_selection_score
from .reading_order import order_boxes_xy_cut

POSTPROCESS_VERSION = "2026-07-14-robust-v4"
_MODEL_INFERENCE_LOCK = RLock()
_LAYOUT_TOKEN_MAX_LENGTH = 512
_LAYOUT_OVERFLOW_STRIDE = 64


def layout_pipeline_fingerprint() -> str:
    model_ref = LAYOUTLMV3_MODEL_DIR or LAYOUTLMV3_MODEL_NAME or ""
    artifacts: list[tuple[str, int, int]] = []
    model_path = Path(model_ref).expanduser() if model_ref else None
    if model_path and model_path.exists():
        candidates = [model_path] if model_path.is_file() else [
            path
            for pattern in ("config.json", "*.safetensors", "*.bin", "*.onnx", "label_list.json")
            for path in model_path.glob(pattern)
        ]
        for path in sorted(set(candidates)):
            try:
                stat = path.stat()
                artifacts.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
            except OSError:
                continue
    payload = {
        "postprocess_version": POSTPROCESS_VERSION,
        "model_ref": str(model_ref),
        "processor_ref": str(LAYOUTLMV3_PROCESSOR_NAME or ""),
        "max_words": LAYOUTLMV3_MAX_WORDS,
        "transformers_token_max_length": _LAYOUT_TOKEN_MAX_LENGTH,
        "transformers_overflow_stride": _LAYOUT_OVERFLOW_STRIDE,
        "transformers_window_merge": "highest_confidence_per_global_word_id",
        "field_resolver_policy": "validated-medoid-consensus-v4",
        "single_engine_policy": "keep-but-require-review",
        "artifacts": artifacts,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FIELD_ALIASES = {
    "so_ky_hieu": "so_ky_hieu",
    "so": "so_ky_hieu",
    "document_number": "so_ky_hieu",
    "doc_number_symbol": "so_ky_hieu",
    "ngay_ban_hanh": "ngay_ban_hanh",
    "ngay": "ngay_ban_hanh",
    "issued_date": "ngay_ban_hanh",
    "place_date": "ngay_ban_hanh",
    "trich_yeu": "trich_yeu",
    "subject": "trich_yeu",
    "title": "trich_yeu",
    "doc_subject": "trich_yeu",
    "co_quan_ban_hanh": "co_quan_ban_hanh",
    "issuing_agency": "co_quan_ban_hanh",
    "issuer": "co_quan_ban_hanh",
    "issue_org_name": "co_quan_ban_hanh",
    "issue_org_superior": "co_quan_ban_hanh",
    "noi_gui": "noi_gui",
    "sender": "noi_gui",
    "noi_nhan": "noi_nhan",
    "receiver": "noi_nhan",
    "recipient": "noi_nhan",
    "addressee": "noi_nhan",
    "recipients": "noi_nhan",
    "loai_van_ban": "loai_van_ban",
    "document_type": "loai_van_ban",
}
EXPECTED_FIELDS = sorted(set(FIELD_ALIASES.values()))
VIETNAMESE_CHARS = (
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
DOC_TYPE_VALUES = {label for _needle, label in DOC_TYPES}
SOURCE_PRIORITY = {
    # These are deliberately only small tie-breakers. Actual OCR quality
    # (CER/WER when available, otherwise the production quality guard) is the
    # main source score; an engine name must never dominate a bad result.
    "paddle_vietocr": 3.0,
    "tesseract": 2.5,
    "paddleocr_vl": 2.0,
    "layoutlmv3": 2.0,
    "easyocr": 1.5,
    "rule_based": 1.0,
}


def _normalize_label_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _key_text(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value or ""))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("Đ", "D").replace("đ", "d")
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def _surface_key(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n#*•,.;:-")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _source_bonus(source: dict[str, Any] | None = None) -> float:
    source = source or {}
    engine = str(source.get("engine") or source.get("_engine") or "")
    variant = str(source.get("variant") or source.get("_variant") or "")
    quality_raw = source.get("quality_score")
    if quality_raw is None:
        quality_raw = source.get("_quality_score")
    quality = _finite_float(quality_raw)
    if quality is None:
        cer_raw = source.get("cer")
        if cer_raw is None:
            cer_raw = source.get("_cer")
        wer_raw = source.get("wer")
        if wer_raw is None:
            wer_raw = source.get("_wer")
        cer = _finite_float(cer_raw)
        wer = _finite_float(wer_raw)
        if cer is not None:
            cer_pct = 100 * cer if cer <= 1 else cer
            wer_pct = cer_pct if wer is None else (100 * wer if wer <= 1 else wer)
            quality = 100 - (0.65 * cer_pct + 0.35 * wer_pct)
    quality = max(0.0, min(100.0, quality)) if quality is not None else 0.0
    return SOURCE_PRIORITY.get(engine, 0) + 0.18 * quality


def _diacritic_score(value: str) -> float:
    letters = [ch for ch in value if ch.isalpha()]
    if not letters:
        return 0
    vietnamese = sum(1 for ch in value if ch in VIETNAMESE_CHARS)
    return 80 * vietnamese / len(letters)


def _clean_field_value(key: str, value: str) -> str:
    value = " ".join(str(value or "").split()).strip()
    if not value:
        return ""
    value = re.sub(r"#{2,}", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n#*•,.;:-")
    if key == "so_ky_hieu":
        value = re.sub(r"^s[ốo0]\s*[:：.-]?\s*", "", value, flags=re.I)
        value = re.sub(r"[/／∕⁄]", "/", value)
        value = re.sub(r"\s+", "", value).upper()
    text_key = _key_text(value)

    if key == "co_quan_ban_hanh" and "bo ke hoach" in text_key and ("dau tu" in text_key or "dau t" in text_key):
        return "BỘ KẾ HOẠCH VÀ ĐẦU TƯ"

    if key == "trich_yeu":
        # Do not search-and-trim at generic verbs. Those words often occur in
        # the middle of a legitimate title ("sửa đổi, bổ sung", "ban hành quy
        # định..."). Structural trimming belongs to the line-aware extractor.
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n#*•,.;:-")

    return value


def _valid_calendar_date(value: str) -> bool:
    match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value)
    if not match:
        return False
    day, month, year = (int(part) for part in match.groups())
    if year < 1900 or year > date.today().year + 2:
        return False
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _valid_document_number(value: str) -> bool:
    # Both 151/2026/NĐ-CP and 1234/QĐ-TTg are common Vietnamese forms.
    # Require a numeric prefix, at least one slash, and an alphabetic symbol so
    # body fractions/dates cannot pass as document numbers.
    normalized = re.sub(r"\s+", "", value.upper())
    return bool(
        re.fullmatch(r"\d{1,6}/(?:\d{4}/)?[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ0-9.-]*", normalized)
    )


def _field_score(key: str, value: str, source: dict[str, Any] | None = None) -> float:
    value = _clean_field_value(key, value)
    if not value:
        return -1
    text_key = _key_text(value)
    bonus = _source_bonus(source)
    if key == "so_ky_hieu":
        if (source or {}).get("_serial_anchor_valid") is False:
            return -1
        if not _valid_document_number(value):
            return -1
        return 200 + bonus + (45 if "Đ" in value.upper() else 0) - (20 if "ND-" in value.upper() else 0)

    if key == "ngay_ban_hanh":
        if not _valid_calendar_date(value):
            return -1
        return 200 + bonus

    if key == "co_quan_ban_hanh":
        if len(value) > 120 or re.search(r"\d{2,}/\d{4}", value) or text_key.startswith(("so:", "so ")):
            return -1
        score = min(len(value), 80) + bonus + _diacritic_score(value)
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
        score = 120 + bonus + _diacritic_score(value)
        if value.strip().upper() in DOC_TYPE_VALUES:
            score += 120
        return score

    if key == "trich_yeu":
        if len(value) < 18 or len(value) > 360:
            return -1
        score = min(len(value), 180) + bonus + _diacritic_score(value)
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
        if any(
            marker in text_key
            for marker in (
                "giao nhn",
                "giy t",
                "h so",
                "thuc hien theo quy",
                "thc hin theo quy",
                "co quan, t chc",
                "tong dat",
                "tng dt",
            )
        ):
            score -= 420
        if text_key.startswith(("thong bao,", "thong bao ", "1.", "2.")):
            score -= 500
        if any(
            marker in text_key
            for marker in (
                "kinh chuyen",
                "cong van den",
                "van phong chinh phu",
                "luat thong ke ngay",
                "ha noi, ngay",
                "ngay:",
                "gio",
            )
        ):
            score -= 320
        if any(candidate_type in text_key for candidate_type in ("quy dinh", "ve viec", "phan cap", "sua doi", "bo sung", "bai bo")):
            score += 160
        if text_key.startswith("bai bo mot phan"):
            score += 80
        return score

    if key in {"noi_gui", "noi_nhan"}:
        if len(value) > 220:
            return -1
        if (source or {}).get("_recipient_anchored") is False:
            return -1
        if text_key.startswith(("dieu ", "nhu dieu", "can cu", "chuong ", "luu ")):
            return -1
        if text_key.startswith(("kt.", "tm.", "pho ")) or any(
            marker in text_key for marker in ("thong doc", "bo truong", "chu tich")
        ):
            return -1
        if value.count("(") != value.count(")") or text_key.endswith((" de", " de ")):
            return -1
        recipient_words = re.findall(r"[A-Za-zÀ-ỹĐđ0-9]+", value)
        if len(recipient_words) < 2 and text_key.upper() not in {"UBND", "HĐND", "VKSND", "TAND"}:
            return -1
        if re.match(r"^\d+[.)]", value) or any(
            marker in text_key
            for marker in (
                " co hieu luc",
                " quyet dinh nay",
                " thi hanh",
                "theo quy dinh",
                "duoc thuc hien",
                "viec thay doi",
            )
        ):
            return -1
    return len(value) + bonus + _diacritic_score(value)


def _candidate_consensus_bonus(
    key: str,
    value: str,
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> float:
    base = _surface_key(value)
    if not base:
        return 0.0
    supporters: set[str] = set()
    for other in candidates:
        if other.get("_exclude_consensus"):
            continue
        other_value = _clean_field_value(key, other.get(key) or "")
        if _field_score(key, other_value, other) < 0:
            continue
        other_key = _surface_key(other_value)
        if not other_key:
            continue
        ratio = 1.0 if base == other_key else SequenceMatcher(None, base, other_key).ratio()
        threshold = 0.90 if key == "trich_yeu" else 0.96
        if ratio >= threshold:
            engine = str(other.get("_engine") or other.get("engine") or "unknown")
            supporters.add(engine)
    # Independent engines agreeing is useful; raw/preprocessed duplicates from
    # one engine deliberately count once.
    return min(66.0, max(0, len(supporters) - 1) * 22.0)


def _candidate_consensus_profile(
    key: str,
    value: str,
    candidates: list[dict[str, Any]],
) -> tuple[int, float]:
    base = _surface_key(value)
    if not base:
        return 0, 0.0
    best_by_engine: dict[str, float] = {}
    for other in candidates:
        if other.get("_exclude_consensus"):
            continue
        other_value = _clean_field_value(key, other.get(key) or "")
        if _field_score(key, other_value, other) < 0:
            continue
        other_key = _surface_key(other_value)
        if not other_key:
            continue
        similarity = 1.0 if base == other_key else SequenceMatcher(None, base, other_key).ratio()
        if key == "co_quan_ban_hanh" and (
            base.startswith(f"{other_key} ") or other_key.startswith(f"{base} ")
        ):
            # Ministry/agency headers are frequently split over two OCR lines.
            # A clean prefix supports the complete name; length/semantic score
            # then prefers the non-truncated candidate.
            similarity = max(similarity, 0.95)
        engine = str(other.get("_engine") or other.get("engine") or "unknown")
        best_by_engine[engine] = max(best_by_engine.get(engine, 0.0), similarity)
    if not best_by_engine:
        return 0, 0.0
    threshold = {
        "trich_yeu": 0.76,
        "co_quan_ban_hanh": 0.82,
        "noi_gui": 0.86,
        "noi_nhan": 0.86,
    }.get(key, 0.96)
    support = sum(similarity >= threshold for similarity in best_by_engine.values())
    medoid_similarity = sum(best_by_engine.values()) / len(best_by_engine)
    return support, medoid_similarity


def _best_scored_candidate(
    key: str,
    candidates: list[dict[str, Any]],
) -> tuple[str, float, dict[str, Any] | None]:
    scored: list[tuple[tuple[float, ...], str, float, dict[str, Any]]] = []
    for candidate in candidates:
        value = _clean_field_value(key, candidate.get(key) or "")
        field_score = _field_score(key, value, candidate)
        if field_score < 0:
            continue
        support, medoid_similarity = _candidate_consensus_profile(key, value, candidates)
        source_score = _source_bonus(candidate)
        semantic_score = field_score - source_score
        final_score = field_score + max(0, support - 1) * 22 + medoid_similarity * 10
        # Independent agreement and field-level medoid are primary. Whole-row
        # OCR quality is only a late tie-breaker, never a reason by itself to
        # replace a cleaner field.
        if key == "co_quan_ban_hanh":
            rank = (float(support), semantic_score, round(medoid_similarity, 6), source_score)
        else:
            rank = (float(support), round(medoid_similarity, 6), semantic_score, source_score)
        scored.append((rank, value, final_score, candidate))
    if not scored:
        return "", -1, None
    _rank, value, score, candidate = max(scored, key=lambda item: item[0])
    return value, score, candidate


def _best_scored_value(key: str, candidates: list[dict[str, Any]]) -> tuple[str, float]:
    value, score, _candidate = _best_scored_candidate(key, candidates)
    return value, score


def _best_value(key: str, candidates: list[dict[str, Any]]) -> str:
    return _best_scored_value(key, candidates)[0]


def _sanitize_fields(fields: dict[str, str], source: dict[str, Any] | None = None) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key in EXPECTED_FIELDS:
        value = _clean_field_value(key, fields.get(key) or "")
        clean[key] = value if _field_score(key, value, source) >= 0 else ""
    return clean


def _canonical_field_label(label: str) -> str:
    raw = str(label or "").strip()
    if not raw or raw.upper() == "O":
        return ""
    prefix, _, name = raw.partition("-")
    if name and prefix.upper() in {"B", "I", "S", "E"}:
        raw = name
    return FIELD_ALIASES.get(_normalize_label_name(raw), "")


def _compatible_model_fields(id2label: dict[int, str] | dict[str, str] | Any) -> list[str]:
    if not isinstance(id2label, dict):
        return []
    fields = {_canonical_field_label(str(label)) for label in id2label.values()}
    return sorted(field for field in fields if field)


def _torch_device(torch: Any) -> str:
    requested = (TORCH_DEVICE or "auto").lower()
    if requested in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_transformers_runtime(
    model_ref: str,
    processor_ref: str,
    device: str,
    pipeline_fingerprint: str,
) -> tuple[Any, Any]:
    # The fingerprint participates in the cache key so replacing a checkpoint
    # invalidates the cached runtime without restarting the server.
    del pipeline_fingerprint
    from transformers import AutoModelForTokenClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_ref, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_ref)
    model.to(device)
    model.eval()
    return processor, model


def _coerce_text(value: Any, *, max_chars: int = 2_000_000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        text = str(value)
    except Exception:
        return ""
    text = text.replace("\x00", " ")
    return text[:max_chars]


def _empty_fields(note: str = "rule_based_unavailable") -> dict[str, str]:
    fields = {key: "" for key in EXPECTED_FIELDS}
    fields["confidence_note"] = note
    return fields


def _safe_rule_fields(text: Any) -> tuple[dict[str, str], str | None]:
    try:
        fields = extract_fields_rule_based(_coerce_text(text)).to_dict()
        if not isinstance(fields, dict):
            raise TypeError("field extractor did not return a mapping")
        return fields, None
    except Exception as exc:
        return _empty_fields(), f"{type(exc).__name__}: {_coerce_text(exc, max_chars=300)}"


def _raw_bbox_and_polygon(box: Any) -> tuple[Any, Any]:
    if isinstance(box, dict):
        return box.get("bbox"), box.get("polygon") or box.get("points")
    return getattr(box, "bbox", None), getattr(box, "polygon", None)


def _bbox_coordinates(raw_bbox: Any, polygon: Any = None) -> list[float] | None:
    if isinstance(raw_bbox, dict):
        raw_bbox = [
            raw_bbox.get("x0", raw_bbox.get("left")),
            raw_bbox.get("y0", raw_bbox.get("top")),
            raw_bbox.get("x1", raw_bbox.get("right")),
            raw_bbox.get("y1", raw_bbox.get("bottom")),
        ]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4 and not any(
        isinstance(item, (list, tuple, dict)) for item in raw_bbox
    ):
        values = [_finite_float(item) for item in raw_bbox]
        return [float(item) for item in values] if all(item is not None for item in values) else None

    points = polygon if polygon is not None else raw_bbox
    if not isinstance(points, (list, tuple)):
        return None
    coordinates: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, dict):
            x_value, y_value = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x_value, y_value = point[0], point[1]
        else:
            continue
        x_coord, y_coord = _finite_float(x_value), _finite_float(y_value)
        if x_coord is not None and y_coord is not None:
            coordinates.append((x_coord, y_coord))
    if len(coordinates) < 2:
        return None
    xs, ys = zip(*coordinates)
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_bbox(box: Any, width: int, height: int) -> list[int]:
    width_value = _finite_float(width)
    height_value = _finite_float(height)
    coordinates = _bbox_coordinates(box)
    if width_value is None or height_value is None or width_value <= 0 or height_value <= 0:
        raise ValueError("Kích thước ảnh không hợp lệ")
    if coordinates is None:
        raise ValueError("BBox không gồm bốn tọa độ hữu hạn")
    x0, y0, x1, y1 = coordinates
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, x0), min(width_value, x1)
    y0, y1 = max(0.0, y0), min(height_value, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("BBox rỗng hoặc nằm ngoài ảnh")

    normalized = [
        max(0, min(1000, round(1000 * x0 / width_value))),
        max(0, min(1000, round(1000 * y0 / height_value))),
        max(0, min(1000, round(1000 * x1 / width_value))),
        max(0, min(1000, round(1000 * y1 / height_value))),
    ]
    # Very small but valid boxes can collapse after conversion to LayoutLM's
    # 0..1000 grid. Expand by one grid unit instead of rejecting the document.
    if normalized[2] <= normalized[0]:
        normalized[2] = min(1000, normalized[0] + 1)
        if normalized[2] <= normalized[0]:
            normalized[0] = max(0, normalized[2] - 1)
    if normalized[3] <= normalized[1]:
        normalized[3] = min(1000, normalized[1] + 1)
        if normalized[3] <= normalized[1]:
            normalized[1] = max(0, normalized[3] - 1)
    return normalized


def _prepare_layout_words(
    boxes: Any,
    width: int,
    height: int,
) -> tuple[list[str], list[list[int]], dict[str, Any]]:
    try:
        source_boxes = list(boxes or [])
    except (TypeError, ValueError):
        source_boxes = []
    try:
        source_boxes, reading_order = order_boxes_xy_cut(source_boxes, int(width), int(height))
    except Exception as exc:
        reading_order = {
            "policy": "input_order_fallback",
            "applied": False,
            "error": f"{type(exc).__name__}: {_coerce_text(exc, max_chars=200)}",
        }
    try:
        limit = max(1, min(512, int(LAYOUTLMV3_MAX_WORDS or 512)))
    except (TypeError, ValueError, OverflowError):
        limit = 512

    words: list[str] = []
    normalized_boxes: list[list[int]] = []
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    repaired = 0
    truncated_text = 0
    split_line_boxes = 0
    accepted_source_boxes = 0
    processed_source_boxes = 0
    for box in source_boxes:
        if len(words) >= limit:
            break
        processed_source_boxes += 1
        raw_text = box.get("text") if isinstance(box, dict) else getattr(box, "text", None)
        word = re.sub(r"\s+", " ", _coerce_text(raw_text, max_chars=4096)).strip()
        if not word or word.lower() in {"none", "null", "nan"}:
            rejected["empty_text"] += 1
            continue
        if len(word) > 512:
            word = word[:512].rstrip()
            truncated_text += 1
        raw_bbox, polygon = _raw_bbox_and_polygon(box)
        coordinates = _bbox_coordinates(raw_bbox, polygon)
        if coordinates is None:
            rejected["invalid_bbox"] += 1
            continue
        if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
            repaired += 1
        if any(
            value < 0 or value > dimension
            for value, dimension in zip(coordinates, (width, height, width, height))
        ):
            repaired += 1
        try:
            # Validate/repair the source box before deriving per-word boxes.
            _normalize_bbox(coordinates, width, height)
        except (TypeError, ValueError, OverflowError):
            rejected["out_of_image_or_degenerate_bbox"] += 1
            continue
        x0, y0, x1, y1 = coordinates
        x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
        token_matches = list(re.finditer(r"\S+", word))
        if len(token_matches) > 1:
            split_line_boxes += 1
        source_added = False
        for match in token_matches:
            if len(words) >= limit:
                break
            token = match.group(0)
            # Most OCR engines in this project return one horizontal line per
            # box, while the checkpoint was trained on word boxes. Interpolate
            # a word box by character position so runtime matches training and
            # long lines cannot consume the whole 512-token budget as one word.
            denominator = max(1, len(word))
            token_x0 = x0 + (x1 - x0) * match.start() / denominator
            token_x1 = x0 + (x1 - x0) * match.end() / denominator
            try:
                norm_bbox = _normalize_bbox([token_x0, y0, token_x1, y1], width, height)
            except (TypeError, ValueError, OverflowError):
                rejected["derived_word_bbox"] += 1
                continue
            dedupe_key = (_key_text(token), tuple(norm_bbox))
            if dedupe_key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(dedupe_key)
            words.append(token)
            normalized_boxes.append(norm_bbox)
            source_added = True
        if source_added:
            accepted_source_boxes += 1

    diagnostics = {
        "input_box_count": len(source_boxes),
        "accepted_source_box_count": accepted_source_boxes,
        "accepted_word_count": len(words),
        "rejected_box_count": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
        "repaired_box_count": repaired,
        "truncated_text_count": truncated_text,
        "split_line_box_count": split_line_boxes,
        "word_box_adapter": "character_width_interpolation",
        "max_words": limit,
        "truncated_box_count": max(0, len(source_boxes) - processed_source_boxes),
        "image_size": [int(width), int(height)],
        "reading_order": reading_order,
    }
    return words, normalized_boxes, diagnostics


def _label_for_prediction(id2label: Any, prediction_id: Any) -> tuple[str, bool]:
    if not isinstance(id2label, dict):
        return "O", False
    try:
        numeric_id = int(prediction_id)
    except (TypeError, ValueError, OverflowError):
        return "O", False
    for key in (numeric_id, str(numeric_id)):
        if key in id2label:
            return _coerce_text(id2label[key], max_chars=100) or "O", True
    return "O", False


def _align_word_predictions(
    words: list[str],
    word_ids: Any,
    pred_ids: Any,
    id2label: Any,
    confidences: Any = None,
) -> tuple[list[str], list[str], list[float], dict[str, int]]:
    try:
        word_id_values = list(word_ids or [])
    except (TypeError, ValueError):
        word_id_values = []
    try:
        prediction_values = list(pred_ids or [])
    except (TypeError, ValueError):
        prediction_values = []
    try:
        confidence_values = list(confidences or [])
    except (TypeError, ValueError):
        confidence_values = []

    selected_words: list[str] = []
    labels: list[str] = []
    selected_confidences: list[float] = []
    used_word_ids: set[int] = set()
    diagnostics: Counter[str] = Counter()
    for token_idx in range(min(len(word_id_values), len(prediction_values))):
        raw_word_id = word_id_values[token_idx]
        if raw_word_id is None:
            continue
        try:
            word_id = int(raw_word_id)
        except (TypeError, ValueError, OverflowError):
            diagnostics["invalid_word_id"] += 1
            continue
        if word_id in used_word_ids:
            continue
        if word_id < 0 or word_id >= len(words):
            diagnostics["out_of_range_word_id"] += 1
            continue
        label, found = _label_for_prediction(id2label, prediction_values[token_idx])
        if not found:
            diagnostics["missing_label_id"] += 1
        confidence = _finite_float(
            confidence_values[token_idx] if token_idx < len(confidence_values) else None
        )
        selected_words.append(words[word_id])
        labels.append(label)
        selected_confidences.append(max(0.0, min(1.0, confidence)) if confidence is not None else 0.0)
        used_word_ids.add(word_id)
    diagnostics["selected_word_count"] = len(selected_words)
    diagnostics["token_count"] = min(len(word_id_values), len(prediction_values))
    diagnostics["tokenizer_truncated_word_count"] = max(0, len(words) - len(used_word_ids))
    diagnostics["tokenizer_truncated"] = int(len(used_word_ids) < len(words))
    return selected_words, labels, selected_confidences, dict(diagnostics)


def _merge_overflow_word_predictions(
    words: list[str],
    window_predictions: list[tuple[Any, Any, Any]],
    id2label: Any,
    *,
    max_length: int = _LAYOUT_TOKEN_MAX_LENGTH,
    stride: int = _LAYOUT_OVERFLOW_STRIDE,
) -> tuple[list[str], list[str], list[float], dict[str, Any]]:
    """Merge overlapping tokenizer windows back into global OCR word order.

    LayoutLMv3's tokenizer can split one Vietnamese OCR word into several BPE
    tokens. As in the training pipeline, only the first sub-token for a word in
    each window is used. If a word occurs in an overlap, the prediction with
    the highest confidence wins. Sorting by the original ``word_id`` restores
    document order before BIO decoding.
    """

    best_by_word_id: dict[int, tuple[str, float]] = {}
    appearances: Counter[int] = Counter()
    diagnostics: Counter[str] = Counter()
    window_word_counts: list[int] = []

    for word_ids, pred_ids, confidences in window_predictions:
        try:
            word_id_values = list(word_ids) if word_ids is not None else []
        except (TypeError, ValueError):
            word_id_values = []
        try:
            prediction_values = list(pred_ids) if pred_ids is not None else []
        except (TypeError, ValueError):
            prediction_values = []
        try:
            confidence_values = list(confidences) if confidences is not None else []
        except (TypeError, ValueError):
            confidence_values = []

        token_count = min(len(word_id_values), len(prediction_values))
        diagnostics["token_count"] += token_count
        seen_in_window: set[int] = set()
        for token_index in range(token_count):
            raw_word_id = word_id_values[token_index]
            if raw_word_id is None:
                continue
            try:
                word_id = int(raw_word_id)
            except (TypeError, ValueError, OverflowError):
                diagnostics["invalid_word_id"] += 1
                continue
            if word_id < 0 or word_id >= len(words):
                diagnostics["out_of_range_word_id"] += 1
                continue
            diagnostics["aligned_token_count"] += 1
            if word_id in seen_in_window:
                continue
            seen_in_window.add(word_id)
            appearances[word_id] += 1

            label, found = _label_for_prediction(id2label, prediction_values[token_index])
            if not found:
                diagnostics["missing_label_id"] += 1
            confidence = _finite_float(
                confidence_values[token_index] if token_index < len(confidence_values) else None
            )
            confidence = max(0.0, min(1.0, confidence)) if confidence is not None else 0.0
            previous = best_by_word_id.get(word_id)
            if previous is None or confidence > previous[1]:
                if previous is not None:
                    diagnostics["overlap_prediction_replacement_count"] += 1
                best_by_word_id[word_id] = (label, confidence)
        window_word_counts.append(len(seen_in_window))

    ordered_word_ids = sorted(best_by_word_id)
    selected_words = [words[word_id] for word_id in ordered_word_ids]
    labels = [best_by_word_id[word_id][0] for word_id in ordered_word_ids]
    selected_confidences = [best_by_word_id[word_id][1] for word_id in ordered_word_ids]
    covered_word_count = len(ordered_word_ids)
    truncated_word_count = max(0, len(words) - covered_word_count)
    diagnostics.update(
        {
            "selected_word_count": covered_word_count,
            "covered_word_count": covered_word_count,
            "tokenizer_truncated_word_count": truncated_word_count,
            "tokenizer_truncated": int(truncated_word_count > 0),
            "inference_window_count": len(window_predictions),
            "overflow_window_count": max(0, len(window_predictions) - 1),
            "overflow_applied": int(len(window_predictions) > 1),
            "overflow_stride": max(0, int(stride)),
            "max_sequence_length": max(1, int(max_length)),
            "overlap_word_count": sum(count > 1 for count in appearances.values()),
            "overlap_prediction_count": sum(max(0, count - 1) for count in appearances.values()),
        }
    )
    result: dict[str, Any] = dict(diagnostics)
    result["window_word_counts"] = window_word_counts
    return selected_words, labels, selected_confidences, result


def _model_inputs_for_window(
    encoding: Any,
    window_index: int,
    window_count: int,
    device: str,
) -> dict[str, Any]:
    """Select one encoded overflow window without retaining GPU batches."""

    metadata_keys = {
        "length",
        "num_truncated_tokens",
        "offset_mapping",
        "overflow_to_sample_mapping",
        "special_tokens_mask",
    }
    model_inputs: dict[str, Any] = {}
    for key, value in encoding.items():
        if key in metadata_keys:
            continue
        selected = value
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            source_index = window_index if len(value) == window_count else 0
            if source_index >= len(value):
                raise ValueError(f"Encoding input {key} thiếu window {window_index}")
            selected = value[source_index]
            # The processor currently returns pixel_values as a list of
            # unbatched [C,H,W] tensors when overflow is enabled. Other list
            # values (if supplied by another compatible processor) are likewise
            # treated as one unbatched window.
            if hasattr(selected, "unsqueeze"):
                selected = selected.unsqueeze(0)
        elif hasattr(value, "shape"):
            try:
                batch_size = int(value.shape[0])
            except (IndexError, TypeError, ValueError, OverflowError):
                batch_size = 0
            if batch_size == window_count:
                selected = value[window_index : window_index + 1]
            elif batch_size == 1:
                selected = value
            elif window_count > 1:
                raise ValueError(
                    f"Encoding input {key} có batch {batch_size}, cần {window_count} windows"
                )
        model_inputs[key] = selected.to(device) if hasattr(selected, "to") else selected
    return model_inputs


def _bio_to_fields_with_indices(
    words: list[str],
    labels: list[str],
) -> tuple[dict[str, str], dict[str, list[int]]]:
    spans: dict[str, list[dict[str, Any]]] = {}
    current_name = ""
    current_span: dict[str, Any] | None = None
    for token_index, (word, label) in enumerate(zip(words, labels)):
        name = _canonical_field_label(label)
        if not name:
            current_name = ""
            current_span = None
            continue
        raw_label = str(label or "")
        raw_prefix, separator, _raw_name = raw_label.partition("-")
        prefix = raw_prefix.upper() if separator and raw_prefix.upper() in {"B", "I", "S", "E"} else "B"
        continues = prefix in {"I", "E"} and current_name == name and current_span is not None
        if not continues:
            current_span = {"words": [], "indices": []}
            spans.setdefault(name, []).append(current_span)
        current_span["words"].append(word)
        current_span["indices"].append(token_index)
        if prefix in {"S", "E"}:
            current_name = ""
            current_span = None
        else:
            current_name = name

    output: dict[str, str] = {}
    selected_indices: dict[str, list[int]] = {}
    for key, field_spans in spans.items():
        alternatives = [
            (" ".join(span["words"]).strip(), list(span["indices"]))
            for span in field_spans
            if span["words"]
        ]
        if not alternatives:
            continue
        # Separate B/S spans are alternatives, not text fragments to blindly
        # concatenate. This prevents a labeled header date from being prepended
        # to the actual subject and supports both BIO and BIOES checkpoints.
        value, indices = max(
            alternatives,
            key=lambda item: _field_score(key, item[0], {"engine": "layoutlmv3"}),
        )
        output[key] = value
        selected_indices[key] = indices
    return output, selected_indices


def _bio_to_fields(words: list[str], labels: list[str]) -> dict[str, str]:
    return _bio_to_fields_with_indices(words, labels)[0]


def _merge_model_fields(
    fallback: dict[str, str],
    model_fields: dict[str, str],
    model_field_confidence: dict[str, float] | None = None,
) -> dict[str, str]:
    merged = _sanitize_fields(fallback, {"engine": "rule_based"})
    model_fields = _sanitize_fields(model_fields, {"engine": "layoutlmv3"})
    model_field_confidence = model_field_confidence or {}
    for key in EXPECTED_FIELDS:
        model_value = model_fields.get(key)
        if not model_value:
            continue
        fallback_value = str(merged.get(key) or "").strip()
        model_score = _field_score(key, model_value, {"engine": "layoutlmv3"})
        confidence = _finite_float(model_field_confidence.get(key))
        # Direct callers from older code may not provide confidences. Preserve
        # compatibility, while real runtimes always pass measured confidence.
        confidence = 1.0 if confidence is None else max(0.0, min(1.0, confidence))
        if not fallback_value:
            if key in {"so_ky_hieu", "ngay_ban_hanh"} and model_score >= 0 and confidence >= 0.60:
                merged[key] = model_value
            elif key == "co_quan_ban_hanh" and model_score >= 260 and confidence >= 0.85:
                merged[key] = model_value
            continue
        fallback_score = _field_score(key, fallback_value, {"engine": "rule_based"})
        same_value = _key_text(model_value) == _key_text(fallback_value)
        if same_value:
            continue
        if (
            key in {"so_ky_hieu", "ngay_ban_hanh"}
            and confidence >= 0.92
            and model_score > fallback_score + 15
        ):
            merged[key] = model_value
        elif (
            key not in {"so_ky_hieu", "ngay_ban_hanh"}
            and confidence >= 0.85
            and model_score > fallback_score + 60
        ):
            merged[key] = model_value
    return merged


def _clean_model_field_value(key: str, value: str) -> str:
    value = _clean_field_value(key, value)
    if not value:
        return value
    if key == "so_ky_hieu":
        number_match = re.search(r"\b(\d{1,5}/\d{4}/[A-ZĐ0-9-]+)\b", value.upper())
        return number_match.group(1) if number_match else value
    if key != "ngay_ban_hanh":
        return value
    date_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", value)
    if date_match:
        day, month, year = date_match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"
    normalized = _normalize_label_name(value)
    date_match = re.search(r"ngay_?(\d{1,2}).{0,12}?thang_?(\d{1,2}).{0,12}?nam_?(\d{4})", normalized)
    if date_match:
        day, month, year = date_match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"
    return value


def _clean_model_fields(fields: dict[str, str]) -> dict[str, str]:
    return {key: _clean_model_field_value(key, value) for key, value in fields.items() if value}


def _model_result(
    *,
    text: str,
    model_ref: str,
    runtime: str,
    processor_ref: str,
    selected_words: list[str],
    labels_by_word: list[str],
    id2label: dict[int, str] | dict[str, str],
    confidence_by_word: list[float] | None = None,
    input_diagnostics: dict[str, Any] | None = None,
    prediction_diagnostics: dict[str, Any] | None = None,
    model_path: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    compatible_fields = _compatible_model_fields(id2label)
    fallback, fallback_error = _safe_rule_fields(text)
    field_label_compatible = bool(compatible_fields)
    decoded_fields, selected_field_indices = (
        _bio_to_fields_with_indices(selected_words, labels_by_word)
        if field_label_compatible
        else ({}, {})
    )
    model_fields = _clean_model_fields(decoded_fields)
    confidence_values = list(confidence_by_word or [])
    confidence_groups: dict[str, list[float]] = {
        field: [
            max(0.0, min(1.0, confidence))
            for index in indices
            if (confidence := _finite_float(
                confidence_values[index] if index < len(confidence_values) else None
            )) is not None
        ]
        for field, indices in selected_field_indices.items()
    }
    model_field_confidence = {
        key: round(sum(values) / len(values), 4)
        for key, values in confidence_groups.items()
        if values
    }
    merged = _merge_model_fields(fallback, model_fields, model_field_confidence)
    merged["confidence_note"] = "layoutlmv3_validated+rule" if field_label_compatible else "rule_based"
    if field_label_compatible:
        note = (
            f"Đã chạy LayoutLMv3 thật bằng {runtime} từ model {model_ref}. "
            "Field từ model được kiểm định theo từng trường rồi mới merge với rule fallback."
        )
    else:
        note = (
            f"Đã chạy LayoutLMv3 thật bằng {runtime} từ model {model_ref}, "
            "nhưng nhãn model không khớp schema công văn nên app chỉ hiển thị nhãn token và vẫn lấy field bằng rule/regex."
        )
    labels_preview_rows = [
        {
            "token": word,
            "label": label,
            "field": _canonical_field_label(label) or "outside",
            "confidence": round(confidence_values[index], 4) if index < len(confidence_values) else None,
        }
        for index, (word, label) in enumerate(zip(selected_words[:80], labels_by_word[:80]))
    ]
    result = {
        "postprocess_version": POSTPROCESS_VERSION,
        "pipeline_fingerprint": layout_pipeline_fingerprint(),
        "mode": "layoutlmv3_model" if field_label_compatible else "layoutlmv3_model_incompatible_labels",
        "extractor": "layoutlmv3",
        "note": note,
        "model_configured": True,
        "model_ran": True,
        "model_path": model_path or model_ref,
        "processor_path": processor_ref,
        "runtime": runtime,
        "model_word_count": len(selected_words),
        "field_label_compatible": field_label_compatible,
        "compatible_fields": compatible_fields,
        "expected_fields": EXPECTED_FIELDS,
        "fields_source": "layoutlmv3_validated+rule" if field_label_compatible else "rule_based",
        "model_fields": model_fields,
        "model_field_confidence": model_field_confidence,
        "accepted_model_fields": {
            key: value for key, value in model_fields.items() if value and merged.get(key) == value
        },
        "rejected_model_fields": {
            key: value for key, value in model_fields.items() if value and merged.get(key) != value
        },
        "fields": merged,
        "labels_preview": list(zip(selected_words[:80], labels_by_word[:80])),
        "labels_preview_rows": labels_preview_rows,
        "layout_input_diagnostics": input_diagnostics or {},
        "prediction_diagnostics": prediction_diagnostics or {},
    }
    if fallback_error:
        result["rule_fallback_error"] = fallback_error
    if device:
        result["torch_device"] = device
    return result


def _footer_recipient(text: str) -> str:
    lines = [_coerce_text(line, max_chars=500).strip() for line in text.splitlines()]
    start = next(
        (index for index, line in enumerate(lines) if _key_text(line).startswith("noi nhan")),
        None,
    )
    if start is None:
        return ""
    inline = re.sub(r"^\s*n[ơo]i\s+nh[aậ]n\s*[:：-]?\s*", "", lines[start], flags=re.I)
    candidates = ([inline] if inline else []) + lines[start + 1 : start + 12]
    for value in candidates:
        value = re.sub(r"^[\s\-*•–—+]+", "", value).strip(" ;,.")
        key = _key_text(value)
        if key.startswith(("kt.", "tm.", "pho ", "nguoi ky")) or any(
            marker in key for marker in ("thong doc", "bo truong", "chu tich")
        ):
            break
        if not value or key.startswith(("luu", "nhu tren", "tm.", "kt.", "noi nhan")):
            continue
        if value.count("(") != value.count(")") or key.endswith(" de"):
            continue
        if len(value) <= 180:
            return value
    return ""


def _candidate_fields_from_row(row: dict[str, Any] | Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if row.get("engine") == "glm_ocr" or row.get("status") != "ok":
        return None
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    first_page_text = _coerce_text(raw.get("first_page_text") or row.get("text")).strip()
    full_text = _coerce_text(row.get("text") or first_page_text).strip()
    if not first_page_text:
        return None
    try:
        fields = extract_fields_rule_based(first_page_text).to_dict()
        if full_text and full_text != first_page_text:
            full_document_fields = extract_fields_rule_based(full_text).to_dict()
            # Header metadata should stay anchored to page 1, while Nơi nhận is
            # commonly located on the final page of a multi-page document.
            if full_document_fields.get("noi_nhan"):
                fields["noi_nhan"] = full_document_fields["noi_nhan"]
            elif _footer_recipient(full_text):
                fields["noi_nhan"] = _footer_recipient(full_text)
    except Exception:
        return None
    fields["_engine"] = row.get("engine")
    fields["_variant"] = row.get("variant")
    fields["_cer"] = row.get("cer")
    fields["_wer"] = row.get("wer")
    full_key = _key_text(full_text)
    fields["_recipient_anchored"] = bool(
        re.search(r"(?:^|\s)(?:noi\s+nhan|kinh\s+gui)\s*[:：-]?", full_key)
    )
    fields["_serial_anchor_valid"] = not bool(
        re.search(r"\b(?:so|s0|s6)\s*[:：.-]?\s*[/／∕⁄]", full_key)
    )
    try:
        fields["_quality_score"] = ocr_selection_score(row)
    except Exception:
        fields["_quality_score"] = 0.0
    return fields


def stabilize_layout_fields_from_rows(layout_result: dict | None, rows: list[dict]) -> dict | None:
    """Validate LayoutLMv3 fields and stabilize them with first-page OCR candidates."""
    if not isinstance(layout_result, dict):
        return layout_result
    if not isinstance(layout_result.get("fields"), dict):
        layout_result["fields"] = _empty_fields()

    candidates = [
        candidate
        for row in (rows if isinstance(rows, (list, tuple)) else [])
        if (candidate := _candidate_fields_from_row(row))
    ]

    fields = _sanitize_fields(layout_result["fields"], {"engine": "layoutlmv3"})
    current_ocr_source = layout_result.get("ocr_source")
    if not isinstance(current_ocr_source, dict):
        current_ocr_source = {}
    current_source = {
        "engine": current_ocr_source.get("engine") or "layoutlmv3",
        "variant": current_ocr_source.get("variant"),
        "quality_score": current_ocr_source.get("selection_score"),
    }
    current_candidate: dict[str, Any] = {
        **fields,
        "_engine": current_source["engine"],
        "_variant": current_source["variant"],
        "_quality_score": current_source["quality_score"],
        "_is_current": True,
        "_exclude_consensus": not bool(current_ocr_source.get("engine")),
    }
    field_sources: dict[str, dict[str, Any]] = {}
    if not candidates:
        fields["confidence_note"] = "layoutlmv3_validated"
        layout_result["fields"] = fields
        layout_result["fields_source"] = "layoutlmv3_validated"
        layout_result["stabilization"] = {"candidate_count": 0, "field_sources": {}}
        return layout_result

    for key in ("so_ky_hieu", "ngay_ban_hanh", "co_quan_ban_hanh", "trich_yeu"):
        best, best_score, best_candidate = _best_scored_candidate(key, [*candidates, current_candidate])
        current_score = _field_score(key, fields.get(key, ""), current_source)
        if best and best_score >= current_score:
            fields[key] = best
            if best_candidate and best_candidate.get("_is_current"):
                field_sources[key] = {"source": "current_validated", "selection_score": round(best_score, 3)}
            else:
                field_sources[key] = {
                    "engine": best_candidate.get("_engine") if best_candidate else None,
                    "variant": best_candidate.get("_variant") if best_candidate else None,
                    "quality_score": best_candidate.get("_quality_score") if best_candidate else None,
                    "selection_score": round(best_score, 3),
                }

    doc_type, doc_type_score, doc_type_candidate = _best_scored_candidate(
        "loai_van_ban", [*candidates, current_candidate]
    )
    if doc_type:
        fields["loai_van_ban"] = doc_type
        field_sources["loai_van_ban"] = {
            "engine": doc_type_candidate.get("_engine") if doc_type_candidate else None,
            "variant": doc_type_candidate.get("_variant") if doc_type_candidate else None,
            "quality_score": doc_type_candidate.get("_quality_score") if doc_type_candidate else None,
            "selection_score": round(doc_type_score, 3),
        }

    for key in ("noi_gui", "noi_nhan"):
        if fields.get(key) and _field_score(key, str(fields.get(key)), {"engine": "layoutlmv3"}) >= 0:
            continue
        best, best_score, best_candidate = _best_scored_candidate(key, candidates)
        support, _similarity = _candidate_consensus_profile(key, best, candidates) if best else (0, 0.0)
        if best and support >= 2:
            fields[key] = best
            field_sources[key] = {
                "engine": best_candidate.get("_engine") if best_candidate else None,
                "variant": best_candidate.get("_variant") if best_candidate else None,
                "quality_score": best_candidate.get("_quality_score") if best_candidate else None,
                "selection_score": round(best_score, 3),
            }

    fields["confidence_note"] = "layoutlmv3+best_ocr_rule"
    layout_result["fields"] = fields
    layout_result["fields_source"] = "layoutlmv3+best_ocr_rule"
    layout_result["stabilization"] = {
        "candidate_count": len(candidates),
        "independent_engine_count": len(
            {str(candidate.get("_engine") or "unknown") for candidate in candidates}
        ),
        "field_sources": field_sources,
        "selection_policy": "validated_field+dynamic_ocr_quality+independent_engine_consensus",
    }
    return layout_result


def finalize_layout_result(layout_result: dict | None) -> dict | None:
    """Make final field decisions/provenance consistent after every resolver."""
    if not isinstance(layout_result, dict):
        return layout_result
    fields = layout_result.get("fields") if isinstance(layout_result.get("fields"), dict) else {}
    model_fields = layout_result.get("model_fields") if isinstance(layout_result.get("model_fields"), dict) else {}
    old_accepted = layout_result.get("accepted_model_fields")
    old_rejected = layout_result.get("rejected_model_fields")
    if old_accepted or old_rejected:
        layout_result["model_merge_decision_before_stabilization"] = {
            "accepted": old_accepted if isinstance(old_accepted, dict) else {},
            "rejected": old_rejected if isinstance(old_rejected, dict) else {},
        }
    layout_result["accepted_model_fields"] = {
        key: value
        for key, value in model_fields.items()
        if value and _surface_key(fields.get(key, "")) == _surface_key(value)
    }
    layout_result["rejected_model_fields"] = {
        key: value
        for key, value in model_fields.items()
        if value and _surface_key(fields.get(key, "")) != _surface_key(value)
    }

    stabilization = layout_result.get("stabilization")
    selected_sources = (
        stabilization.get("field_sources", {})
        if isinstance(stabilization, dict) and isinstance(stabilization.get("field_sources"), dict)
        else {}
    )
    final_provenance: dict[str, dict[str, Any]] = {}
    review_reasons: list[str] = []
    independent_engine_count = (
        int(stabilization.get("independent_engine_count") or 0)
        if isinstance(stabilization, dict)
        else 0
    )
    for key in EXPECTED_FIELDS:
        value = _clean_field_value(key, fields.get(key) or "")
        valid = bool(value) and _field_score(key, value, {}) >= 0
        source = selected_sources.get(key)
        if not isinstance(source, dict):
            if key in layout_result["accepted_model_fields"]:
                source = {"source": "layoutlmv3_model"}
            elif value:
                source = {"source": "validated_rule_or_existing"}
            else:
                source = {"source": "missing"}
        status = "valid" if valid else ("missing" if not value else "invalid")
        critical = key in {
            "so_ky_hieu",
            "ngay_ban_hanh",
            "co_quan_ban_hanh",
            "loai_van_ban",
            "trich_yeu",
        }
        if (
            valid
            and critical
            and independent_engine_count < 2
            and layout_result.get("mode") != "text_input_direct"
        ):
            status = "unverified_single_engine"
            review_reasons.append(f"single_engine_unverified:{key}")
        final_provenance[key] = {
            **source,
            "status": status,
        }
        if critical and not valid:
            review_reasons.append(f"missing_or_invalid:{key}")
    layout_result["final_field_provenance"] = final_provenance
    layout_result["review_required"] = bool(review_reasons)
    layout_result["review_reasons"] = review_reasons
    return layout_result


def _is_onnx_model_ref(model_ref: str) -> bool:
    lowered = model_ref.lower().replace("\\", "/")
    path = Path(model_ref)
    if path.exists():
        return path.suffix.lower() == ".onnx" or (path.is_dir() and any(path.rglob("*.onnx")))
    return lowered == "welcomyou/layoutlmv3-vn-admin-kie"


def _resolve_onnx_layoutlmv3(model_ref: str) -> tuple[Path, Path]:
    path = Path(model_ref)
    if path.exists():
        if path.is_file() and path.suffix.lower() == ".onnx":
            return path, path.parent
        candidates = sorted(path.rglob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy file .onnx trong {path}")
        return candidates[0], candidates[0].parent

    from huggingface_hub import snapshot_download

    repo_dir = Path(
        snapshot_download(
            model_ref,
            allow_patterns=[
                "**/*.onnx",
                "**/config.json",
                "**/label_list.json",
                "**/merges.txt",
                "**/special_tokens_map.json",
                "**/tokenizer.json",
                "**/tokenizer_config.json",
                "**/vocab.json",
            ],
        )
    )
    candidates = sorted(repo_dir.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"Model {model_ref} không có file .onnx")
    return candidates[0], candidates[0].parent


def _load_onnx_labels(model_dir: Path) -> dict[int, str]:
    label_path = model_dir / "label_list.json"
    if label_path.exists():
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        return {index: str(label) for index, label in enumerate(labels)}
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config.get("id2label") or {}
        return {int(index): str(label) for index, label in id2label.items()}
    raise FileNotFoundError(f"Không tìm thấy label_list.json/config.json trong {model_dir}")


def _run_onnx_layoutlmv3(
    model_ref: str,
    text: str,
    words: list[str],
    norm_boxes: list[list[int]],
    input_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path, model_dir = _resolve_onnx_layoutlmv3(model_ref)
    id2label = _load_onnx_labels(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    encoding = tokenizer(
        words,
        boxes=norm_boxes,
        return_tensors="np",
        truncation=True,
        max_length=min(512, max(64, LAYOUTLMV3_MAX_WORDS)),
    )
    word_ids = encoding.word_ids(batch_index=0)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs: dict[str, Any] = {}
    for item in session.get_inputs():
        if item.name in encoding:
            onnx_inputs[item.name] = encoding[item.name].astype(np.int64)
        elif item.name == "token_type_ids":
            onnx_inputs[item.name] = np.zeros_like(encoding["input_ids"], dtype=np.int64)
        else:
            raise ValueError(f"ONNX model cần input chưa hỗ trợ: {item.name}")
    logits = session.run(None, onnx_inputs)[0]
    pred_ids = logits.argmax(-1)[0].tolist()
    shifted = logits[0] - np.max(logits[0], axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
    token_confidences = probabilities.max(axis=-1).tolist()
    selected_words, labels_by_word, word_confidences, prediction_diagnostics = _align_word_predictions(
        words,
        word_ids,
        pred_ids,
        id2label,
        token_confidences,
    )
    return _model_result(
        text=text,
        model_ref=model_ref,
        runtime="onnxruntime",
        processor_ref=str(model_dir),
        selected_words=selected_words,
        labels_by_word=labels_by_word,
        id2label=id2label,
        confidence_by_word=word_confidences,
        input_diagnostics=input_diagnostics,
        prediction_diagnostics=prediction_diagnostics,
        model_path=str(onnx_path),
    )


def _fallback_result(text: str, mode: str, note: str, **extra: Any) -> dict[str, Any]:
    fields, extraction_error = _safe_rule_fields(text)
    result = {
        "postprocess_version": POSTPROCESS_VERSION,
        "pipeline_fingerprint": layout_pipeline_fingerprint(),
        "mode": mode,
        "extractor": "rule_based",
        "note": note,
        "model_configured": bool(LAYOUTLMV3_MODEL_DIR or LAYOUTLMV3_MODEL_NAME),
        "model_path": LAYOUTLMV3_MODEL_DIR or LAYOUTLMV3_MODEL_NAME,
        "model_ran": False,
        "field_label_compatible": False,
        "compatible_fields": [],
        "expected_fields": EXPECTED_FIELDS,
        "fields_source": "rule_based",
        "fields": fields,
    }
    if extraction_error:
        result["rule_fallback_error"] = extraction_error
    result.update(extra)
    return result


def layoutlmv3_postprocess(image_path: Path, text: str, boxes: list[OCRBox] | None) -> dict[str, Any]:
    """Run field extraction with a real LayoutLMv3 checkpoint when configured."""
    text = _coerce_text(text)
    model_ref = ""
    if LAYOUTLMV3_MODEL_DIR:
        local_model_path = Path(LAYOUTLMV3_MODEL_DIR).expanduser()
        if not local_model_path.exists():
            return _fallback_result(
                text,
                "rule_based_model_missing",
                f"Không tìm thấy checkpoint LayoutLMv3 tại {local_model_path}. App dùng rule/regex và không chạy model giả.",
                model_path=str(local_model_path),
            )
        model_ref = str(local_model_path)
    elif LAYOUTLMV3_MODEL_NAME:
        model_ref = LAYOUTLMV3_MODEL_NAME

    if not model_ref:
        return _fallback_result(
            text,
            "rule_based_extractor",
            "Chưa có checkpoint LayoutLMv3 token-classification đã fine-tune cho trường công văn. App dùng rule/regex và không chạy model giả.",
        )

    if not boxes:
        return _fallback_result(
            text,
            "rule_based_no_boxes",
            "OCR output không có bounding boxes. LayoutLMv3 cần ảnh + text + bbox nên app dùng rule/regex.",
            model_path=model_ref,
        )

    try:
        with Image.open(Path(image_path)) as source_image:
            source_image.load()
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        words, norm_boxes, input_diagnostics = _prepare_layout_words(
            boxes,
            image.width,
            image.height,
        )
        if not words:
            return _fallback_result(
                text,
                "rule_based_no_valid_boxes",
                "Không có word/bbox hợp lệ sau khi kiểm tra từng box. App dùng rule/regex an toàn.",
                model_path=model_ref,
                layout_input_diagnostics=input_diagnostics,
            )

        if _is_onnx_model_ref(model_ref):
            return _run_onnx_layoutlmv3(
                model_ref,
                text,
                words,
                norm_boxes,
                input_diagnostics,
            )

        import torch

        processor_ref = LAYOUTLMV3_PROCESSOR_NAME or model_ref
        device = _torch_device(torch)
        token_max_length = _LAYOUT_TOKEN_MAX_LENGTH
        overflow_stride = min(_LAYOUT_OVERFLOW_STRIDE, max(0, token_max_length - 8))
        window_predictions: list[tuple[Any, Any, Any]] = []
        with _MODEL_INFERENCE_LOCK:
            processor, model = _load_transformers_runtime(
                model_ref,
                processor_ref,
                device,
                layout_pipeline_fingerprint(),
            )
            encoding = processor(
                image,
                words,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                max_length=token_max_length,
                return_overflowing_tokens=True,
                stride=overflow_stride,
                padding="max_length",
            )
            encoded_input_ids = encoding.get("input_ids")
            if encoded_input_ids is None:
                raise ValueError("Processor không trả về input_ids")
            try:
                window_count = int(encoded_input_ids.shape[0])
            except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
                window_count = len(encoded_input_ids)
            if window_count < 1:
                raise ValueError("Processor không tạo được inference window")
            id2label = model.config.id2label
            with torch.no_grad():
                for window_index in range(window_count):
                    word_ids = encoding.word_ids(batch_index=window_index)
                    model_inputs = _model_inputs_for_window(
                        encoding,
                        window_index,
                        window_count,
                        device,
                    )
                    outputs = model(**model_inputs)
                    logits = outputs.logits
                    if len(logits.shape) != 3 or int(logits.shape[0]) != 1:
                        raise ValueError(
                            f"Model trả logits sai shape cho window {window_index}: {tuple(logits.shape)}"
                        )
                    pred_ids = logits.argmax(-1)[0].detach().cpu().tolist()
                    token_confidences = (
                        logits.softmax(-1).max(-1).values[0].detach().cpu().tolist()
                    )
                    window_predictions.append((word_ids, pred_ids, token_confidences))

        selected_words, labels_by_word, word_confidences, prediction_diagnostics = (
            _merge_overflow_word_predictions(
                words,
                window_predictions,
                id2label,
                max_length=token_max_length,
                stride=overflow_stride,
            )
        )
        if not selected_words:
            return _fallback_result(
                text,
                "rule_based_layoutlmv3_output_error",
                "Model không trả về token nào căn chỉnh được với OCR. App dùng rule/regex an toàn.",
                model_path=model_ref,
                layout_input_diagnostics=input_diagnostics,
                prediction_diagnostics=prediction_diagnostics,
            )
        return _model_result(
            text=text,
            model_ref=model_ref,
            runtime="transformers",
            processor_ref=processor_ref,
            selected_words=selected_words,
            labels_by_word=labels_by_word,
            id2label=id2label,
            confidence_by_word=word_confidences,
            input_diagnostics=input_diagnostics,
            prediction_diagnostics=prediction_diagnostics,
            device=device,
        )
    except Exception as exc:
        return _fallback_result(
            text,
            "rule_based_layoutlmv3_error",
            f"Không chạy được LayoutLMv3 nên app dùng rule/regex. Lỗi: {type(exc).__name__}: {_coerce_text(exc, max_chars=500)}",
            model_path=model_ref,
            error_type=type(exc).__name__,
        )
