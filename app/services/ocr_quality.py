from __future__ import annotations

import re
import unicodedata
from typing import Any


VIETNAMESE_ADMIN_KEYWORDS = {
    "bo",
    "cong",
    "co",
    "chu",
    "chinh",
    "doc",
    "dinh",
    "dao",
    "do",
    "duc",
    "giao",
    "hanh",
    "hoa",
    "hoi",
    "luat",
    "nam",
    "ngan",
    "ngay",
    "nghi",
    "nghia",
    "nha",
    "nuoc",
    "phap",
    "phuc",
    "quoc",
    "quy",
    "so",
    "thang",
    "thong",
    "tu",
    "viet",
    "xa",
}

BROKEN_VIETNAMESE_TOKENS = {
    "cng",
    "dc",
    "dnh",
    "hnh",
    "lp",
    "lut",
    "mt",
    "ngh",
    "ngha",
    "nghip",
    "ph",
    "quc",
    "son",
    "thm",
    "vit",
}


def strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def has_vietnamese_mark(value: str) -> bool:
    for ch in value:
        if ch in {"đ", "Đ"}:
            return True
        if any(unicodedata.category(mark) == "Mn" for mark in unicodedata.normalize("NFD", ch)):
            return True
    return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text or "", re.UNICODE)


def analyze_ocr_text_quality(text: str) -> dict[str, Any]:
    """Estimate Vietnamese OCR usability without ground truth.

    This is not a replacement for CER/WER. It is a guardrail for production runs
    where no ground-truth file is uploaded, especially to catch outputs that look
    confident but drop Vietnamese diacritics and vowels.
    """
    tokens = _tokens(text)
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not tokens or not letters:
        return {
            "token_count": len(tokens),
            "letter_count": len(letters),
            "marked_letter_ratio": 0.0,
            "marked_token_ratio": 0.0,
            "admin_keyword_hits": 0,
            "broken_token_hits": 0,
            "likely_vietnamese": False,
            "diacritic_loss": False,
            "severe_diacritic_loss": False,
            "vietnamese_quality_score": 0.0,
            "quality_penalty": 0.0,
            "note": "Không đủ text để đánh giá chất lượng tiếng Việt.",
        }

    marked_letters = sum(1 for ch in letters if has_vietnamese_mark(ch))
    marked_tokens = sum(1 for token in tokens if has_vietnamese_mark(token))
    normalized_tokens = [strip_diacritics(token).lower() for token in tokens]
    keyword_hits = sum(1 for token in normalized_tokens if token in VIETNAMESE_ADMIN_KEYWORDS)
    broken_hits = sum(1 for token in normalized_tokens if token in BROKEN_VIETNAMESE_TOKENS)

    marked_letter_ratio = marked_letters / max(1, len(letters))
    marked_token_ratio = marked_tokens / max(1, len(tokens))
    keyword_ratio = keyword_hits / max(1, len(tokens))
    broken_ratio = broken_hits / max(1, len(tokens))
    likely_vietnamese = marked_token_ratio >= 0.18 or keyword_hits >= 8

    diacritic_loss = bool(likely_vietnamese and len(tokens) >= 80 and marked_token_ratio < 0.55)
    severe_diacritic_loss = bool(
        likely_vietnamese
        and len(tokens) >= 80
        and (marked_token_ratio < 0.45 or broken_ratio >= 0.08)
    )

    # In administrative Vietnamese text, a healthy OCR output usually has many
    # marked tokens. Scale to 100 but keep room for texts with acronyms/numbers.
    vietnamese_quality_score = min(100.0, round((marked_token_ratio / 0.78) * 100, 2))
    if likely_vietnamese:
        vietnamese_quality_score = max(0.0, vietnamese_quality_score - min(35.0, broken_ratio * 180))

    penalty = 0.0
    note = "Chất lượng dấu tiếng Việt ổn."
    if severe_diacritic_loss:
        penalty = 30.0
        note = "Nghi ngờ rụng dấu/mất nguyên âm tiếng Việt nghiêm trọng."
    elif diacritic_loss:
        penalty = 16.0
        note = "Nghi ngờ rụng dấu tiếng Việt."

    return {
        "token_count": len(tokens),
        "letter_count": len(letters),
        "marked_letter_ratio": round(marked_letter_ratio, 4),
        "marked_token_ratio": round(marked_token_ratio, 4),
        "admin_keyword_hits": keyword_hits,
        "admin_keyword_ratio": round(keyword_ratio, 4),
        "broken_token_hits": broken_hits,
        "broken_token_ratio": round(broken_ratio, 4),
        "likely_vietnamese": likely_vietnamese,
        "diacritic_loss": diacritic_loss,
        "severe_diacritic_loss": severe_diacritic_loss,
        "vietnamese_quality_score": round(vietnamese_quality_score, 2),
        "quality_penalty": penalty,
        "note": note,
    }


def ocr_selection_score(row: dict, max_text_len: int | None = None) -> float:
    """Score an OCR row for production selection when CER/WER may be missing."""
    if row.get("status") != "ok" or not (row.get("text") or "").strip():
        return -1.0

    if row.get("cer") is not None:
        cer_pct = float(row.get("cer") or 0) * 100
        wer_pct = float(row.get("wer") or 0) * 100 if row.get("wer") is not None else cer_pct
        return max(0.0, 100 - (0.65 * cer_pct + 0.35 * wer_pct))

    text = row.get("text") or ""
    guard = row.get("quality_guard") or analyze_ocr_text_quality(text)
    text_len = len(text)
    max_len = max(1, int(max_text_len or text_len or 1))
    text_score = min(100.0, text_len / max_len * 100)

    avg_conf = None
    boxes = row.get("boxes") or []
    vals: list[float] = []
    for box in boxes:
        if isinstance(box, dict) and isinstance(box.get("confidence"), (int, float)):
            vals.append(float(box["confidence"]))
    if vals:
        avg_conf = sum(vals) / len(vals)
        if avg_conf <= 1:
            avg_conf *= 100

    vi_score = float(guard.get("vietnamese_quality_score") or 0)
    if avg_conf is None:
        score = 0.55 * text_score + 0.45 * vi_score
    else:
        score = 0.45 * avg_conf + 0.20 * text_score + 0.35 * vi_score

    score -= float(guard.get("quality_penalty") or 0)
    if row.get("variant") == "raw":
        score += 3.0
    if row.get("engine") == "tesseract":
        score += 2.0
    if row.get("engine") == "paddle_vietocr" and guard.get("severe_diacritic_loss"):
        score -= 20.0
    return round(max(0.0, min(100.0, score)), 3)
