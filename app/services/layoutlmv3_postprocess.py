from __future__ import annotations

import re
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import (
    LAYOUTLMV3_MAX_WORDS,
    LAYOUTLMV3_MODEL_DIR,
    LAYOUTLMV3_MODEL_NAME,
    LAYOUTLMV3_PROCESSOR_NAME,
    TORCH_DEVICE,
)
from app.ocr_engines.base import OCRBox
from .field_extract import DOC_TYPES, extract_fields_rule_based


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
    "easyocr": 70,
    "paddleocr_vl": 55,
    "tesseract": 35,
    "layoutlmv3": 25,
    "paddle_vietocr": 10,
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


def _source_bonus(source: dict[str, Any] | None = None) -> float:
    source = source or {}
    engine = str(source.get("engine") or source.get("_engine") or "")
    variant = str(source.get("variant") or source.get("_variant") or "")
    return SOURCE_PRIORITY.get(engine, 0) + (8 if variant == "raw" else 0)


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
    text_key = _key_text(value)

    if key == "co_quan_ban_hanh" and "bo ke hoach" in text_key and ("dau tu" in text_key or "dau t" in text_key):
        return "BỘ KẾ HOẠCH VÀ ĐẦU TƯ"

    if key == "trich_yeu":
        words = value.split()
        key_words = _key_text(value).split()
        for needle in (("quy", "dinh"), ("qay", "dinh"), ("ve", "viec"), ("sua", "doi"), ("bo", "sung")):
            n = len(needle)
            for idx in range(0, max(0, len(key_words) - n + 1)):
                if tuple(key_words[idx : idx + n]) == needle:
                    value = " ".join(words[idx:])
                    break
            else:
                continue
            break
        value = re.split(r"\b(?:Giờ|Gio|Kinh|Căn cứ|Can cu)\b", value, maxsplit=1, flags=re.I)[0]
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n#*•,.;:-")

    return value


def _field_score(key: str, value: str, source: dict[str, Any] | None = None) -> float:
    value = _clean_field_value(key, value)
    if not value:
        return -1
    text_key = _key_text(value)
    bonus = _source_bonus(source)
    engine = str((source or {}).get("engine") or (source or {}).get("_engine") or "")

    if key == "so_ky_hieu":
        if not re.search(r"\d{1,5}/\d{4}/", value):
            return -1
        return 200 + bonus + (45 if "Đ" in value.upper() else 0) - (20 if "ND-" in value.upper() else 0)

    if key == "ngay_ban_hanh":
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
            return 20 + bonus
        score = 200 + bonus
        if engine == "paddle_vietocr":
            score += 70
        elif engine == "paddleocr_vl":
            score -= 55
        elif engine == "tesseract":
            score -= 20
        return score

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
        if any(candidate_type in text_key for candidate_type in ("quy dinh", "ve viec", "phan cap", "sua doi", "bo sung")):
            score += 160
        return score

    if key in {"noi_gui", "noi_nhan"} and len(value) > 220:
        return -1
    return len(value) + bonus + _diacritic_score(value)


def _best_scored_value(key: str, candidates: list[dict[str, Any]]) -> tuple[str, float]:
    scored = [
        (_clean_field_value(key, candidate.get(key) or ""), _field_score(key, candidate.get(key) or "", candidate))
        for candidate in candidates
    ]
    scored = [(value, score) for value, score in scored if score >= 0]
    if not scored:
        return "", -1
    return max(scored, key=lambda item: item[1])


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


def _compatible_model_fields(id2label: dict[int, str] | dict[str, str]) -> list[str]:
    fields = {_canonical_field_label(str(label)) for label in id2label.values()}
    return sorted(field for field in fields if field)


def _torch_device(torch: Any) -> str:
    requested = (TORCH_DEVICE or "auto").lower()
    if requested in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_bbox(box: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / max(1, width)))),
        max(0, min(1000, int(1000 * y0 / max(1, height)))),
        max(0, min(1000, int(1000 * x1 / max(1, width)))),
        max(0, min(1000, int(1000 * y1 / max(1, height)))),
    ]


def _bio_to_fields(words: list[str], labels: list[str]) -> dict[str, str]:
    fields: dict[str, list[list[str]]] = {}
    current = ""
    for word, label in zip(words, labels):
        name = _canonical_field_label(label)
        if not name:
            current = ""
            continue
        prefix = str(label or "").split("-", 1)[0].upper()
        if prefix != "I" or name != current or not fields.get(name):
            fields.setdefault(name, []).append([])
        fields[name][-1].append(word)
        current = name
    return {
        key: " ".join(" ".join(span).strip() for span in spans if span).strip()
        for key, spans in fields.items()
    }


def _merge_model_fields(fallback: dict[str, str], model_fields: dict[str, str]) -> dict[str, str]:
    merged = _sanitize_fields(fallback, {"engine": "rule_based"})
    model_fields = _sanitize_fields(model_fields, {"engine": "layoutlmv3"})
    for key in EXPECTED_FIELDS:
        model_value = model_fields.get(key)
        if not model_value:
            continue
        fallback_value = str(merged.get(key) or "").strip()
        model_score = _field_score(key, model_value, {"engine": "layoutlmv3"})
        if not fallback_value:
            if key in {"so_ky_hieu", "ngay_ban_hanh"} and model_score >= 0:
                merged[key] = model_value
            elif key == "co_quan_ban_hanh" and model_score >= 260:
                merged[key] = model_value
            continue
        fallback_score = _field_score(key, fallback_value, {"engine": "rule_based"})
        if key in {"so_ky_hieu", "ngay_ban_hanh"} and model_score >= fallback_score:
            merged[key] = model_value
        elif key not in {"so_ky_hieu", "ngay_ban_hanh"} and model_score > fallback_score + 60:
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
    model_path: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    compatible_fields = _compatible_model_fields(id2label)
    fallback = extract_fields_rule_based(text).to_dict()
    field_label_compatible = bool(compatible_fields)
    model_fields = (
        _clean_model_fields(_bio_to_fields(selected_words, labels_by_word))
        if field_label_compatible
        else {}
    )
    merged = _merge_model_fields(fallback, model_fields)
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
        }
        for word, label in zip(selected_words[:80], labels_by_word[:80])
    ]
    result = {
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
        "accepted_model_fields": {
            key: value for key, value in model_fields.items() if value and merged.get(key) == value
        },
        "rejected_model_fields": {
            key: value for key, value in model_fields.items() if value and merged.get(key) != value
        },
        "fields": merged,
        "labels_preview": list(zip(selected_words[:80], labels_by_word[:80])),
        "labels_preview_rows": labels_preview_rows,
    }
    if device:
        result["torch_device"] = device
    return result


def _candidate_fields_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("engine") == "glm_ocr" or row.get("status") != "ok":
        return None
    text = ((row.get("raw") or {}).get("first_page_text") or row.get("text") or "").strip()
    if not text:
        return None
    try:
        fields = extract_fields_rule_based(text).to_dict()
    except Exception:
        return None
    fields["_engine"] = row.get("engine")
    fields["_variant"] = row.get("variant")
    return fields


def stabilize_layout_fields_from_rows(layout_result: dict | None, rows: list[dict]) -> dict | None:
    """Validate LayoutLMv3 fields and stabilize them with first-page OCR candidates."""
    if not layout_result or not isinstance(layout_result.get("fields"), dict):
        return layout_result

    candidates = [
        candidate
        for row in rows
        if (candidate := _candidate_fields_from_row(row))
    ]

    fields = _sanitize_fields(layout_result["fields"], {"engine": "layoutlmv3"})
    if not candidates:
        fields["confidence_note"] = "layoutlmv3_validated"
        layout_result["fields"] = fields
        layout_result["fields_source"] = "layoutlmv3_validated"
        return layout_result

    for key in ("so_ky_hieu", "ngay_ban_hanh", "co_quan_ban_hanh", "trich_yeu"):
        best, best_score = _best_scored_value(key, candidates)
        current_score = _field_score(key, fields.get(key, ""), {"engine": "layoutlmv3"})
        if best and best_score >= current_score:
            fields[key] = best

    doc_type_values = [
        (candidate.get("loai_van_ban") or "").strip()
        for candidate in candidates
        if _field_score("loai_van_ban", (candidate.get("loai_van_ban") or "").strip(), candidate) >= 0
    ]
    if doc_type_values:
        counts = Counter(doc_type_values)
        fields["loai_van_ban"] = max(
            doc_type_values,
            key=lambda value: (counts[value], _field_score("loai_van_ban", value)),
        )

    for key in ("noi_gui", "noi_nhan"):
        if fields.get(key) and _field_score(key, str(fields.get(key)), {"engine": "layoutlmv3"}) >= 0:
            continue
        best = _best_value(key, candidates)
        if best:
            fields[key] = best

    fields["confidence_note"] = "layoutlmv3+best_ocr_rule"
    layout_result["fields"] = fields
    layout_result["fields_source"] = "layoutlmv3+best_ocr_rule"
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
    labels_by_word: list[str] = []
    used_word_ids = []
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in used_word_ids:
            continue
        used_word_ids.append(word_id)
        labels_by_word.append(id2label[int(pred_ids[token_idx])])
    selected_words = [words[index] for index in used_word_ids if index < len(words)]
    return _model_result(
        text=text,
        model_ref=model_ref,
        runtime="onnxruntime",
        processor_ref=str(model_dir),
        selected_words=selected_words,
        labels_by_word=labels_by_word,
        id2label=id2label,
        model_path=str(onnx_path),
    )


def _fallback_result(text: str, mode: str, note: str, **extra: Any) -> dict[str, Any]:
    result = {
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
        "fields": extract_fields_rule_based(text).to_dict(),
    }
    result.update(extra)
    return result


def layoutlmv3_postprocess(image_path: Path, text: str, boxes: list[OCRBox] | None) -> dict[str, Any]:
    """Run field extraction with a real LayoutLMv3 checkpoint when configured."""
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
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        image = Image.open(image_path).convert("RGB")
        valid_boxes = [box for box in boxes if box.text and box.bbox][:LAYOUTLMV3_MAX_WORDS]
        words = [box.text for box in valid_boxes]
        norm_boxes = [_normalize_bbox(box.bbox, image.width, image.height) for box in valid_boxes]
        if not words:
            raise ValueError("Không có word/bbox hợp lệ")

        if _is_onnx_model_ref(model_ref):
            return _run_onnx_layoutlmv3(model_ref, text, words, norm_boxes)

        processor_ref = LAYOUTLMV3_PROCESSOR_NAME or model_ref
        processor = AutoProcessor.from_pretrained(processor_ref, apply_ocr=False)
        model = AutoModelForTokenClassification.from_pretrained(model_ref)
        device = _torch_device(torch)
        model.to(device)
        model.eval()

        encoding = processor(
            image,
            words,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=min(512, max(64, LAYOUTLMV3_MAX_WORDS)),
        )
        word_ids = encoding.word_ids(batch_index=0)
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoding.items()
        }
        with torch.no_grad():
            outputs = model(**model_inputs)

        pred_ids = outputs.logits.argmax(-1)[0].detach().cpu().tolist()
        id2label = model.config.id2label
        labels_by_word: list[str] = []
        used_word_ids = []
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in used_word_ids:
                continue
            used_word_ids.append(word_id)
            labels_by_word.append(id2label[int(pred_ids[token_idx])])

        selected_words = [words[index] for index in used_word_ids if index < len(words)]
        return _model_result(
            text=text,
            model_ref=model_ref,
            runtime="transformers",
            processor_ref=processor_ref,
            selected_words=selected_words,
            labels_by_word=labels_by_word,
            id2label=id2label,
            device=device,
        )
    except Exception as exc:
        return _fallback_result(
            text,
            "rule_based_layoutlmv3_error",
            f"Không chạy được LayoutLMv3 nên app dùng rule/regex. Lỗi: {exc}",
            model_path=model_ref,
        )
