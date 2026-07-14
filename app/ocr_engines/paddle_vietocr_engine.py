from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import (
    PADDLE_VIETOCR_REFINE,
    PADDLE_VIETOCR_MAX_BOXES,
    PADDLE_VIETOCR_REFINE_TIMEOUT_SEC,
    ROOT_DIR,
    VIETOCR_BATCH_SIZE,
    VIETOCR_CONFIG_PATH,
    VIETOCR_MODEL_PROFILE,
    VIETOCR_WEIGHTS_PATH,
)
from app.services.devices import resolve_paddle_device, resolve_torch_device
from app.services.reading_order import order_boxes_xy_cut
from app.services.tempdirs import workspace_temporary_directory
from .base import BaseOCREngine, OCRBox, OCRResult
from .paddle_runtime import install_modelscope_stub

_PADDLE = None
_VIETOCR = None
_VIETOCR_MODEL_INFO: dict[str, Any] = {}
_VIETOCR_RECOGNITION_INFO: dict[str, Any] = {}

# VietOCR's greedy decoder is configured with a 128-token/character ceiling in
# the models used by this project.  A prediction that reaches that ceiling is
# almost always a missing-EOS loop, not a legitimate line of a Vietnamese
# administrative document.  The remaining limits are intentionally generous:
# they only reject a refinement when it is clearly less plausible than the
# Paddle recognition for the exact same crop.
_HYBRID_POLICY_VERSION = "paddle-vietocr-hybrid-v2"
_VIETOCR_DECODER_CHAR_CAP = 128
_VIETOCR_MIN_LENGTH_RATIO = 0.35
# Calibrated against the failing long-line document: 1.35 (with the absolute
# 12-character margin below) removes insertion loops while retaining ordinary
# diacritic and word corrections.
_VIETOCR_MAX_LENGTH_RATIO = 1.35
_VIETOCR_MIN_CONFIDENCE = 0.05
_LONG_CROP_ASPECT_THRESHOLD = 18.0
_LONG_CROP_TARGET_ASPECT = 15.5
_LONG_CROP_MAX_SEGMENTS = 3


def _alnum_length(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def _normalized_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^\w]+", text.casefold(), flags=re.UNICODE) if token]


def _has_pathological_repetition(text: str) -> bool:
    """Detect the repeated token/phrase loops produced when EOS is missed."""
    sample = " ".join(str(text or "").split())[:512]
    if not sample:
        return False

    tokens = _normalized_tokens(sample)
    # Catch "Quy Quy Quy" as well as a phrase repeated three times.  Three
    # consecutive copies are rare in document text but characteristic of a
    # runaway decoder.
    max_width = min(8, len(tokens) // 3)
    for width in range(1, max_width + 1):
        for start in range(0, len(tokens) - width * 3 + 1):
            phrase = tokens[start:start + width]
            if (
                tokens[start + width:start + width * 2] == phrase
                and tokens[start + width * 2:start + width * 3] == phrase
            ):
                return True

    # Some corrupt predictions repeat without whitespace.  Work on an alnum
    # projection so punctuation differences do not hide the loop.
    compact = "".join(char for char in sample.casefold() if char.isalnum())
    max_width = min(24, len(compact) // 3)
    for width in range(1, max_width + 1):
        if width == 1 and len(compact) < 5:
            continue
        for start in range(0, len(compact) - width * 3 + 1):
            phrase = compact[start:start + width]
            if compact.startswith(phrase * 3, start):
                # A single character must repeat at least five times to avoid
                # treating ordinary three-letter words as decoder loops.
                if width > 1 or compact.startswith(phrase * 5, start):
                    return True
    return False


def _evaluate_vietocr_candidate(
    original: OCRBox,
    candidate: OCRBox | None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return whether a VietOCR crop is safe to use and auditable metrics."""
    original_text = str(original.text or "").strip()
    candidate_text = str(candidate.text or "").strip() if candidate is not None else ""
    original_alnum = _alnum_length(original_text)
    candidate_alnum = _alnum_length(candidate_text)
    length_ratio = (candidate_alnum / original_alnum) if original_alnum else None
    candidate_confidence = _safe_float(candidate.confidence) if candidate is not None else None
    original_confidence = _safe_float(original.confidence)
    reasons: list[str] = []

    if candidate is None:
        reasons.append("missing_prediction")
    elif not candidate_text:
        reasons.append("empty_prediction")
    else:
        if len(candidate_text) >= _VIETOCR_DECODER_CHAR_CAP:
            reasons.append("decoder_cap_or_missing_eos")
        if _has_pathological_repetition(candidate_text):
            reasons.append("pathological_repetition")
        if candidate_alnum == 0:
            reasons.append("no_alphanumeric_content")
        elif candidate_alnum <= 1 and original_alnum >= 3:
            reasons.append("too_little_content")
        if candidate_text.casefold() in {"none", "null", "nan", "<unk>", "unk"}:
            reasons.append("placeholder_output")
        if "\ufffd" in candidate_text or any(ord(char) < 32 and not char.isspace() for char in candidate_text):
            reasons.append("invalid_characters")

        if length_ratio is not None:
            if (
                length_ratio > _VIETOCR_MAX_LENGTH_RATIO
                and candidate_alnum - original_alnum >= 12
            ):
                reasons.append("excessive_length_growth")
            if (
                length_ratio < _VIETOCR_MIN_LENGTH_RATIO
                and original_alnum - candidate_alnum >= 8
            ):
                reasons.append("excessive_length_shrinkage")

        if (
            candidate_confidence is not None
            and candidate_confidence < _VIETOCR_MIN_CONFIDENCE
            and (original_confidence is None or original_confidence >= candidate_confidence + 0.15)
        ):
            reasons.append("very_low_confidence")

    metrics = {
        "paddle_chars": len(original_text),
        "vietocr_chars": len(candidate_text),
        "paddle_alnum_chars": original_alnum,
        "vietocr_alnum_chars": candidate_alnum,
        "length_ratio": round(length_ratio, 4) if length_ratio is not None else None,
        "paddle_confidence": original_confidence,
        "vietocr_confidence": candidate_confidence,
    }
    return not reasons, reasons, metrics


def _apply_hybrid_refinement(
    paddle_boxes: list[OCRBox],
    vietocr_boxes: list[OCRBox] | None,
    model_info: dict[str, Any] | None = None,
) -> tuple[list[OCRBox], dict[str, Any]]:
    """Select VietOCR per crop, falling back to the original Paddle text."""
    candidates = vietocr_boxes or []
    output: list[OCRBox] = []
    accepted = 0
    fallback_details: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for index, original in enumerate(paddle_boxes):
        candidate = candidates[index] if index < len(candidates) else None
        use_candidate, reasons, metrics = _evaluate_vietocr_candidate(original, candidate)
        if use_candidate and candidate is not None:
            accepted += 1
            output.append(
                OCRBox(
                    text=str(candidate.text or "").strip(),
                    confidence=(
                        _safe_float(candidate.confidence)
                        if _safe_float(candidate.confidence) is not None
                        else original.confidence
                    ),
                    # Geometry and labels always come from Paddle.  VietOCR is
                    # a recognizer only and must never move/replace detections.
                    bbox=original.bbox,
                    label=original.label,
                    polygon=original.polygon,
                )
            )
            continue

        output.append(original)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        fallback_details.append({"box_index": index, "reasons": reasons, **metrics})

    stats: dict[str, Any] = {
        "policy": _HYBRID_POLICY_VERSION,
        "status": "completed",
        "total_paddle_boxes": len(paddle_boxes),
        "vietocr_candidates_received": len(candidates),
        "accepted_refinements": accepted,
        "fallback_to_paddle": len(paddle_boxes) - accepted,
        "extra_candidates_ignored": max(0, len(candidates) - len(paddle_boxes)),
        "fallback_reason_counts": reason_counts,
        "fallback_details": fallback_details,
        "paddle_geometry_preserved": True,
        "thresholds": {
            "decoder_char_cap": _VIETOCR_DECODER_CHAR_CAP,
            "min_length_ratio": _VIETOCR_MIN_LENGTH_RATIO,
            "max_length_ratio": _VIETOCR_MAX_LENGTH_RATIO,
            "min_confidence": _VIETOCR_MIN_CONFIDENCE,
        },
        "model": model_info or {},
    }
    return output, stats


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        vals = [_safe_float(item) for item in value]
        nums = [item for item in vals if item is not None and math.isfinite(item)]
        return sum(nums) / len(nums) if nums else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _normalize_polygon(poly: Any) -> list[list[int]] | None:
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    try:
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            return None
        points: list[list[int]] = []
        for point in poly[:4]:
            if hasattr(point, "tolist"):
                point = point.tolist()
            if len(point) < 2:
                return None
            points.append([int(round(float(point[0]))), int(round(float(point[1])))])
        return points if len(points) == 4 else None
    except Exception:
        return None


def _bbox_from_poly(poly: Any) -> list[int] | None:
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    try:
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            x0, y0, x1, y1 = [int(round(float(v))) for v in poly]
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    except Exception:
        return None
    polygon = _normalize_polygon(poly)
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _make_box(text: Any, confidence: Any, poly: Any) -> OCRBox | None:
    clean_text = str(text or "").strip()
    if not clean_text:
        return None
    return OCRBox(
        text=clean_text,
        confidence=_safe_float(confidence),
        bbox=_bbox_from_poly(poly),
        polygon=_normalize_polygon(poly),
    )


def _normalize_paddle_result(result: Any) -> list[OCRBox]:
    boxes: list[OCRBox] = []
    if result is None:
        return boxes

    if isinstance(result, list):
        for page in result:
            if not isinstance(page, dict) or "rec_texts" not in page:
                continue
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or []
            for idx, text in enumerate(texts):
                conf = scores[idx] if idx < len(scores) else None
                box = _make_box(text, conf, polys[idx] if idx < len(polys) else None)
                if box:
                    boxes.append(box)
        if boxes:
            return boxes

    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]
    for item in candidates or []:
        try:
            poly = item[0]
            rec = item[1]
            box = _make_box(rec[0], rec[1] if len(rec) > 1 else None, poly)
            if box:
                boxes.append(box)
        except Exception:
            continue
    return boxes


def _crop_by_bbox(image: Image.Image, bbox: list[int], pad: int = 2) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    crop = image.crop(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(image.width, x1 + pad),
            min(image.height, y1 + pad),
        )
    )
    if crop.width <= 1 or crop.height <= 1:
        return None
    return crop.convert("RGB")


def _crop_by_polygon(image: Image.Image, polygon: list[list[int]]) -> Image.Image | None:
    try:
        import cv2
        import numpy as np

        pts = np.array(polygon, dtype=np.float32)
        crop_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        crop_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if crop_width <= 1 or crop_height <= 1:
            return None

        dst = np.array(
            [[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            np.array(image),
            matrix,
            (crop_width, crop_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped.shape[0] / max(1, warped.shape[1]) >= 1.5:
            warped = np.rot90(warped)
        return Image.fromarray(warped).convert("RGB")
    except Exception:
        return None


def _crop_text_region(image: Image.Image, box: OCRBox) -> Image.Image | None:
    if box.polygon:
        crop = _crop_by_polygon(image, box.polygon)
        if crop is not None:
            return crop
    if box.bbox:
        return _crop_by_bbox(image, box.bbox)
    return None


def _split_long_crop_at_whitespace(
    crop: Image.Image,
) -> tuple[list[Image.Image] | None, dict[str, Any]]:
    """Split a very wide line at real vertical whitespace valleys.

    VietOCR's official recognizer is trained with a 512 px maximum input
    width.  Compressing a 25--30:1 line into that canvas destroys character
    detail.  Splitting at inter-word whitespace keeps every segment within the
    model's normal aspect range without inventing or deleting source pixels.
    """
    width, height = crop.size
    aspect = width / max(1, height)
    diagnostics: dict[str, Any] = {
        "aspect_ratio": round(aspect, 4),
        "source_width": width,
        "source_height": height,
        "segment_count": 1,
        "cut_columns": [],
        "applied": False,
        "reason": "below_aspect_threshold",
    }
    if height <= 1 or width <= 1 or aspect < _LONG_CROP_ASPECT_THRESHOLD:
        return None, diagnostics

    try:
        import numpy as np

        requested_segments = min(
            _LONG_CROP_MAX_SEGMENTS,
            max(2, int(math.ceil(aspect / _LONG_CROP_TARGET_ASPECT))),
        )
        gray = np.asarray(crop.convert("L"))
        # A permissive threshold keeps anti-aliased character strokes in the
        # projection while white inter-word gaps stay at zero density.
        ink_density = (gray < 210).mean(axis=0).astype(float)
        window = min(7, max(3, int(round(height * 0.10)) | 1))
        smooth = np.convolve(ink_density, np.ones(window) / window, mode="same")
        cuts: list[int] = []
        previous = 0
        nominal_width = width / requested_segments
        minimum_segment_width = max(16, int(height * 4.5))

        for part in range(1, requested_segments):
            target = int(round(width * part / requested_segments))
            radius = max(10, int(round(nominal_width * 0.24)))
            lower = max(previous + minimum_segment_width, target - radius)
            upper = min(width - minimum_segment_width, target + radius)
            if upper <= lower:
                diagnostics["reason"] = "no_safe_segment_width"
                return None, diagnostics
            search = smooth[lower:upper]
            relative = int(np.argmin(search))
            cut = lower + relative
            valley = float(search[relative])
            local_median = float(np.median(search)) if search.size else 0.0
            # Long Vietnamese lines normally provide a zero/near-zero
            # inter-word channel.  If the best location still crosses dense
            # ink, retain the original full crop instead of cutting a glyph.
            valley_limit = max(0.025, local_median * 0.42)
            if valley > valley_limit:
                diagnostics.update(
                    {
                        "reason": "no_reliable_whitespace_valley",
                        "rejected_cut": cut,
                        "valley_density": round(valley, 5),
                        "valley_limit": round(valley_limit, 5),
                    }
                )
                return None, diagnostics
            cuts.append(cut)
            previous = cut

        bounds = [0, *cuts, width]
        segments = [
            crop.crop((left, 0, right, height)).convert("RGB")
            for left, right in zip(bounds, bounds[1:])
        ]
        if len(segments) < 2 or any(segment.width < minimum_segment_width for segment in segments):
            diagnostics["reason"] = "invalid_segments"
            return None, diagnostics
        diagnostics.update(
            {
                "segment_count": len(segments),
                "cut_columns": cuts,
                "segment_widths": [segment.width for segment in segments],
                "applied": True,
                "reason": "long_crop_split_at_whitespace",
            }
        )
        return segments, diagnostics
    except Exception as exc:
        diagnostics.update({"reason": "split_error", "error": str(exc)})
        return None, diagnostics


def _combine_segment_predictions(
    predictions: list[tuple[str, float | None]],
) -> tuple[str, float | None]:
    useful = [(str(text or "").strip(), _safe_float(confidence)) for text, confidence in predictions]
    useful = [(text, confidence) for text, confidence in useful if text]
    if not useful:
        return "", None
    text = " ".join(item[0] for item in useful)
    weighted = [
        (confidence, max(1, _alnum_length(segment_text)))
        for segment_text, confidence in useful
        if confidence is not None
    ]
    if not weighted:
        return text, None
    total_weight = sum(weight for _, weight in weighted)
    confidence = sum(value * weight for value, weight in weighted) / max(1, total_weight)
    return text, confidence


def _predict_vietocr_batch(predictor: Any, batch: list[Image.Image]) -> list[tuple[str, float | None]]:
    if hasattr(predictor, "predict_batch"):
        try:
            texts, probs = predictor.predict_batch(batch, return_prob=True)
            return [(str(text).strip(), _safe_float(prob)) for text, prob in zip(texts or [], probs or [])]
        except TypeError:
            texts = predictor.predict_batch(batch)
            return [(str(text).strip(), None) for text in texts or []]

    predictions: list[tuple[str, float | None]] = []
    for crop in batch:
        try:
            text, prob = predictor.predict(crop, return_prob=True)
            predictions.append((str(text).strip(), _safe_float(prob)))
        except TypeError:
            predictions.append((str(predictor.predict(crop)).strip(), None))
    return predictions


def _load_vietocr_predictor():
    global _VIETOCR, _VIETOCR_MODEL_INFO
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except Exception:
        return None

    try:
        if _VIETOCR is None:
            profile = (VIETOCR_MODEL_PROFILE or "pretrained").strip().lower()
            config_path = Path(VIETOCR_CONFIG_PATH).expanduser() if VIETOCR_CONFIG_PATH else None
            # A local official checkpoint is also valid for the pretrained
            # profile.  Restricting this path to custom profiles silently
            # forced an unnecessary URL/cache lookup in every worker.
            weights_path = Path(VIETOCR_WEIGHTS_PATH).expanduser() if VIETOCR_WEIGHTS_PATH else None
            if config_path and config_path.exists():
                config = Cfg.load_config_from_file(str(config_path))
                config_source = str(config_path)
            else:
                config = Cfg.load_config_from_name("vgg_transformer")
                config_source = "vietocr:vgg_transformer"
            # An explicit local copy of the official checkpoint keeps the demo
            # self-contained and avoids a network download in every fresh
            # Windows refinement subprocess.  Custom profiles use this same
            # override for their own checkpoint.
            if weights_path and weights_path.exists():
                config["weights"] = str(weights_path)
            # The full VietOCR checkpoint already contains the VGG backbone.
            # Leaving torchvision's pretrained flag enabled triggers a network
            # download before the local state dict is loaded, breaking truly
            # offline/managed demo runs.
            if isinstance(config.get("cnn"), dict):
                config["cnn"]["pretrained"] = False
            config["device"] = resolve_torch_device()
            config["predictor"]["beamsearch"] = False
            _VIETOCR_MODEL_INFO = {
                "profile": profile,
                "config": config_source,
                "weights": config.get("weights"),
                "device": config["device"],
                "batch_size": VIETOCR_BATCH_SIZE,
            }
            _VIETOCR = Predictor(config)
        return _VIETOCR
    except Exception:
        _VIETOCR = None
        return None


def _try_vietocr_recognize_local(image_path: Path, boxes: list[OCRBox]) -> list[OCRBox] | None:
    global _VIETOCR_RECOGNITION_INFO
    predictor = _load_vietocr_predictor()
    if predictor is None:
        return None

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            # Start with explicit empty candidates.  This lets the parent
            # process distinguish a missing/failed crop prediction from a
            # genuine prediction that happens to match Paddle exactly.
            new_boxes = [
                OCRBox(
                    text="",
                    confidence=None,
                    bbox=box.bbox,
                    label=box.label,
                    polygon=box.polygon,
                )
                for box in boxes
            ]
            inference_crops: list[Image.Image] = []
            crop_groups: list[dict[str, Any]] = []
            split_details: list[dict[str, Any]] = []
            for box_index, b in enumerate(boxes):
                crop = _crop_text_region(image, b)
                if crop is None:
                    continue
                segments, split_info = _split_long_crop_at_whitespace(crop)
                selected_crops = segments or [crop.copy()]
                start_index = len(inference_crops)
                inference_crops.extend(segment.copy() for segment in selected_crops)
                crop_groups.append(
                    {
                        "box_index": box_index,
                        "prediction_start": start_index,
                        "prediction_count": len(selected_crops),
                        "split_applied": bool(segments),
                    }
                )
                if split_info.get("applied") or split_info.get("aspect_ratio", 0) >= _LONG_CROP_ASPECT_THRESHOLD:
                    split_details.append({"box_index": box_index, **split_info})
            _VIETOCR_RECOGNITION_INFO = {
                "policy": "long-crop-whitespace-split-primary-v1",
                "total_boxes": len(boxes),
                "cropped_boxes": len(crop_groups),
                "missing_crops": len(boxes) - len(crop_groups),
                "split_boxes": sum(bool(group["split_applied"]) for group in crop_groups),
                "full_crop_boxes": sum(not bool(group["split_applied"]) for group in crop_groups),
                "inference_crop_count": len(inference_crops),
                "split_segment_count": sum(
                    int(group["prediction_count"])
                    for group in crop_groups
                    if group["split_applied"]
                ),
                "aspect_threshold": _LONG_CROP_ASPECT_THRESHOLD,
                "target_segment_aspect": _LONG_CROP_TARGET_ASPECT,
                "max_segments": _LONG_CROP_MAX_SEGMENTS,
                "details": split_details,
            }
            if not inference_crops:
                return new_boxes

            predictions: list[tuple[str, float | None]] = []
            batch_size = max(1, int(VIETOCR_BATCH_SIZE or 1))
            for start in range(0, len(inference_crops), batch_size):
                batch = inference_crops[start:start + batch_size]
                predictions.extend(_predict_vietocr_batch(predictor, batch))

            for group in crop_groups:
                prediction_start = int(group["prediction_start"])
                prediction_count = int(group["prediction_count"])
                group_predictions = predictions[prediction_start:prediction_start + prediction_count]
                text, confidence = _combine_segment_predictions(group_predictions)
                box_index = int(group["box_index"])
                original = boxes[box_index]
                new_boxes[box_index] = OCRBox(
                    text=text,
                    confidence=confidence,
                    bbox=original.bbox,
                    label=original.label,
                    polygon=original.polygon,
                )
            return new_boxes
    except Exception as exc:
        _VIETOCR_RECOGNITION_INFO = {
            "policy": "long-crop-whitespace-split-primary-v1",
            "status": "error",
            "error": str(exc),
        }
        return None


def _try_vietocr_recognize(image_path: Path, boxes: list[OCRBox]) -> tuple[list[OCRBox] | None, dict[str, Any], str]:
    """Run VietOCR in a fresh process to avoid Paddle/Torch CUDA DLL conflicts on Windows."""
    global _VIETOCR_MODEL_INFO
    if not boxes:
        return boxes, {}, ""
    try:
        with workspace_temporary_directory(prefix="vietocr_refine_worker_") as tmp:
            tmp_dir = Path(tmp)
            boxes_json = tmp_dir / "boxes.json"
            out_json = tmp_dir / "refined.json"
            boxes_json.write_text(
                json.dumps(
                    [
                        {
                            "text": b.text,
                            "confidence": b.confidence,
                            "bbox": b.bbox,
                            "label": b.label,
                            "polygon": b.polygon,
                        }
                        for b in boxes
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.ocr_engines.vietocr_refine_worker",
                    str(image_path),
                    str(boxes_json),
                    str(out_json),
                ],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=PADDLE_VIETOCR_REFINE_TIMEOUT_SEC,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                return None, {}, detail or "VietOCR refine worker failed"
            if not out_json.exists():
                return None, {}, "VietOCR refine worker did not create output JSON"
            data = json.loads(out_json.read_text(encoding="utf-8"))
            if data.get("status") != "ok":
                failed_info = dict(data.get("model_info") or {})
                if data.get("recognition_info"):
                    failed_info["recognition"] = data["recognition_info"]
                return None, failed_info, data.get("error") or "VietOCR refine worker error"
            refined = [
                OCRBox(
                    text=b.get("text", ""),
                    confidence=b.get("confidence"),
                    bbox=b.get("bbox"),
                    label=b.get("label"),
                    polygon=b.get("polygon"),
                )
                for b in data.get("boxes") or []
                if isinstance(b, dict)
            ]
            model_info = dict(data.get("model_info") or {})
            if data.get("recognition_info"):
                model_info["recognition"] = data["recognition_info"]
            _VIETOCR_MODEL_INFO = model_info
            return refined, model_info, ""
    except subprocess.TimeoutExpired:
        return None, {}, f"VietOCR refine qua thoi gian {PADDLE_VIETOCR_REFINE_TIMEOUT_SEC}s."
    except Exception as exc:
        return None, {}, str(exc)


def run_paddle_vietocr_direct(image_path: Path, variant: str = "preprocessed") -> OCRResult:
    start = time.perf_counter()
    global _PADDLE
    try:
        install_modelscope_stub()
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="skipped",
            error="Chưa cài paddleocr/paddlepaddle. Chạy: pip install -r requirements-optional-ocr.txt",
            elapsed_sec=time.perf_counter() - start,
        )
    except Exception as exc:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="error",
            error=f"PaddleOCR import failed: {exc}",
            elapsed_sec=time.perf_counter() - start,
        )
    try:
        paddle_device = resolve_paddle_device()
        if _PADDLE is None:
            try:
                _PADDLE = PaddleOCR(
                    lang="vi",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=paddle_device,
                )
            except (TypeError, ValueError):
                _PADDLE = PaddleOCR(lang="vi")

        try:
            if hasattr(_PADDLE, "predict"):
                raw = _PADDLE.predict(str(image_path))
            else:
                raw = _PADDLE.ocr(str(image_path), cls=True)
        except TypeError:
            raw = _PADDLE.ocr(str(image_path))

        boxes = _normalize_paddle_result(raw)
        refine_note = "VietOCR refine disabled"
        vietocr_model_info: dict[str, Any] | None = None
        hybrid_stats: dict[str, Any] = {
            "policy": _HYBRID_POLICY_VERSION,
            "status": "disabled",
            "total_paddle_boxes": len(boxes),
            "vietocr_candidates_received": 0,
            "accepted_refinements": 0,
            "fallback_to_paddle": 0,
            "fallback_reason_counts": {},
            "fallback_details": [],
            "paddle_geometry_preserved": True,
            "model": {},
        }
        if PADDLE_VIETOCR_REFINE:
            if len(boxes) > PADDLE_VIETOCR_MAX_BOXES:
                refine_note = (
                    f"VietOCR refine skipped: {len(boxes)} text boxes exceeds "
                    f"PADDLE_VIETOCR_MAX_BOXES={PADDLE_VIETOCR_MAX_BOXES}"
                )
                hybrid_stats.update(
                    {
                        "status": "skipped_max_boxes",
                        "kept_paddle_without_attempt": len(boxes),
                    }
                )
            else:
                paddle_boxes = boxes
                viet_boxes, vietocr_model_info, refine_error = _try_vietocr_recognize(image_path, paddle_boxes)
                if viet_boxes is not None:
                    boxes, hybrid_stats = _apply_hybrid_refinement(
                        paddle_boxes,
                        viet_boxes,
                        model_info=vietocr_model_info,
                    )
                    accepted = hybrid_stats["accepted_refinements"]
                    fallback = hybrid_stats["fallback_to_paddle"]
                    refine_note = (
                        f"VietOCR hybrid accepted {accepted}/{len(paddle_boxes)} crops; "
                        f"Paddle fallback for {fallback} crops"
                    )
                else:
                    refine_note = f"VietOCR refine requested but unavailable: {refine_error}".strip()
                    hybrid_stats.update(
                        {
                            "status": "worker_error",
                            "kept_paddle_without_attempt": len(paddle_boxes),
                            "worker_error": refine_error,
                            "model": vietocr_model_info or {},
                        }
                    )
        reading_order: dict[str, Any] = {
            "policy": "recursive-xy-cut-v1",
            "status": "not_run",
            "box_count": len(boxes),
            "applied": False,
        }
        try:
            with Image.open(image_path) as source_image:
                page_width, page_height = source_image.size
            boxes, reading_order = order_boxes_xy_cut(
                boxes,
                page_width=page_width,
                page_height=page_height,
            )
            reading_order["status"] = "completed"
        except Exception as ordering_error:
            # Reading order is a presentation/layout enhancement.  OCR text
            # must remain available in detector order if geometry is malformed
            # or the image metadata cannot be read.
            reading_order["status"] = "fallback_detector_order"
            reading_order["error"] = str(ordering_error)

        text = "\n".join([b.text for b in boxes])
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="ok",
            text=text,
            boxes=boxes,
            elapsed_sec=time.perf_counter() - start,
            raw={
                "note": (
                    "PaddleOCR detects text regions, then VietOCR recognizes each crop."
                    if PADDLE_VIETOCR_REFINE
                    else "PaddleOCR detect/recognize. VietOCR refinement is off by default because it is very slow on CPU."
                ),
                "refine": refine_note,
                "crop_mode": "paddle_polygon_crop_then_whitespace_split_long_lines_then_vietocr_batch",
                "integration_reference": "longvu358/VietnameseOCR and bmd1905/vietnamese-ocr; adapted to installed PaddleOCR/VietOCR APIs.",
                "paddle_device": paddle_device,
                "vietocr_device": resolve_torch_device() if PADDLE_VIETOCR_REFINE else None,
                "vietocr_model": vietocr_model_info if PADDLE_VIETOCR_REFINE else None,
                "hybrid_refinement": hybrid_stats,
                "reading_order": reading_order,
                "vietocr_refine_timeout_sec": PADDLE_VIETOCR_REFINE_TIMEOUT_SEC if PADDLE_VIETOCR_REFINE else None,
                "vietocr_max_boxes": PADDLE_VIETOCR_MAX_BOXES if PADDLE_VIETOCR_REFINE else None,
                "worker": "subprocess",
            },
        )
    except Exception as exc:
        return OCRResult(
            engine="paddle_vietocr",
            variant=variant,
            status="error",
            error=f"PaddleOCR + VietOCR lỗi: {exc}\n{traceback.format_exc()}",
            elapsed_sec=time.perf_counter() - start,
        )


def _result_from_dict(data: dict[str, Any], fallback_variant: str, elapsed: float) -> OCRResult:
    boxes = [
        OCRBox(
            text=b.get("text", ""),
            confidence=b.get("confidence"),
            bbox=b.get("bbox"),
            label=b.get("label"),
            polygon=b.get("polygon"),
        )
        for b in data.get("boxes") or []
        if isinstance(b, dict)
    ]
    return OCRResult(
        engine=data.get("engine") or "paddle_vietocr",
        variant=data.get("variant") or fallback_variant,
        status=data.get("status") or "error",
        text=data.get("text") or "",
        boxes=boxes,
        elapsed_sec=elapsed,
        error=data.get("error") or "",
        raw=data.get("raw"),
    )


class PaddleVietOCREngine(BaseOCREngine):
    name = "paddle_vietocr"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            with workspace_temporary_directory(prefix="paddle_vietocr_worker_") as tmp:
                out_json = Path(tmp) / "result.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.ocr_engines.paddle_vietocr_worker",
                        str(image_path),
                        variant,
                        str(out_json),
                    ],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                elapsed = time.perf_counter() - start
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error=f"PaddleOCR worker lỗi: {detail}",
                        elapsed_sec=elapsed,
                    )
                if not out_json.exists():
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error="PaddleOCR worker không tạo result.json",
                        elapsed_sec=elapsed,
                    )
                return _result_from_dict(json.loads(out_json.read_text(encoding="utf-8")), variant, elapsed)
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error="PaddleOCR quá thời gian 30 phút.",
                elapsed_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"PaddleOCR + VietOCR lỗi: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )
