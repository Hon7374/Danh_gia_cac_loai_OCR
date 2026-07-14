from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import _read_ground_truth_file


@dataclass
class WordBox:
    text: str
    confidence: float | None
    bbox: list[int]
    page: int


@dataclass
class LineCrop:
    job_id: str
    page: int
    line_index: int
    bbox: list[int]
    text: str
    confidence: float | None
    image_path: Path
    token_start: int
    token_end: int
    label: str = ""
    label_source: str = "ocr_pseudo"


TOKEN_RE = re.compile(r"\S+")
PUNCT_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a2": "c",
    }
)


def ascii_slug(value: str, fallback: str = "sample") -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:90] or fallback


def word_key(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").strip())


def clean_label(text: str) -> str:
    text = (text or "").replace("\ufeff", " ").replace("\x07", " ").replace("\x0b", " ")
    text = unicodedata.normalize("NFC", text.translate(PUNCT_TRANSLATION))
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("M"))
    return re.sub(r"\s+", " ", text).strip()


def valid_label(text: str, min_len: int, max_len: int) -> bool:
    text = clean_label(text)
    if len(text) < min_len or len(text) > max_len:
        return False
    if any(ch in text for ch in "\ufffd□■�"):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(1, min(3, min_len))


def crop_sha256(image: Image.Image) -> str:
    """Hash decoded pixels, not PNG metadata or compression bytes."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(clean_label(text).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_document_identity(job_dir: Path, report: dict[str, Any]) -> tuple[str, str]:
    """Identify the uploaded document across jobs, preferring its original SHA-256."""
    uploaded_file = report.get("uploaded_file")
    if uploaded_file:
        try:
            upload_path = (job_dir / Path(str(uploaded_file))).resolve()
            upload_path.relative_to(job_dir.resolve())
            if upload_path.is_file():
                return f"sha256:{file_sha256(upload_path)}", "uploaded_file_sha256"
        except (OSError, RuntimeError, ValueError):
            pass

    archive = report.get("document_archive")
    if isinstance(archive, dict):
        candidates = [archive.get("original_file")]
        archive_manifest = archive.get("manifest")
        if isinstance(archive_manifest, dict):
            candidates.append(archive_manifest.get("original_file"))
        for candidate in candidates:
            if isinstance(candidate, dict):
                sha256 = str(candidate.get("sha256") or "").strip().lower()
                if re.fullmatch(r"[0-9a-f]{64}", sha256):
                    return f"sha256:{sha256}", "archive_original_sha256"

    # Legacy reports may no longer retain the upload. A stable hash of all raw
    # rendered pages still groups repeated copies of the same scanned document.
    raw_images = report.get("raw_images") or []
    if not raw_images and report.get("raw_image"):
        raw_images = [report["raw_image"]]
    page_hashes: list[str] = []
    for raw_image in raw_images:
        try:
            image_path = (job_dir / Path(str(raw_image))).resolve()
            image_path.relative_to(job_dir.resolve())
            if image_path.is_file():
                page_hashes.append(file_sha256(image_path))
        except (OSError, RuntimeError, ValueError):
            page_hashes = []
            break
    if page_hashes and len(page_hashes) == len(raw_images):
        combined = hashlib.sha256("\0".join(page_hashes).encode("ascii")).hexdigest()
        return f"pages-sha256:{combined}", "raw_pages_sha256"
    return f"job:{job_dir.name}", "job_id_fallback"


def avg_confidence(boxes: list[WordBox]) -> float | None:
    vals = [float(b.confidence) for b in boxes if isinstance(b.confidence, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def split_line_segments(boxes: list[WordBox], page_width: int) -> list[list[WordBox]]:
    if len(boxes) <= 1:
        return [boxes]
    boxes = sorted(boxes, key=lambda b: b.bbox[0])
    heights = [max(1, b.bbox[3] - b.bbox[1]) for b in boxes]
    med_h = median(heights) if heights else 16
    gap_threshold = max(80, int(med_h * 5), int(page_width * 0.09))
    segments: list[list[WordBox]] = [[boxes[0]]]
    for prev, current in zip(boxes, boxes[1:]):
        gap = current.bbox[0] - prev.bbox[2]
        if gap > gap_threshold:
            segments.append([])
        segments[-1].append(current)
    return segments


def group_word_boxes_into_lines(boxes: list[WordBox], page_width: int) -> list[list[WordBox]]:
    rows: list[list[WordBox]] = []
    for box in sorted(boxes, key=lambda b: ((b.bbox[1] + b.bbox[3]) / 2, b.bbox[0])):
        y_center = (box.bbox[1] + box.bbox[3]) / 2
        height = max(1, box.bbox[3] - box.bbox[1])
        best_idx = None
        best_dist = None
        for idx, row in enumerate(rows):
            row_centers = [(b.bbox[1] + b.bbox[3]) / 2 for b in row]
            row_heights = [max(1, b.bbox[3] - b.bbox[1]) for b in row]
            row_center = sum(row_centers) / len(row_centers)
            row_height = median(row_heights)
            threshold = max(8, float(max(height, row_height)) * 0.65)
            dist = abs(y_center - row_center)
            if dist <= threshold and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            rows.append([box])
        else:
            rows[best_idx].append(box)

    rows = [sorted(row, key=lambda b: b.bbox[0]) for row in rows]
    rows.sort(key=lambda row: (min(b.bbox[1] for b in row), min(b.bbox[0] for b in row)))
    segments: list[list[WordBox]] = []
    for row in rows:
        segments.extend(split_line_segments(row, page_width))
    return segments


def make_line_crop(
    job_id: str,
    page: int,
    line_index: int,
    boxes: list[WordBox],
    image_path: Path,
    image_size: tuple[int, int],
    token_start: int,
) -> LineCrop:
    x0 = min(b.bbox[0] for b in boxes)
    y0 = min(b.bbox[1] for b in boxes)
    x1 = max(b.bbox[2] for b in boxes)
    y1 = max(b.bbox[3] for b in boxes)
    heights = [max(1, b.bbox[3] - b.bbox[1]) for b in boxes]
    pad_x = max(4, int((median(heights) if heights else 16) * 0.35))
    pad_y = max(3, int((median(heights) if heights else 16) * 0.25))
    width, height = image_size
    bbox = [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    ]
    text = clean_label(" ".join(b.text for b in sorted(boxes, key=lambda b: b.bbox[0])))
    token_count = len(tokenize(text))
    return LineCrop(
        job_id=job_id,
        page=page,
        line_index=line_index,
        bbox=bbox,
        text=text,
        confidence=avg_confidence(boxes),
        image_path=image_path,
        token_start=token_start,
        token_end=token_start + token_count,
    )


def get_result_row(report: dict[str, Any], engine: str, variant: str) -> dict[str, Any] | None:
    for row in report.get("results") or []:
        if row.get("engine") == engine and row.get("variant") == variant and row.get("status") == "ok":
            return row
    return None


def load_ground_truth(job_dir: Path, report: dict[str, Any]) -> str:
    meta = report.get("ground_truth_file")
    if not isinstance(meta, dict):
        return ""
    rel = meta.get("relative_path")
    if not rel:
        return ""
    path = job_dir / rel
    if not path.exists():
        return ""
    try:
        text, _, _ = _read_ground_truth_file(path)
        return text
    except Exception:
        return ""


def build_token_mapping(source_tokens: list[str], truth_text: str) -> dict[int, int]:
    truth_tokens = tokenize(truth_text)
    if not source_tokens or not truth_tokens:
        return {}
    source_keys = [word_key(token) for token in source_tokens]
    truth_keys = [word_key(token) for token in truth_tokens]
    matcher = SequenceMatcher(None, source_keys, truth_keys, autojunk=False)
    mapping: dict[int, int] = {}
    for match in matcher.get_matching_blocks():
        if match.size <= 0:
            continue
        for offset in range(match.size):
            src_idx = match.a + offset
            truth_idx = match.b + offset
            if source_keys[src_idx] and truth_keys[truth_idx]:
                mapping[src_idx] = truth_idx
    return mapping


def align_label(line: LineCrop, source_tokens: list[str], truth_tokens: list[str], mapping: dict[int, int]) -> tuple[str, str]:
    line_tokens = source_tokens[line.token_start:line.token_end]
    token_indexes = [idx for idx in range(line.token_start, line.token_end) if idx in mapping]
    if line_tokens and len(token_indexes) / max(1, len(line_tokens)) >= 0.65:
        mapped = sorted(mapping[idx] for idx in token_indexes)
        span_start, span_end = mapped[0], mapped[-1] + 1
        max_span = max(len(line_tokens) + 4, int(len(line_tokens) * 1.7))
        if 0 < span_end - span_start <= max_span:
            return clean_label(" ".join(truth_tokens[span_start:span_end])), "ground_truth_aligned"
    return line.text, "ocr_pseudo"


def collect_job_lines(
    job_dir: Path,
    report: dict[str, Any],
    engine: str,
    variant: str,
    min_conf: float,
) -> tuple[list[LineCrop], dict[str, Any]]:
    row = get_result_row(report, engine, variant)
    if row is None:
        return [], {"job_id": job_dir.name, "status": "missing_source_result"}

    raw_images = report.get("raw_images") or []
    if not raw_images and report.get("raw_image"):
        # Older jobs stored a single rendered page under ``raw_image``.
        raw_images = [report["raw_image"]]
    image_by_page: dict[int, Path] = {
        page_idx + 1: job_dir / Path(str(rel)) for page_idx, rel in enumerate(raw_images)
    }
    boxes_by_page: dict[int, list[WordBox]] = {}
    for raw_box in row.get("boxes") or []:
        text = clean_label(str(raw_box.get("text") or ""))
        bbox = raw_box.get("bbox")
        page = int(raw_box.get("page") or 1)
        if not text or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            clean_bbox = [int(v) for v in bbox]
        except Exception:
            continue
        if clean_bbox[2] <= clean_bbox[0] or clean_bbox[3] <= clean_bbox[1]:
            continue
        conf = raw_box.get("confidence")
        if isinstance(conf, (int, float)) and float(conf) < min_conf:
            continue
        boxes_by_page.setdefault(page, []).append(WordBox(text=text, confidence=conf, bbox=clean_bbox, page=page))

    lines: list[LineCrop] = []
    token_start = 0
    for page in sorted(boxes_by_page):
        image_path = image_by_page.get(page)
        if not image_path or not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image_size = image.size
        page_lines = group_word_boxes_into_lines(boxes_by_page[page], image_size[0])
        for line_index, line_boxes in enumerate(page_lines, start=1):
            line = make_line_crop(job_dir.name, page, line_index, line_boxes, image_path, image_size, token_start)
            token_start = line.token_end
            lines.append(line)

    truth = load_ground_truth(job_dir, report)
    truth_tokens = tokenize(truth)
    source_tokens: list[str] = []
    for line in lines:
        source_tokens.extend(tokenize(line.text))
    mapping = build_token_mapping(source_tokens, truth) if truth else {}
    for line in lines:
        if truth_tokens and mapping:
            line.label, line.label_source = align_label(line, source_tokens, truth_tokens, mapping)
        else:
            line.label, line.label_source = line.text, "ocr_pseudo"

    return lines, {
        "job_id": job_dir.name,
        "status": "ok",
        "line_count": len(lines),
        "ground_truth": bool(truth),
        "ground_truth_aligned": sum(1 for line in lines if line.label_source == "ground_truth_aligned"),
    }


def _choose_validation_documents(
    samples_by_document: dict[str, list[dict[str, Any]]],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], list[str]]:
    """Choose whole source documents close to the requested validation ratio."""
    document_ids = sorted(samples_by_document)
    if val_ratio <= 0:
        return set(), []
    if len(document_ids) < 2:
        return set(), [
            "Validation is empty because document-level splitting requires at least two unique source documents."
        ]

    rng = random.Random(seed)
    rng.shuffle(document_ids)
    counts = [len(samples_by_document[document_id]) for document_id in document_ids]
    total = sum(counts)
    target = min(total - 1, max(1, int(round(total * min(val_ratio, 0.99)))))

    # Exact subset-sum selection gives a materially better split than taking one
    # shuffled job when document sizes differ greatly. In normal use total <= 3000.
    possibilities: dict[int, tuple[int, ...]] = {0: ()}
    for idx, count in enumerate(counts):
        for subtotal, chosen in list(possibilities.items()):
            new_total = subtotal + count
            if new_total not in possibilities:
                possibilities[new_total] = chosen + (idx,)
    eligible = [
        subtotal
        for subtotal, chosen in possibilities.items()
        if 0 < subtotal < total and 0 < len(chosen) < len(document_ids)
    ]
    best_total = min(eligible, key=lambda subtotal: (abs(subtotal - target), subtotal > target, subtotal))
    validation_documents = {document_ids[idx] for idx in possibilities[best_total]}
    warnings: list[str] = []
    actual_ratio = best_total / total
    if abs(actual_ratio - val_ratio) > 0.10:
        warnings.append(
            "The validation ratio differs from the request because source documents are kept intact "
            f"({actual_ratio:.3f} actual vs {val_ratio:.3f} requested)."
        )
    return validation_documents, warnings


def _count_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("label_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _balanced_sample_subset(
    samples: list[dict[str, Any]],
    max_samples: int,
    max_samples_per_document: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap dominant documents and apply a deterministic round-robin global limit."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["source_document_id"]), []).append(sample)

    capped: dict[str, list[dict[str, Any]]] = {}
    removed_per_document: dict[str, int] = {}
    for document_id, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("job_id") or ""),
                int(row.get("page") or 0),
                tuple(row.get("bbox") or []),
                str(row.get("image_sha256") or ""),
            ),
        )
        random.Random(f"{seed}:{document_id}").shuffle(rows)
        if max_samples_per_document > 0 and len(rows) > max_samples_per_document:
            removed_per_document[document_id] = len(rows) - max_samples_per_document
            rows = rows[:max_samples_per_document]
        capped[document_id] = rows

    document_ids = sorted(capped)
    random.Random(seed).shuffle(document_ids)
    selected: list[dict[str, Any]] = []
    positions = {document_id: 0 for document_id in document_ids}
    while document_ids and (max_samples <= 0 or len(selected) < max_samples):
        remaining: list[str] = []
        for document_id in document_ids:
            position = positions[document_id]
            rows = capped[document_id]
            if position < len(rows) and (max_samples <= 0 or len(selected) < max_samples):
                selected.append(rows[position])
                positions[document_id] = position + 1
            if positions[document_id] < len(rows):
                remaining.append(document_id)
        document_ids = remaining

    selected_counts: dict[str, int] = {}
    for sample in selected:
        document_id = str(sample["source_document_id"])
        selected_counts[document_id] = selected_counts.get(document_id, 0) + 1
    return selected, {
        "before_balancing": len(samples),
        "after_balancing": len(selected),
        "global_limit": max_samples,
        "per_document_limit": max_samples_per_document,
        "removed_by_document_cap": sum(removed_per_document.values()),
        "removed_by_global_limit": max(
            0,
            sum(len(rows) for rows in capped.values()) - len(selected),
        ),
        "removed_per_document": removed_per_document,
        "selected_per_document": dict(sorted(selected_counts.items())),
    }


def write_annotations(samples: list[dict[str, Any]], output_dir: Path, val_ratio: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    samples_by_document: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        document_id = str(sample.get("source_document_id") or f"job:{sample['job_id']}")
        samples_by_document.setdefault(document_id, []).append(sample)

    validation_documents, split_warnings = _choose_validation_documents(samples_by_document, val_ratio, seed)
    train_samples = [sample for sample in samples if sample["source_document_id"] not in validation_documents]
    val_samples = [sample for sample in samples if sample["source_document_id"] in validation_documents]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    for sample in train_samples:
        sample["split"] = "train"
    for sample in val_samples:
        sample["split"] = "validation"

    train_jobs = sorted({str(sample["job_id"]) for sample in train_samples})
    val_jobs = sorted({str(sample["job_id"]) for sample in val_samples})
    train_documents = sorted({str(sample["source_document_id"]) for sample in train_samples})
    val_documents = sorted({str(sample["source_document_id"]) for sample in val_samples})
    train_image_hashes = {str(sample["image_sha256"]) for sample in train_samples}
    val_image_hashes = {str(sample["image_sha256"]) for sample in val_samples}
    train_content_hashes = {str(sample["content_sha256"]) for sample in train_samples}
    val_content_hashes = {str(sample["content_sha256"]) for sample in val_samples}
    image_overlap = train_image_hashes & val_image_hashes
    content_overlap = train_content_hashes & val_content_hashes
    job_overlap = set(train_jobs) & set(val_jobs)
    document_overlap = set(train_documents) & set(val_documents)
    if image_overlap or content_overlap or job_overlap or document_overlap:
        raise RuntimeError("Dataset split integrity check failed: train/validation leakage detected.")

    def write_file(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(f"{row['image']}\t{row['label']}\n")

    write_file(output_dir / "train.txt", train_samples)
    write_file(output_dir / "val.txt", val_samples)
    return {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_annotation": str(output_dir / "train.txt"),
        "val_annotation": str(output_dir / "val.txt"),
        "requested_val_ratio": val_ratio,
        "actual_val_ratio": round(len(val_samples) / len(samples), 6) if samples else 0.0,
        "split_jobs": {"train": train_jobs, "validation": val_jobs},
        "split_source_documents": {"train": train_documents, "validation": val_documents},
        "split_source_counts": {
            "train": _count_sources(train_samples),
            "validation": _count_sources(val_samples),
        },
        "split_integrity": {
            "job_overlap_count": len(job_overlap),
            "source_document_overlap_count": len(document_overlap),
            "image_hash_overlap_count": len(image_overlap),
            "content_hash_overlap_count": len(content_overlap),
            "passed": not (job_overlap or document_overlap or image_overlap or content_overlap),
        },
        "split_warnings": split_warnings,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    jobs_dir = args.jobs_dir.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    if args.overwrite and output_dir.exists():
        resolved = output_dir.resolve()
        if ROOT not in resolved.parents and resolved != ROOT:
            raise RuntimeError(f"Refuse to overwrite path outside project: {resolved}")
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    all_report_paths = sorted(jobs_dir.glob("*/report.json"))
    test_job_ids = set(getattr(args, "test_job_id", None) or [])
    locked_document_ids: set[str] = set()
    unresolved_test_jobs = set(test_job_ids)
    for report_path in all_report_paths:
        if report_path.parent.name not in test_job_ids:
            continue
        try:
            test_report = json.loads(report_path.read_text(encoding="utf-8"))
            document_id, _ = source_document_identity(report_path.parent, test_report)
        except Exception as exc:
            raise RuntimeError(f"Cannot lock test job {report_path.parent.name}: {exc}") from exc
        locked_document_ids.add(document_id)
        unresolved_test_jobs.discard(report_path.parent.name)
    if unresolved_test_jobs:
        raise RuntimeError(f"Test job(s) not found: {', '.join(sorted(unresolved_test_jobs))}")

    report_paths = list(all_report_paths)
    if args.job_id:
        wanted = set(args.job_id)
        report_paths = [path for path in report_paths if path.parent.name in wanted]

    samples_by_image_hash: dict[str, dict[str, Any]] = {}
    blocked_image_hashes: set[str] = set()
    job_status: list[dict[str, Any]] = []
    skipped = 0
    skipped_by_reason: dict[str, int] = {}
    duplicate_stats = {
        "exact_image_and_label": 0,
        "conflicting_label_images": 0,
        "blocked_conflict_samples": 0,
        "removed_samples": 0,
    }
    include_pseudo = bool(getattr(args, "include_pseudo", False))
    held_out_jobs: list[str] = []

    candidates_by_document: dict[
        str,
        list[tuple[Path, dict[str, Any], str, tuple[int, int, int, int, float]]],
    ] = {}
    for report_path in report_paths:
        job_dir = report_path.parent
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            document_id, document_id_basis = source_document_identity(job_dir, report)
        except Exception as exc:
            job_status.append({"job_id": job_dir.name, "status": f"bad_report:{exc}"})
            continue
        if document_id in locked_document_ids:
            held_out_jobs.append(job_dir.name)
            job_status.append(
                {
                    "job_id": job_dir.name,
                    "status": "held_out_test_document",
                    "source_document_id": document_id,
                    "source_document_id_basis": document_id_basis,
                }
            )
            continue
        source_row = get_result_row(report, args.engine, args.variant)
        score = (
            int(bool(source_row and source_row.get("status") == "ok")),
            int(bool(report.get("ground_truth_text_length"))),
            int(report.get("ground_truth_text_length") or 0),
            len((source_row or {}).get("boxes") or []),
            float(report_path.stat().st_mtime),
        )
        candidates_by_document.setdefault(document_id, []).append(
            (report_path, report, document_id_basis, score)
        )

    selected_reports: list[tuple[Path, dict[str, Any], str, str]] = []
    for document_id, candidates in sorted(candidates_by_document.items()):
        selected = max(candidates, key=lambda item: item[3])
        selected_reports.append((selected[0], selected[1], document_id, selected[2]))
        for report_path, _report, basis, _score in candidates:
            if report_path == selected[0]:
                continue
            job_status.append(
                {
                    "job_id": report_path.parent.name,
                    "status": "duplicate_source_document_alias",
                    "selected_job_id": selected[0].parent.name,
                    "source_document_id": document_id,
                    "source_document_id_basis": basis,
                }
            )

    def mark_skipped(reason: str, count: int = 1) -> None:
        nonlocal skipped
        skipped += count
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + count

    for report_path, report, document_id, document_id_basis in selected_reports:
        job_dir = report_path.parent
        lines, status = collect_job_lines(job_dir, report, args.engine, args.variant, args.min_conf)
        status["source_document_id"] = document_id
        status["source_document_id_basis"] = document_id_basis
        job_status.append(status)
        for line in lines:
            label = clean_label(line.label)
            if not valid_label(label, args.min_label_len, args.max_label_len):
                mark_skipped("invalid_label")
                continue
            if line.label_source == "ocr_pseudo" and not include_pseudo:
                mark_skipped("pseudo_label_excluded")
                continue
            if line.label_source == "ocr_pseudo" and line.confidence is not None and line.confidence < args.pseudo_min_conf:
                mark_skipped("pseudo_below_confidence")
                continue
            out_name = f"{ascii_slug(line.job_id)}_p{line.page:03d}_l{line.line_index:04d}.png"
            rel_image = Path("images") / out_name
            with Image.open(line.image_path) as image:
                crop = image.convert("RGB").crop(tuple(line.bbox))
                if crop.width < 4 or crop.height < 4:
                    mark_skipped("crop_too_small")
                    continue
                image_hash = crop_sha256(crop)
            label_hash = text_sha256(label)
            content_hash = hashlib.sha256(f"{image_hash}\0{label_hash}".encode("ascii")).hexdigest()

            if image_hash in blocked_image_hashes:
                duplicate_stats["blocked_conflict_samples"] += 1
                duplicate_stats["removed_samples"] += 1
                mark_skipped("duplicate_image_label_conflict")
                continue

            previous = samples_by_image_hash.get(image_hash)
            if previous is not None:
                if previous["label_sha256"] == label_hash:
                    duplicate_stats["exact_image_and_label"] += 1
                    duplicate_stats["removed_samples"] += 1
                    mark_skipped("exact_duplicate")
                    # Prefer a ground-truth-aligned copy if a matching pseudo
                    # sample happened to be encountered first.
                    if not (
                        previous.get("label_source") == "ocr_pseudo"
                        and line.label_source == "ground_truth_aligned"
                    ):
                        continue
                else:
                    # Identical pixels with contradictory labels are unsafe. Drop
                    # both instead of arbitrarily selecting a target string.
                    samples_by_image_hash.pop(image_hash, None)
                    blocked_image_hashes.add(image_hash)
                    duplicate_stats["conflicting_label_images"] += 1
                    duplicate_stats["removed_samples"] += 2
                    mark_skipped("duplicate_image_label_conflict", 2)
                    continue

            sample = {
                "image": rel_image.as_posix(),
                "label": label,
                "label_source": line.label_source,
                "job_id": line.job_id,
                "source_document_id": document_id,
                "source_document_id_basis": document_id_basis,
                "page": line.page,
                "bbox": line.bbox,
                "confidence": line.confidence,
                "source_text": line.text,
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "content_sha256": content_hash,
                "_source_image": str(line.image_path),
                "_crop_bbox": line.bbox,
            }
            samples_by_image_hash[image_hash] = sample

    samples = list(samples_by_image_hash.values())
    if not samples:
        hint = " Pass --include-pseudo to opt in to high-confidence OCR pseudo-labels." if not include_pseudo else ""
        raise RuntimeError(f"No VietOCR fine-tune samples were created.{hint}")

    samples, sampling_stats = _balanced_sample_subset(
        samples,
        max_samples=max(0, int(args.max_samples or 0)),
        max_samples_per_document=max(
            0,
            int(getattr(args, "max_samples_per_document", 1200) or 0),
        ),
        seed=args.seed,
    )

    # Write only unique, conflict-free, balanced crops. Re-hashing catches concurrent source
    # changes and keeps manifest hashes tied to the actual generated images.
    for sample in samples:
        source_image = Path(str(sample.pop("_source_image")))
        crop_bbox = tuple(int(value) for value in sample.pop("_crop_bbox"))
        with Image.open(source_image) as image:
            crop = image.convert("RGB").crop(crop_bbox)
            if crop_sha256(crop) != sample["image_sha256"]:
                raise RuntimeError(f"Source image changed while building dataset: {source_image}")
            crop.save(output_dir / sample["image"], optimize=True)

    source_counts = _count_sources(samples)
    annotation_stats = write_annotations(samples, output_dir, args.val_ratio, args.seed)
    manifest = {
        "dataset_type": "vietocr_line_recognition",
        "source_engine": args.engine,
        "source_variant": args.variant,
        "sample_count": len(samples),
        "source_counts": source_counts,
        "unique_source_document_count": len({sample["source_document_id"] for sample in samples}),
        "skipped": skipped,
        "skipped_by_reason": skipped_by_reason,
        "pseudo_labels": {
            "included": include_pseudo,
            "default_policy": "excluded",
            "minimum_confidence_when_included": args.pseudo_min_conf,
        },
        "duplicates": {
            **duplicate_stats,
            "unique_image_hashes": len({sample["image_sha256"] for sample in samples}),
            "unique_content_hashes": len({sample["content_sha256"] for sample in samples}),
        },
        "sampling": sampling_stats,
        "test_document_lock": {
            "requested_job_ids": sorted(test_job_ids),
            "locked_source_document_ids": sorted(locked_document_ids),
            "excluded_job_ids": sorted(held_out_jobs),
        },
        "jobs": job_status,
        "output_dir": str(output_dir),
        **annotation_stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VietOCR fine-tune line dataset from existing benchmark jobs.")
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset_template" / "vietocr_finetune")
    parser.add_argument("--job-id", action="append", help="Limit to one job id. Repeat for multiple jobs.")
    parser.add_argument(
        "--test-job-id",
        action="append",
        help=(
            "Hold out a test job and every other job with the same original document SHA-256. "
            "Repeat for multiple test documents."
        ),
    )
    parser.add_argument("--engine", default="tesseract")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--min-conf", type=float, default=70.0)
    parser.add_argument("--pseudo-min-conf", type=float, default=88.0)
    parser.add_argument(
        "--include-pseudo",
        action="store_true",
        help="Opt in to high-confidence OCR pseudo-labels (excluded by default).",
    )
    parser.add_argument("--min-label-len", type=int, default=2)
    parser.add_argument("--max-label-len", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument(
        "--max-samples-per-document",
        type=int,
        default=1200,
        help="Cap any one source document before the balanced global limit; use 0 for no per-document cap.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build_dataset(parse_args())
    print(json.dumps({
        "sample_count": manifest["sample_count"],
        "train_count": manifest["train_count"],
        "val_count": manifest["val_count"],
        "source_counts": manifest["source_counts"],
        "unique_source_document_count": manifest["unique_source_document_count"],
        "split_source_documents": manifest["split_source_documents"],
        "output_dir": manifest["output_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
