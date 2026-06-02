from __future__ import annotations

import argparse
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


def write_annotations(samples: list[dict[str, Any]], output_dir: Path, val_ratio: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    val_count = int(round(len(samples) * val_ratio)) if len(samples) > 10 else max(1, len(samples) // 5)
    val_count = min(max(1, val_count), max(1, len(samples) - 1)) if len(samples) > 1 else 0
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

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

    report_paths = sorted(jobs_dir.glob("*/report.json"))
    if args.job_id:
        wanted = set(args.job_id)
        report_paths = [path for path in report_paths if path.parent.name in wanted]

    samples: list[dict[str, Any]] = []
    job_status: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    skipped = 0

    for report_path in report_paths:
        job_dir = report_path.parent
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            job_status.append({"job_id": job_dir.name, "status": f"bad_report:{exc}"})
            continue
        lines, status = collect_job_lines(job_dir, report, args.engine, args.variant, args.min_conf)
        job_status.append(status)
        for line in lines:
            label = clean_label(line.label)
            if not valid_label(label, args.min_label_len, args.max_label_len):
                skipped += 1
                continue
            if line.label_source == "ocr_pseudo" and line.confidence is not None and line.confidence < args.pseudo_min_conf:
                skipped += 1
                continue
            out_name = f"{ascii_slug(line.job_id)}_p{line.page:03d}_l{line.line_index:04d}.png"
            rel_image = Path("images") / out_name
            out_path = output_dir / rel_image
            with Image.open(line.image_path) as image:
                crop = image.convert("RGB").crop(tuple(line.bbox))
                if crop.width < 4 or crop.height < 4:
                    skipped += 1
                    continue
                crop.save(out_path, optimize=True)
            sample = {
                "image": rel_image.as_posix(),
                "label": label,
                "label_source": line.label_source,
                "job_id": line.job_id,
                "page": line.page,
                "bbox": line.bbox,
                "confidence": line.confidence,
                "source_text": line.text,
            }
            samples.append(sample)
            source_counts[line.label_source] = source_counts.get(line.label_source, 0) + 1
            if args.max_samples and len(samples) >= args.max_samples:
                break
        if args.max_samples and len(samples) >= args.max_samples:
            break

    if not samples:
        raise RuntimeError("No VietOCR fine-tune samples were created.")

    annotation_stats = write_annotations(samples, output_dir, args.val_ratio, args.seed)
    manifest = {
        "dataset_type": "vietocr_line_recognition",
        "source_engine": args.engine,
        "source_variant": args.variant,
        "sample_count": len(samples),
        "source_counts": source_counts,
        "skipped": skipped,
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
    parser.add_argument("--engine", default="tesseract")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--min-conf", type=float, default=70.0)
    parser.add_argument("--pseudo-min-conf", type=float, default=88.0)
    parser.add_argument("--min-label-len", type=int, default=2)
    parser.add_argument("--max-label-len", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=3000)
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
        "output_dir": manifest["output_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
