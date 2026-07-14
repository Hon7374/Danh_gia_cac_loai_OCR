from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from app.config import PADDLEOCR_VL_CMD, PADDLEOCR_VL_DEVICE, PADDLEOCR_VL_ENGINE, ROOT_DIR
from app.services.devices import resolve_paddle_device
from app.services.tempdirs import workspace_temporary_directory
from .base import BaseOCREngine, OCRBox, OCRResult
from .paddle_runtime import install_modelscope_stub

_PADDLEOCR_VL = None

_NUMBERED_REPEAT_RE = re.compile(
    r"(?<!\w)(?P<label>[^\W\d_]{2,24})\s+(?P<number>\d{1,4})\.",
    flags=re.UNICODE,
)
_NUMBERED_REPEAT_MIN_RUN = 40


def _split_configured_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        parts = [
            part[1:-1]
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}
            else part
            for part in parts
        ]
    if parts and Path(parts[0]).name.lower() in {"paddleocr", "paddleocr.exe"}:
        # The stock PaddleOCR launcher may import ModelScope/Torch before Paddle,
        # which can trigger CUDA DLL conflicts on Windows. Use the project wrapper.
        parts = [sys.executable, "-m", "app.ocr_engines.paddleocr_cli", *parts[1:]]
    return parts


def _extract_text(obj: Any) -> str:
    chunks: list[str] = []
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ["markdown", "text", "content", "result", "html", "recognized_text", "layout_text", "rec_text"]:
            if key in obj:
                s = _extract_text(obj[key])
                if s:
                    chunks.append(s)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                s = _extract_text(v)
                if s:
                    chunks.append(s)
    if isinstance(obj, list):
        for v in obj:
            s = _extract_text(v)
            if s:
                chunks.append(s)
    return "\n".join(dict.fromkeys(chunks)).strip()


def _read_generated_output(output_dir: Path) -> tuple[str, dict[str, Any]]:
    raw: dict[str, Any] = {"output_dir": str(output_dir), "files": []}
    text_chunks: list[str] = []
    json_chunks: list[str] = []
    if not output_dir.exists():
        return "", raw
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        raw["files"].append(str(p))
        if p.suffix.lower() in {".md", ".markdown", ".txt"}:
            try:
                text_chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        elif p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                raw[str(p)] = obj
                s = _extract_text(obj)
                if s:
                    json_chunks.append(s)
            except Exception:
                pass
    # PaddleOCR-VL commonly writes the same result as both Markdown and JSON.
    # Markdown/TXT is the canonical OCR text; JSON remains in ``raw`` for
    # boxes/debugging and is only a text fallback when no plain output exists.
    chunks = text_chunks or json_chunks
    unique_chunks = list(dict.fromkeys(c.strip() for c in chunks if c.strip()))
    return "\n".join(unique_chunks).strip(), raw


def _find_runaway_numbered_sequences(text: str) -> list[dict[str, Any]]:
    """Find implausibly long ``same-label + increasing number`` hallucinations.

    A normal legal list contains text between successive headings.  PaddleOCR-VL
    can instead emit a dense sequence such as ``Dấu 2. Dấu 3. ... Dấu 601.``
    from one tiny crop.  Requiring at least 40 consecutive numbers and allowing
    only punctuation/whitespace between matches keeps ordinary numbered lists.
    """
    matches = list(_NUMBERED_REPEAT_RE.finditer(text or ""))
    if not matches:
        return []

    runs: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = [matches[0]]
    for match in matches[1:]:
        previous = current[-1]
        gap = text[previous.end():match.start()]
        same_label = match.group("label").casefold() == previous.group("label").casefold()
        consecutive = int(match.group("number")) == int(previous.group("number")) + 1
        punctuation_only_gap = len(gap) <= 12 and not re.search(r"[\w\d]", gap, flags=re.UNICODE)
        if same_label and consecutive and punctuation_only_gap:
            current.append(match)
        else:
            if len(current) >= _NUMBERED_REPEAT_MIN_RUN:
                runs.append(current)
            current = [match]
    if len(current) >= _NUMBERED_REPEAT_MIN_RUN:
        runs.append(current)

    return [
        {
            "start": run[0].start(),
            "end": run[-1].end(),
            "label": run[0].group("label"),
            "start_number": int(run[0].group("number")),
            "end_number": int(run[-1].group("number")),
            "count": len(run),
        }
        for run in runs
    ]


def _sanitize_vl_hallucinations(
    text: str,
    boxes: list[OCRBox],
) -> tuple[str, list[OCRBox], dict[str, Any]]:
    """Remove only proven runaway numbered sequences and retain an audit trail."""
    cleaned_text = text or ""
    kept_boxes: list[OCRBox] = []
    removed: list[dict[str, Any]] = []

    for box in boxes:
        sequences = _find_runaway_numbered_sequences(box.text)
        if not sequences:
            kept_boxes.append(box)
            continue
        removed.append(
            {
                "reason": "consecutive_numbered_repetition",
                "text_length": len(box.text),
                "bbox": list(box.bbox) if box.bbox else None,
                "label": box.label,
                "sequences": sequences,
            }
        )
        # Generated Markdown and layout JSON contain the same box text.  Remove
        # the exact offending crop output from the canonical text as well.
        cleaned_text = cleaned_text.replace(box.text, " ")

    # Keep the guard effective even when a backend returns plain text without
    # parseable layout boxes.
    text_only_sequences = _find_runaway_numbered_sequences(cleaned_text)
    for sequence in reversed(text_only_sequences):
        cleaned_text = cleaned_text[: sequence["start"]] + " " + cleaned_text[sequence["end"] :]

    cleaned_text = re.sub(r"[ \t]+\n", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    diagnostics = {
        "policy": "consecutive-numbered-repetition-v1",
        "minimum_run": _NUMBERED_REPEAT_MIN_RUN,
        "removed_box_count": len(removed),
        "removed_text_characters": max(0, len(text or "") - len(cleaned_text)),
        "removed_sequence_count": sum(len(item["sequences"]) for item in removed) + len(text_only_sequences),
        "removed": removed,
        "text_only_sequences": text_only_sequences,
    }
    return cleaned_text, kept_boxes, diagnostics


def _extract_layout_boxes(obj: Any) -> list[OCRBox]:
    boxes: list[OCRBox] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            for match in re.finditer(
                r"label:\s*(?P<label>[^\n]+)\s*\nbbox:\s*\[(?P<bbox>[^\]]+)\]\s*\ncontent:\s*(?P<content>.*?)(?=\n#################|\Z)",
                value,
                flags=re.DOTALL,
            ):
                try:
                    bbox = [int(float(part.strip())) for part in match.group("bbox").split(",")[:4]]
                except ValueError:
                    continue
                content = match.group("content").strip()
                content = re.sub(r"\n?#################\s*$", "", content).strip()
                if content:
                    boxes.append(OCRBox(text=content, bbox=bbox, label=match.group("label").strip()))
            return
        if isinstance(value, dict):
            text = value.get("content") or value.get("text") or value.get("rec_text")
            raw_bbox = value.get("bbox") or value.get("coordinate")
            if isinstance(text, str) and isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
                try:
                    bbox = [int(float(part)) for part in raw_bbox[:4]]
                    boxes.append(OCRBox(text=text.strip(), bbox=bbox, label=str(value.get("label") or "") or None))
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                if isinstance(child, (dict, list, str)):
                    visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)

    visit(obj)
    unique: list[OCRBox] = []
    seen = set()
    for box in boxes:
        key = (box.text, tuple(box.bbox or []), box.label)
        if key not in seen:
            unique.append(box)
            seen.add(key)
    return unique


class PaddleOCRVLEngine(BaseOCREngine):
    name = "paddleocr_vl"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        try:
            with workspace_temporary_directory(prefix="paddleocr_vl_worker_") as tmp:
                out_json = Path(tmp) / "result.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.ocr_engines.paddleocr_vl_worker",
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
                        error=f"PaddleOCR-VL worker lỗi: {detail}",
                        elapsed_sec=elapsed,
                    )
                if not out_json.exists():
                    return OCRResult(
                        engine=self.name,
                        variant=variant,
                        status="error",
                        error="PaddleOCR-VL worker không tạo result.json",
                        elapsed_sec=elapsed,
                    )
                data = json.loads(out_json.read_text(encoding="utf-8"))
                return OCRResult(
                    engine=data.get("engine") or self.name,
                    variant=data.get("variant") or variant,
                    status=data.get("status") or "error",
                    text=data.get("text") or "",
                    boxes=[
                        OCRBox(
                            text=box.get("text", ""),
                            confidence=box.get("confidence"),
                            bbox=box.get("bbox"),
                            label=box.get("label"),
                        )
                        for box in (data.get("boxes") or [])
                        if isinstance(box, dict)
                    ],
                    elapsed_sec=elapsed,
                    error=data.get("error") or "",
                    raw=data.get("raw"),
                )
        except subprocess.TimeoutExpired:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error="PaddleOCR-VL quá thời gian 30 phút.",
                elapsed_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"PaddleOCR-VL lỗi: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )


def run_paddleocr_vl_direct(image_path: Path, variant: str = "preprocessed") -> OCRResult:
    start = time.perf_counter()

    # 1) Cách bền nhất: cấu hình command. Hỗ trợ placeholder {input}, {output_dir}.
    if PADDLEOCR_VL_CMD:
        try:
            with workspace_temporary_directory(prefix="paddleocr_vl_") as tmp:
                out_dir = Path(tmp) / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                device = resolve_paddle_device(PADDLEOCR_VL_DEVICE)
                cmd = (
                    PADDLEOCR_VL_CMD
                    .replace("{input}", str(image_path))
                    .replace("{output_dir}", str(out_dir))
                    .replace("{device}", device)
                )
                completed = subprocess.run(
                    _split_configured_command(cmd),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                stdout = completed.stdout.strip()
                stderr = completed.stderr.strip()
                file_text, raw_files = _read_generated_output(out_dir)
                if completed.returncode != 0:
                    return OCRResult(
                        engine="paddleocr_vl",
                        variant=variant,
                        status="error",
                        error=f"PADDLEOCR_VL_CMD trả mã {completed.returncode}: {stderr or stdout}",
                        elapsed_sec=time.perf_counter() - start,
                    )
                text = file_text or stdout
                raw_payload = {"stdout": stdout, "stderr": stderr, "generated": raw_files, "device": device, "worker": "subprocess"}
                boxes = _extract_layout_boxes(raw_payload)
                text, boxes, hallucination_guard = _sanitize_vl_hallucinations(text, boxes)
                raw_payload["hallucination_guard"] = hallucination_guard
                return OCRResult(
                    engine="paddleocr_vl",
                    variant=variant,
                    status="ok" if text else "skipped",
                    text=text,
                    boxes=boxes,
                    elapsed_sec=time.perf_counter() - start,
                    raw=raw_payload,
                )
        except Exception as exc:
            return OCRResult(
                engine="paddleocr_vl",
                variant=variant,
                status="error",
                error=f"Lỗi khi chạy PADDLEOCR_VL_CMD: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )

    # 2) API Python chính thức mới: from paddleocr import PaddleOCRVL.
    try:
        global _PADDLEOCR_VL
        install_modelscope_stub()
        from paddleocr import PaddleOCRVL  # type: ignore

        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
        }
        device = resolve_paddle_device(PADDLEOCR_VL_DEVICE)
        if device:
            kwargs["device"] = device
        if PADDLEOCR_VL_ENGINE:
            kwargs["vl_rec_backend"] = PADDLEOCR_VL_ENGINE
        if _PADDLEOCR_VL is None:
            _PADDLEOCR_VL = PaddleOCRVL(**kwargs)
        pipeline = _PADDLEOCR_VL
        output = pipeline.predict(str(image_path))
        pages = list(output)
        chunks: list[str] = []
        raw: list[Any] = []
        with workspace_temporary_directory(prefix="paddleocr_vl_api_") as tmp:
            out_dir = Path(tmp) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            for res in pages:
                raw.append(str(res))
                for method in ["save_to_markdown", "save_to_json"]:
                    if hasattr(res, method):
                        try:
                            getattr(res, method)(save_path=out_dir)
                        except Exception:
                            pass
            file_text, raw_files = _read_generated_output(out_dir)
        if file_text:
            chunks.append(file_text)
        else:
            chunks.extend([str(x) for x in raw if str(x).strip()])
        text = "\n".join(chunks).strip()
        boxes = _extract_layout_boxes({"pages": raw, "generated": raw_files})
        text, boxes, hallucination_guard = _sanitize_vl_hallucinations(text, boxes)
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="ok" if text else "skipped",
            text=text,
            boxes=boxes,
            elapsed_sec=time.perf_counter() - start,
            raw={
                "api": "PaddleOCRVL",
                "pages": raw,
                "device": device,
                "worker": "subprocess",
                "hallucination_guard": hallucination_guard,
            },
        )
    except ModuleNotFoundError:
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="skipped",
            error="Chưa cài PaddleOCR-VL/PaddleOCR 3.x. Chạy scripts/setup_latest_windows.ps1 -Full hoặc cấu hình PADDLEOCR_VL_CMD.",
            elapsed_sec=time.perf_counter() - start,
        )
    except Exception as exc:
        return OCRResult(
            engine="paddleocr_vl",
            variant=variant,
            status="error",
            error=f"PaddleOCR-VL lỗi: {exc}",
            elapsed_sec=time.perf_counter() - start,
        )
