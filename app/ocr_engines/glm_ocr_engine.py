from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

from app.config import GLM_OCR_API_KEY, GLM_OCR_ENDPOINT, ROOT_DIR
from .base import BaseOCREngine, OCRBox, OCRResult

_LOCAL_PROCESSOR = None
_LOCAL_MODEL = None
_LOCAL_MODEL_ID = "zai-org/GLM-OCR"


def _extract_text_recursive(obj: Any) -> str:
    chunks: list[str] = []
    if obj is None:
        return ""
    if isinstance(obj, str):
        if len(obj.strip()) >= 2:
            chunks.append(obj.strip())
    elif isinstance(obj, dict):
        priority_keys = [
            "md_results",
            "markdown",
            "text",
            "content",
            "result",
            "recognized_text",
            "layout_text",
            "data",
        ]
        for key in priority_keys:
            if key in obj:
                text = _extract_text_recursive(obj[key])
                if text:
                    chunks.append(text)
        for key, value in obj.items():
            if key not in priority_keys and isinstance(value, (dict, list)):
                text = _extract_text_recursive(value)
                if text:
                    chunks.append(text)
    elif isinstance(obj, list):
        for value in obj:
            text = _extract_text_recursive(value)
            if text:
                chunks.append(text)

    seen = set()
    output: list[str] = []
    for chunk in chunks:
        if chunk not in seen:
            output.append(chunk)
            seen.add(chunk)
    return "\n".join(output).strip()


def _glm_config() -> tuple[str, str]:
    env_file = dotenv_values(ROOT_DIR / ".env")
    api_key = (
        os.getenv("GLM_OCR_API_KEY")
        or os.getenv("ZAI_API_KEY")
        or env_file.get("GLM_OCR_API_KEY")
        or env_file.get("ZAI_API_KEY")
        or GLM_OCR_API_KEY
        or ""
    ).strip()
    endpoint = (os.getenv("GLM_OCR_ENDPOINT") or env_file.get("GLM_OCR_ENDPOINT") or GLM_OCR_ENDPOINT).strip()
    return api_key, endpoint


def _data_url_and_raw_base64(image_path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    raw_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{raw_b64}", raw_b64


def _extract_layout_boxes(obj: Any) -> list[OCRBox]:
    boxes: list[OCRBox] = []

    def walk(value: Any):
        if isinstance(value, dict):
            details = value.get("layout_details")
            if isinstance(details, list):
                for page in details:
                    if not isinstance(page, list):
                        continue
                    for item in page:
                        if not isinstance(item, dict):
                            continue
                        text = str(item.get("content") or "").strip()
                        bbox = item.get("bbox_2d")
                        width = item.get("width")
                        height = item.get("height")
                        if not text or not isinstance(bbox, list) or len(bbox) != 4:
                            continue
                        try:
                            x0, y0, x1, y1 = [float(v) for v in bbox]
                            if width and height and max(x0, y0, x1, y1) <= 1:
                                x0, x1 = x0 * float(width), x1 * float(width)
                                y0, y1 = y0 * float(height), y1 * float(height)
                            boxes.append(
                                OCRBox(
                                    text=text,
                                    confidence=None,
                                    bbox=[int(x0), int(y0), int(x1), int(y1)],
                                )
                            )
                        except Exception:
                            continue
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(obj)
    return boxes


def _local_glm_enabled() -> bool:
    value = os.getenv("GLM_OCR_LOCAL", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _run_local_hf_glm_ocr(image_path: Path) -> tuple[str, dict[str, Any]]:
    global _LOCAL_PROCESSOR, _LOCAL_MODEL

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if _LOCAL_PROCESSOR is None or _LOCAL_MODEL is None:
        _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(_LOCAL_MODEL_ID)
        _LOCAL_MODEL = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=_LOCAL_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        _LOCAL_MODEL.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path.resolve().as_posix()},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = _LOCAL_PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_LOCAL_MODEL.device)
    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated_ids = _LOCAL_MODEL.generate(**inputs, max_new_tokens=4096)
    output_text = _LOCAL_PROCESSOR.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return output_text.strip(), {"mode": "local_huggingface", "model": _LOCAL_MODEL_ID}


class GLMOCREngine(BaseOCREngine):
    name = "glm_ocr"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        start = time.perf_counter()
        api_key, endpoint = _glm_config()
        if not api_key:
            if not _local_glm_enabled():
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status="skipped",
                    error="GLM-OCR local dang tat va chua co GLM_OCR_API_KEY/ZAI_API_KEY.",
                    elapsed_sec=time.perf_counter() - start,
                )
            try:
                text, raw = _run_local_hf_glm_ocr(image_path)
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status="ok" if text else "skipped",
                    text=text,
                    boxes=[],
                    elapsed_sec=time.perf_counter() - start,
                    raw=raw,
                    error="" if text else "GLM-OCR local tra ve rong, chua co text de so sanh.",
                )
            except Exception as exc:
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status="error",
                    error=f"GLM-OCR local mien phi loi: {exc}",
                    elapsed_sec=time.perf_counter() - start,
                )

        try:
            data_url, raw_b64 = _data_url_and_raw_base64(image_path)
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "glm-ocr",
                "file": data_url,
                "return_crop_images": False,
                "need_layout_visualization": False,
            }
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=180)

            if response.status_code >= 400:
                fallback_payload = {**payload, "file": raw_b64}
                response = requests.post(endpoint, headers=headers, data=json.dumps(fallback_payload), timeout=180)
            if response.status_code >= 400:
                preview = response.text.strip().replace("\n", " ")[:500]
                return OCRResult(
                    engine=self.name,
                    variant=variant,
                    status="error",
                    error=f"GLM-OCR API tra HTTP {response.status_code}: {preview}",
                    elapsed_sec=time.perf_counter() - start,
                )

            response.raise_for_status()
            try:
                raw = response.json()
                text = _extract_text_recursive(raw)
            except Exception:
                raw = {"raw_text": response.text}
                text = response.text

            boxes = _extract_layout_boxes(raw)
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="ok" if text else "skipped",
                text=text,
                boxes=boxes,
                elapsed_sec=time.perf_counter() - start,
                raw=raw,
                error="" if text else "GLM-OCR API tra ve rong, chua co text de so sanh.",
            )
        except Exception as exc:
            return OCRResult(
                engine=self.name,
                variant=variant,
                status="error",
                error=f"GLM-OCR API loi hoac endpoint/key chua dung: {exc}",
                elapsed_sec=time.perf_counter() - start,
            )
