from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import JOBS_DIR


def new_job_dir() -> tuple[str, Path]:
    job_id = uuid.uuid4().hex[:12]
    p = JOBS_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return job_id, p


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        flat.append({
            "engine": r.get("engine"),
            "variant": r.get("variant"),
            "status": r.get("status"),
            "elapsed_sec": r.get("elapsed_sec"),
            "cer": r.get("cer"),
            "wer": r.get("wer"),
            "error": r.get("error"),
            "text_preview": (r.get("text") or "")[:500],
        })
    pd.DataFrame(flat).to_csv(path, index=False, encoding="utf-8-sig")
