from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class OCRBox:
    text: str
    confidence: float | None = None
    bbox: list[int] | None = None  # [x0, y0, x1, y1]
    label: str | None = None
    polygon: list[list[int]] | None = None


@dataclass
class OCRResult:
    engine: str
    variant: str
    status: str
    text: str = ""
    boxes: list[OCRBox] | None = None
    elapsed_sec: float = 0.0
    error: str = ""
    raw: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["boxes"] = [asdict(b) for b in (self.boxes or [])]
        return d


class BaseOCREngine:
    name = "base"

    def run(self, image_path: Path, variant: str = "preprocessed") -> OCRResult:
        raise NotImplementedError
