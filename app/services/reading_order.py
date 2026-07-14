from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class _LayoutItem:
    original_index: int
    value: Any
    rect: tuple[float, float, float, float]
    confidence: float | None

    @property
    def cx(self) -> float:
        return (self.rect[0] + self.rect[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.rect[1] + self.rect[3]) / 2.0

    @property
    def height(self) -> float:
        return max(1.0, self.rect[3] - self.rect[1])


def _box_value(box: Any, name: str) -> Any:
    if isinstance(box, dict):
        return box.get(name)
    return getattr(box, name, None)


def _rect(box: Any) -> tuple[float, float, float, float] | None:
    bbox = _box_value(box, "bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _confidence(box: Any) -> float | None:
    value = _box_value(box, "confidence")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _projection_gaps(
    items: Sequence[_LayoutItem],
    axis: str,
) -> list[tuple[float, float]]:
    if axis == "x":
        intervals = sorted((item.rect[0], item.rect[2]) for item in items)
    else:
        intervals = sorted((item.rect[1], item.rect[3]) for item in items)
    if len(intervals) < 2:
        return []

    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [
        (right_start - left_end, (left_end + right_start) / 2.0)
        for (_, left_end), (right_start, _) in zip(merged, merged[1:])
        if right_start > left_end
    ]


def _trusted_for_layout(items: Sequence[_LayoutItem]) -> list[_LayoutItem]:
    """Ignore weak stamp/signature detections only while finding whitespace.

    The ignored boxes are retained in the final OCR output.  Excluding them from
    projection profiles prevents a red seal or blue signature from bridging the
    genuine whitespace between a recipient list and a signature column.
    """

    trusted = [
        item
        for item in items
        if item.confidence is None or item.confidence >= 0.40
    ]
    return trusted if len(trusted) >= 2 else list(items)


def _strongest_cut(
    items: Sequence[_LayoutItem],
    axis: str,
    page_width: int,
    page_height: int,
) -> tuple[float, float] | None:
    trusted = _trusted_for_layout(items)
    gaps = sorted(_projection_gaps(trusted, axis), reverse=True)
    if not gaps:
        return None

    typical_height = median(item.height for item in trusted)
    if axis == "y":
        # A gap of roughly one text-line height is a reliable block boundary.
        # The small page-relative floor keeps the rule stable across DPI values.
        minimum = max(10.0, page_height * 0.012, typical_height * 0.72)
    else:
        # Column cuts are deliberately stricter than horizontal block cuts.
        # Ordinary spaces within a line must never turn a one-column body into
        # multiple columns.
        minimum = max(20.0, page_width * 0.035, typical_height * 1.10)

    for gap, split in gaps:
        if gap < minimum:
            break
        if axis == "y":
            before = [item for item in items if item.cy < split]
            after = [item for item in items if item.cy >= split]
        else:
            before = [item for item in items if item.cx < split]
            after = [item for item in items if item.cx >= split]
        if before and after:
            return gap, split
    return None


def _xy_cut(
    items: Sequence[_LayoutItem],
    page_width: int,
    page_height: int,
    counters: dict[str, int],
) -> list[_LayoutItem]:
    if len(items) <= 1:
        return list(items)

    # Horizontal whitespace defines independent top-to-bottom regions first.
    # This is important for administrative headers: the authority/motto pair
    # is a two-column block, while the document number/date below it is another
    # pair and must not be folded into the first column.
    horizontal = _strongest_cut(items, "y", page_width, page_height)
    if horizontal is not None:
        _, split = horizontal
        top = [item for item in items if item.cy < split]
        bottom = [item for item in items if item.cy >= split]
        counters["horizontal_cuts"] += 1
        return _xy_cut(top, page_width, page_height, counters) + _xy_cut(
            bottom, page_width, page_height, counters
        )

    # Within a region, a persistent vertical whitespace channel is treated as
    # a column boundary.  Each column is read fully before the next one.
    vertical = _strongest_cut(items, "x", page_width, page_height)
    if vertical is not None:
        _, split = vertical
        left = [item for item in items if item.cx < split]
        right = [item for item in items if item.cx >= split]
        counters["vertical_cuts"] += 1
        return _xy_cut(left, page_width, page_height, counters) + _xy_cut(
            right, page_width, page_height, counters
        )

    # A one-column body has no strong vertical channel.  Preserve the detector
    # order exactly instead of second-guessing closely spaced/italic lines.
    return sorted(items, key=lambda item: item.original_index)


def order_boxes_xy_cut(
    boxes: Sequence[T],
    page_width: int,
    page_height: int,
) -> tuple[list[T], dict[str, Any]]:
    """Return OCR boxes in block-aware reading order using geometry only.

    The routine never changes text, confidence, geometry, or box membership.
    Invalid/missing geometries stay in their original slots.  Only boxes with a
    valid rectangle participate in recursive XY-cut ordering.
    """

    original = list(boxes)
    diagnostics: dict[str, Any] = {
        "policy": "recursive-xy-cut-v1",
        "page_width": int(page_width or 0),
        "page_height": int(page_height or 0),
        "box_count": len(original),
        "valid_geometry_boxes": 0,
        "invalid_geometry_boxes": 0,
        "horizontal_cuts": 0,
        "vertical_cuts": 0,
        "changed_positions": 0,
        "applied": False,
    }
    if len(original) < 2 or page_width <= 0 or page_height <= 0:
        return original, diagnostics

    items: list[_LayoutItem] = []
    for index, box in enumerate(original):
        rect = _rect(box)
        if rect is None:
            continue
        items.append(
            _LayoutItem(
                original_index=index,
                value=box,
                rect=rect,
                confidence=_confidence(box),
            )
        )
    diagnostics["valid_geometry_boxes"] = len(items)
    diagnostics["invalid_geometry_boxes"] = len(original) - len(items)
    if len(items) < 2:
        return original, diagnostics

    counters = {"horizontal_cuts": 0, "vertical_cuts": 0}
    ordered_items = _xy_cut(items, int(page_width), int(page_height), counters)

    # Leave invalid boxes anchored at their source positions and reorder only
    # the valid slots.  This avoids dropping or relocating nonstandard engine
    # output while retaining a deterministic text/box sequence.
    output = list(original)
    valid_slots = [item.original_index for item in items]
    for slot, item in zip(valid_slots, ordered_items):
        output[slot] = item.value

    diagnostics.update(counters)
    diagnostics["changed_positions"] = sum(
        1 for before, after in zip(original, output) if before is not after
    )
    diagnostics["applied"] = diagnostics["changed_positions"] > 0
    return output, diagnostics
