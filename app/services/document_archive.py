from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image

from app.config import STORAGE_DIR

WORD_LAYOUT_VERSION = "2026-05-26-structured-v2"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    return path.relative_to(STORAGE_DIR).as_posix()


def _safe_part(value: str, fallback: str = "unknown") -> str:
    value = re.sub(r"[^\w.\-]+", "_", value.strip(), flags=re.UNICODE)
    value = value.strip("._-")
    return value or fallback


def _copy_file(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": _rel(dst),
        "filename": dst.name,
        "size_bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return _rel(path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")
    return _rel(path)


def _word_paragraph(text: str = "", bold: bool = False) -> str:
    if not text:
        return "<w:p/>"
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:p><w:r>"
        f"{props}<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r></w:p>"
    )


def _write_docx(path: Path, title: str, metadata: dict[str, Any], body_text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = [_word_paragraph(title, bold=True), _word_paragraph()]
    for key, value in metadata.items():
        paragraphs.append(_word_paragraph(f"{key}: {value}"))
    paragraphs.extend([_word_paragraph(), _word_paragraph("Nội dung OCR", bold=True), _word_paragraph()])
    for line in (body_text or "").splitlines():
        paragraphs.append(_word_paragraph(line))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraphs)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

    return {
        "path": _rel(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def _run_text(text: str, bold: bool = False, italic: bool = False, size: int = 28) -> str:
    bold_xml = "<w:b/>" if bold else ""
    italic_xml = "<w:i/>" if italic else ""
    props = (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        f"{bold_xml}{italic_xml}<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
    )
    return f'<w:r>{props}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _editable_paragraph(
    text: str,
    align: str = "left",
    bold: bool = False,
    italic: bool = False,
    indent_twips: int = 0,
    first_line_twips: int = 0,
    before_twips: int = 0,
    after_twips: int = 80,
    size: int = 28,
) -> str:
    if not text.strip():
        return '<w:p><w:pPr><w:spacing w:after="80"/></w:pPr></w:p>'
    align_xml = "" if align == "left" else f'<w:jc w:val="{align}"/>'
    indent_parts = []
    if indent_twips > 0 and align in {"left", "both"}:
        indent_parts.append(f'w:left="{indent_twips}"')
    if first_line_twips > 0 and align in {"left", "both"}:
        indent_parts.append(f'w:firstLine="{first_line_twips}"')
    indent_xml = f"<w:ind {' '.join(indent_parts)}/>" if indent_parts else ""
    return (
        "<w:p>"
        f"<w:pPr>{align_xml}{indent_xml}<w:spacing w:before=\"{before_twips}\" w:after=\"{after_twips}\"/></w:pPr>"
        f"{_run_text(text, bold=bold, italic=italic, size=size)}"
        "</w:p>"
    )


def _bbox(box: dict[str, Any]) -> list[float] | None:
    raw = box.get("bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (TypeError, ValueError):
        return None


def _group_boxes_to_lines(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for box in boxes:
        b = _bbox(box)
        text = str(box.get("text") or "").strip()
        if not b or not text:
            continue
        normalized.append({"text": text, "bbox": b, "cy": (b[1] + b[3]) / 2, "height": max(1.0, b[3] - b[1])})
    normalized.sort(key=lambda item: (item["cy"], item["bbox"][0]))

    lines: list[list[dict[str, Any]]] = []
    for item in normalized:
        if not lines:
            lines.append([item])
            continue
        current = lines[-1]
        avg_y = sum(x["cy"] for x in current) / len(current)
        avg_h = sum(x["height"] for x in current) / len(current)
        threshold = max(8.0, avg_h * 0.65)
        if item["height"] > 80 or avg_h > 80:
            threshold = max(8.0, min(40.0, avg_h * 0.3))
        if abs(item["cy"] - avg_y) <= threshold:
            current.append(item)
        else:
            lines.append([item])

    grouped = []
    for line in lines:
        line.sort(key=lambda item: item["bbox"][0])
        min_x = min(item["bbox"][0] for item in line)
        min_y = min(item["bbox"][1] for item in line)
        max_x = max(item["bbox"][2] for item in line)
        max_y = max(item["bbox"][3] for item in line)
        text = " ".join(item["text"] for item in line).strip()
        if text:
            grouped.append({"text": text, "bbox": [min_x, min_y, max_x, max_y], "items": line})
    return grouped


def _clean_ocr_line(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    if text in {"number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"}:
        return ""
    return text


def _page_width_from_lines(lines: list[dict[str, Any]]) -> float:
    if not lines:
        return 1000.0
    max_x = max(line["bbox"][2] for line in lines)
    min_x = min(line["bbox"][0] for line in lines)
    return max(1000.0, max_x + max(0.0, min_x))


def _line_alignment(line: dict[str, Any], page_width: float) -> tuple[str, int]:
    text = _clean_ocr_line(line["text"])
    b = line["bbox"]
    width = max(1.0, b[2] - b[0])
    center = (b[0] + b[2]) / 2
    if _looks_like_heading(text):
        return "center", 0
    if b[0] > page_width * 0.56:
        return "right", 0
    if width < page_width * 0.74 and abs(center - page_width / 2) < page_width * 0.16:
        return "center", 0
    indent = int(max(0, min(1800, (b[0] / max(page_width, 1)) * 6200)))
    return "both", indent


def _looks_like_heading(text: str) -> bool:
    stripped = _clean_ocr_line(text)
    if not stripped:
        return False
    upper_ratio = sum(1 for c in stripped if c.isupper()) / max(1, sum(1 for c in stripped if c.isalpha()))
    lower = stripped.lower()
    if (
        upper_ratio > 0.72
        or lower.startswith(("cộng hòa", "ngân hàng", "thông tư", "chương ", "điều "))
        or lower in {"độc lập - tự do - hạnh phúc", "độc lập- tự do - hạnh phúc"}
    ):
        return True
    return upper_ratio > 0.72 or stripped.lower().startswith(("cộng hòa", "ngân hàng", "thông tư", "chương "))


def _split_line_segments(line: dict[str, Any], page_width: float) -> list[dict[str, Any]]:
    items = line.get("items") or []
    if len(items) < 2:
        return [{"text": line["text"], "bbox": line["bbox"]}]
    segments: list[list[dict[str, Any]]] = [[items[0]]]
    has_block_text = any("\n" in item["text"] for item in items)
    threshold = 45.0 if has_block_text else max(page_width * 0.12, 90.0)
    for item in items[1:]:
        prev = segments[-1][-1]
        gap = item["bbox"][0] - prev["bbox"][2]
        if gap >= threshold:
            segments.append([item])
        else:
            segments[-1].append(item)
    result = []
    for segment in segments:
        text = " ".join(part["text"] for part in segment).strip()
        if not text:
            continue
        result.append(
            {
                "text": text,
                "bbox": [
                    min(part["bbox"][0] for part in segment),
                    min(part["bbox"][1] for part in segment),
                    max(part["bbox"][2] for part in segment),
                    max(part["bbox"][3] for part in segment),
                ],
            }
        )
    return result or [{"text": line["text"], "bbox": line["bbox"]}]


def _two_column_row(left_text: str, right_text: str, bold: bool = False, italic: bool = False) -> str:
    def cell(text: str, align: str, width: int) -> str:
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
            f'{_editable_paragraph(text, align=align, bold=bold, italic=italic, after_twips=0)}'
            "</w:tc>"
        )

    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
        '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
        "</w:tblPr><w:tblGrid><w:gridCol w:w=\"4800\"/><w:gridCol w:w=\"5200\"/></w:tblGrid>"
        f"<w:tr>{cell(left_text, 'center', 4800)}{cell(right_text, 'center', 5200)}</w:tr></w:tbl>"
    )


def _pages_from_boxes(boxes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not boxes or not all(isinstance(box, dict) and box.get("page") for box in boxes):
        return []
    pages: dict[int, list[dict[str, Any]]] = {}
    for box in boxes:
        try:
            page = int(box.get("page"))
        except (TypeError, ValueError):
            continue
        pages.setdefault(page, []).append(box)
    return [_group_boxes_to_lines(pages[page]) for page in sorted(pages)]


def _layout_boxes_from_raw(row: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []

    def visit(value: Any, page: int = 1) -> None:
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
                text = re.sub(r"\n?#################\s*$", "", match.group("content").strip()).strip()
                if text:
                    boxes.append({"text": text, "bbox": bbox, "page": page, "label": match.group("label").strip()})
            return
        if isinstance(value, dict):
            current_page = page
            try:
                current_page = int(value.get("page") or value.get("page_index") or page)
            except (TypeError, ValueError):
                current_page = page
            for child in value.values():
                if isinstance(child, (dict, list, str)):
                    visit(child, current_page)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value, start=1):
                visit(child, idx if isinstance(child, str) else page)

    visit(row.get("raw"))
    return boxes


def _pages_from_text(text: str) -> list[list[str]]:
    chunks = [chunk for chunk in re.split(r"\f+|\n\s*---+\s*page\s*\d*\s*---+\s*\n", text or "", flags=re.IGNORECASE) if chunk.strip()]
    if not chunks:
        chunks = [text or ""]
    pages = []
    for chunk in chunks:
        lines = [line.rstrip() for line in chunk.splitlines()]
        pages.append(lines if lines else [""])
    return pages


def _write_editable_ocr_docx(path: Path, title: str, row: dict[str, Any], body_text: str) -> dict[str, Any]:
    """Create an editable Word file from OCR text/boxes, approximating the scan layout without embedding page images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = []

    source_boxes = [box for box in row.get("boxes") or [] if isinstance(box, dict)]
    if not source_boxes:
        source_boxes = _layout_boxes_from_raw(row)
    box_pages = _pages_from_boxes(source_boxes)
    if box_pages:
        for page_idx, lines in enumerate(box_pages):
            if page_idx:
                paragraphs.append(_page_break())
            page_width = _page_width_from_lines(lines)
            for line in lines:
                segments = _split_line_segments(line, page_width)
                if len(segments) == 2 and segments[0]["bbox"][0] < page_width * 0.46 and segments[1]["bbox"][0] > page_width * 0.42:
                    left = _clean_ocr_line(segments[0]["text"])
                    right = _clean_ocr_line(segments[1]["text"])
                    if left and right:
                        paragraphs.append(_two_column_row(left, right, bold=True))
                    continue
                text = _clean_ocr_line(line["text"])
                if not text:
                    continue
                align, indent = _line_alignment({**line, "text": text}, page_width)
                heading = _looks_like_heading(text)
                first_line = 560 if align == "both" and indent < 900 and not heading else 0
                paragraphs.append(
                    _editable_paragraph(
                        text,
                        align=align,
                        bold=heading,
                        italic=text.lower().startswith(("căn cứ", "theo đề nghị")),
                        indent_twips=indent,
                        first_line_twips=first_line,
                        before_twips=120 if heading else 0,
                        after_twips=80,
                    )
                )
    else:
        for page_idx, lines in enumerate(_pages_from_text(body_text)):
            if page_idx:
                paragraphs.append(_page_break())
            for line in lines:
                text = _clean_ocr_line(line)
                if not text:
                    continue
                heading = _looks_like_heading(text)
                align = "center" if heading else "both"
                paragraphs.append(_editable_paragraph(text, align=align, bold=heading, first_line_twips=560 if not heading else 0))

    if not paragraphs:
        paragraphs = [_editable_paragraph(title, align="center", bold=True)]

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraphs)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="0" w:footer="0" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

    return {
        "path": _rel(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _image_ext(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        return "jpg"
    if ext in {"png", "gif", "bmp", "webp"}:
        return ext
    return "png"


def _image_content_type(ext: str) -> str:
    return "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"


def _fit_image_emu(path: Path) -> tuple[int, int]:
    # A4 portrait in twips, with small margins to keep the scanned page close to the PDF view.
    page_w_twips = 11906
    page_h_twips = 16838
    margin_twips = 360
    max_w_emu = (page_w_twips - margin_twips * 2) * 635
    max_h_emu = (page_h_twips - margin_twips * 2) * 635
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        width, height = 1000, 1414
    ratio = min(max_w_emu / max(width, 1), max_h_emu / max(height, 1))
    return int(width * ratio), int(height * ratio)


def _image_drawing_xml(rel_id: str, image_index: int, filename: str, cx: int, cy: int) -> str:
    name = escape(filename)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{image_index}" name="Page {image_index}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{image_index}" name="{name}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>"""


def _write_layout_docx(path: Path, page_paths: list[Path], title: str) -> dict[str, Any]:
    """Create a Word file that preserves the source layout by placing each scanned page as an image."""
    if not page_paths:
        return _write_docx(path, title=title, metadata={"Ghi chú": "Không có ảnh trang để giữ bố cục"}, body_text="")

    path.parent.mkdir(parents=True, exist_ok=True)
    relationships = []
    drawings = []
    media_entries: list[tuple[Path, str]] = []
    defaults = {
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
        "xml": "application/xml",
    }

    for idx, page_path in enumerate(page_paths, start=1):
        ext = _image_ext(page_path)
        defaults[ext] = _image_content_type(ext)
        media_name = f"image{idx:03d}.{ext}"
        rel_id = f"rId{idx}"
        relationships.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
        )
        cx, cy = _fit_image_emu(page_path)
        drawings.append(_image_drawing_xml(rel_id, idx, page_path.name, cx, cy))
        if idx != len(page_paths):
            drawings.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        media_entries.append((page_path, f"word/media/{media_name}"))

    default_xml = "\n".join(
        f'  <Default Extension="{ext}" ContentType="{content_type}"/>'
        for ext, content_type in defaults.items()
    )
    content_types_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
{default_xml}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdOfficeDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(relationships)}
</Relationships>"""
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {''.join(drawings)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="360" w:right="360" w:bottom="360" w:left="360" w:header="0" w:footer="0" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)
        zf.writestr("word/document.xml", document_xml)
        for source_path, archive_name in media_entries:
            zf.write(source_path, archive_name)

    return {
        "path": _rel(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _upsert_index(index_path: Path, item: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("document_id") != item.get("document_id"):
                rows.append(row)
    rows.append(item)
    index_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def sync_archive_metadata(manifest_path: Path, report: dict[str, Any]) -> bool:
    """Sync stored dossier metadata from the latest validated OCR report fields."""
    if not manifest_path.exists() or not manifest_path.is_file():
        return False
    fields = ((report.get("layoutlmv3_postprocess") or {}).get("fields") or {})
    if not isinstance(fields, dict) or not fields:
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    changed = False
    extracted_rel = manifest.get("extracted_fields_path")
    if extracted_rel:
        extracted_path = STORAGE_DIR / str(extracted_rel)
        try:
            current_fields = json.loads(extracted_path.read_text(encoding="utf-8")) if extracted_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            current_fields = {}
        if current_fields != fields:
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            extracted_path.write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")
            changed = True

    workflow_rel = manifest.get("workflow_manifest_path")
    workflow = {}
    if workflow_rel:
        workflow_path = STORAGE_DIR / str(workflow_rel)
        try:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8")) if workflow_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            workflow = {}
        updates = {
            "document_type": fields.get("loai_van_ban") or "",
            "document_number": fields.get("so_ky_hieu") or "",
            "issued_date": fields.get("ngay_ban_hanh") or "",
            "issuing_agency": fields.get("co_quan_ban_hanh") or "",
            "sender": fields.get("noi_gui") or "",
            "receiver": fields.get("noi_nhan") or "",
        }
        before = dict(workflow)
        workflow.update(updates)
        audit_events = workflow.setdefault("audit_events", [])
        if workflow != before:
            audit_events.append(
                {
                    "at": _now_iso(),
                    "event": "metadata_synced",
                    "actor": "layoutlmv3_guard",
                    "job_id": manifest.get("job_id"),
                }
            )
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
            changed = True

    index_item = {
        "document_id": manifest.get("document_id"),
        "job_id": manifest.get("job_id"),
        "created_at": manifest.get("created_at"),
        "source_filename": manifest.get("source_filename"),
        "page_count": manifest.get("page_count"),
        "status": manifest.get("status"),
        "document_number": fields.get("so_ky_hieu") or "",
        "issued_date": fields.get("ngay_ban_hanh") or "",
        "document_type": fields.get("loai_van_ban") or "",
        "issuing_agency": fields.get("co_quan_ban_hanh") or "",
        "subject": fields.get("trich_yeu") or "",
        "manifest_path": _rel(manifest_path),
        "storage_root": manifest.get("storage_root"),
    }
    if index_item.get("document_id"):
        _upsert_index(STORAGE_DIR / "documents" / "index.jsonl", index_item)
        changed = True

    return changed


def _word_metadata_for_row(row: dict[str, Any], page_count: int) -> dict[str, Any]:
    return {
        "Engine": row.get("engine") or "",
        "Input": row.get("variant") or "",
        "Status": row.get("status") or "",
        "Số trang": page_count,
        "Thời gian xử lý": f"{float(row.get('elapsed_sec') or 0):.2f}s",
        "Độ dài text": len(row.get("text") or ""),
    }


def ensure_archive_word_outputs(manifest_path: Path) -> bool:
    """Backfill Word outputs for an existing archive manifest."""
    if not manifest_path.exists() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    archive_root = STORAGE_DIR / str(manifest.get("storage_root") or "")
    if not archive_root.exists():
        return False

    changed = False
    for output in manifest.get("ocr_outputs") or []:
        text = ""
        if output.get("text_path"):
            text_path = STORAGE_DIR / str(output["text_path"])
            if text_path.exists():
                text = text_path.read_text(encoding="utf-8", errors="ignore")

        engine = _safe_part(str(output.get("engine") or "engine"))
        variant = _safe_part(str(output.get("variant") or "variant"))
        base_name = f"{engine}__{variant}"
        word_target = archive_root / "08_word_outputs" / f"{base_name}.docx"
        should_rebuild = (
            output.get("word_kind") != "editable_ocr_layout"
            or output.get("word_layout_version") != WORD_LAYOUT_VERSION
            or output.get("word_path") != _rel(word_target)
            or not word_target.exists()
        )
        source_row = output
        if output.get("json_path"):
            json_path = STORAGE_DIR / str(output["json_path"])
            if json_path.exists():
                try:
                    loaded = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        source_row = loaded
                except json.JSONDecodeError:
                    source_row = output
        word_file = {
            "path": _rel(word_target),
            "filename": word_target.name,
            "size_bytes": word_target.stat().st_size,
            "sha256": _sha256(word_target),
        } if word_target.exists() and not should_rebuild else _write_editable_ocr_docx(
            word_target,
            title=f"Văn bản OCR editable - {output.get('engine')} / {output.get('variant')}",
            row=source_row,
            body_text=text,
        )
        before = dict(output)
        output["word_path"] = word_file["path"]
        output["word_file"] = word_file
        output["word_kind"] = "editable_ocr_layout"
        output["word_layout_version"] = WORD_LAYOUT_VERSION
        output["word_note"] = "Word editable dựng từ OCR text/box; có thể sửa trực tiếp trong Word. Bố cục được mô phỏng theo dòng/trang OCR, không nhúng ảnh scan làm nội dung chính."
        output["text_length"] = output.get("text_length") or len(text)
        changed = changed or output != before

    if changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def archive_scan(job_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Persist a scan result in a management-friendly document archive layout."""
    job_id = report.get("job_id") or job_dir.name
    created_at = _now_iso()
    created = datetime.now()
    document_id = _safe_part(str(job_id), "document")
    archive_root = STORAGE_DIR / "documents" / f"{created:%Y}" / f"{created:%m}" / document_id

    fields = ((report.get("layoutlmv3_postprocess") or {}).get("fields") or {})
    source_rel = report.get("uploaded_file") or ""
    source_path = job_dir / source_rel if source_rel else None
    source_name = Path(source_rel).name if source_rel else "uploaded_file"

    original = _copy_file(source_path, archive_root / "01_original" / source_name) if source_path else None

    raw_pages = []
    for idx, rel_path in enumerate(report.get("raw_images") or [], start=1):
        copied = _copy_file(job_dir / rel_path, archive_root / "02_pages" / "raw" / f"page_{idx:03d}.png")
        if copied:
            copied["page"] = idx
            raw_pages.append(copied)

    preprocessed_pages = []
    for idx, rel_path in enumerate(report.get("preprocessed_images") or [], start=1):
        copied = _copy_file(job_dir / rel_path, archive_root / "02_pages" / "opencv_preprocessed" / f"page_{idx:03d}.png")
        if copied:
            copied["page"] = idx
            preprocessed_pages.append(copied)

    ground_truth = None
    gt_meta = report.get("ground_truth_file") or {}
    if gt_meta.get("relative_path"):
        gt_src = job_dir / gt_meta["relative_path"]
        ground_truth = _copy_file(gt_src, archive_root / "03_ground_truth" / Path(gt_meta["relative_path"]).name)

    ocr_outputs = []
    for row in report.get("results") or []:
        engine = _safe_part(str(row.get("engine") or "engine"))
        variant = _safe_part(str(row.get("variant") or "variant"))
        base_name = f"{engine}__{variant}"
        text = row.get("text") or ""
        text_path = _write_text(archive_root / "04_ocr_text" / f"{base_name}.txt", text)
        json_path = _write_json(archive_root / "05_ocr_json" / f"{base_name}.json", row)
        page_count = ((row.get("raw") or {}).get("page_count") or report.get("page_count") or 1)
        word_file = _write_editable_ocr_docx(
            archive_root / "08_word_outputs" / f"{base_name}.docx",
            title=f"Văn bản OCR editable - {row.get('engine')} / {row.get('variant')}",
            row=row,
            body_text=text,
        )
        ocr_outputs.append(
            {
                "engine": row.get("engine"),
                "variant": row.get("variant"),
                "status": row.get("status"),
                "elapsed_sec": row.get("elapsed_sec"),
                "text_length": len(text),
                "text_path": text_path,
                "json_path": json_path,
                "word_path": word_file["path"],
                "word_file": word_file,
                "word_kind": "editable_ocr_layout",
                "word_layout_version": WORD_LAYOUT_VERSION,
                "word_note": "Word editable dựng từ OCR text/box; có thể sửa trực tiếp trong Word. Bố cục được mô phỏng theo dòng/trang OCR, không nhúng ảnh scan làm nội dung chính.",
                "page_count": page_count,
            }
        )

    fields_path = _write_json(archive_root / "06_metadata" / "extracted_fields.json", fields)
    workflow = {
        "document_id": document_id,
        "job_id": job_id,
        "status": "scanned",
        "direction": "unclassified",
        "document_type": fields.get("loai_van_ban") or "",
        "document_number": fields.get("so_ky_hieu") or "",
        "issued_date": fields.get("ngay_ban_hanh") or "",
        "issuing_agency": fields.get("co_quan_ban_hanh") or "",
        "sender": fields.get("noi_gui") or "",
        "receiver": fields.get("noi_nhan") or "",
        "owning_unit": "",
        "current_handler": "",
        "review_status": "pending_review",
        "retention_policy": "local_demo",
        "audit_events": [
            {
                "at": created_at,
                "event": "scan_archived",
                "actor": "local_ocr_demo",
                "job_id": job_id,
            }
        ],
    }
    workflow_path = _write_json(archive_root / "06_metadata" / "workflow_manifest.json", workflow)

    exports = {
        "benchmark_csv": None,
        "comparison_summary": None,
        "report_snapshot": _write_json(archive_root / "07_exports" / "report_snapshot.json", report),
    }
    csv_copy = _copy_file(job_dir / "benchmark_results.csv", archive_root / "07_exports" / "benchmark_results.csv")
    summary_copy = _copy_file(job_dir / "comparison_summary.json", archive_root / "07_exports" / "comparison_summary.json")
    if csv_copy:
        exports["benchmark_csv"] = csv_copy["path"]
    if summary_copy:
        exports["comparison_summary"] = summary_copy["path"]

    manifest = {
        "document_id": document_id,
        "job_id": job_id,
        "created_at": created_at,
        "source_filename": source_name,
        "page_count": report.get("page_count") or len(raw_pages) or 1,
        "status": "scanned",
        "storage_root": _rel(archive_root),
        "original_file": original,
        "pages": {
            "raw": raw_pages,
            "opencv_preprocessed": preprocessed_pages,
        },
        "ground_truth": ground_truth,
        "extracted_fields_path": fields_path,
        "workflow_manifest_path": workflow_path,
        "ocr_outputs": ocr_outputs,
        "exports": exports,
    }
    manifest_path = _write_json(archive_root / "manifest.json", manifest)

    index_item = {
        "document_id": document_id,
        "job_id": job_id,
        "created_at": created_at,
        "source_filename": source_name,
        "page_count": manifest["page_count"],
        "status": manifest["status"],
        "document_number": workflow["document_number"],
        "issued_date": workflow["issued_date"],
        "document_type": workflow["document_type"],
        "issuing_agency": workflow["issuing_agency"],
        "subject": fields.get("trich_yeu") or "",
        "manifest_path": manifest_path,
        "storage_root": manifest["storage_root"],
    }
    _upsert_index(STORAGE_DIR / "documents" / "index.jsonl", index_item)

    return {
        "document_id": document_id,
        "status": "scanned",
        "storage_root": manifest["storage_root"],
        "manifest_path": manifest_path,
        "index_path": "documents/index.jsonl",
        "workflow_manifest_path": workflow_path,
        "extracted_fields_path": fields_path,
        "ocr_output_count": len(ocr_outputs),
        "created_at": created_at,
    }
