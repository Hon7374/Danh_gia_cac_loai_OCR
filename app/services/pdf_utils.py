from __future__ import annotations

from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

from app.config import OCR_MAX_IMAGE_SIDE, OCR_PDF_DPI

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}


def _resize_for_ocr(img: Image.Image) -> Image.Image:
    max_side = max(0, int(OCR_MAX_IMAGE_SIDE or 0))
    if not max_side:
        return img
    width, height = img.size
    current_max = max(width, height)
    if current_max <= max_side:
        return img
    scale = max_side / current_max
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _save_png(img: Image.Image, out: Path) -> Path:
    img = _resize_for_ocr(img.convert("RGB"))
    img.save(out, optimize=True)
    return out


def ensure_images(input_path: Path, output_dir: Path, dpi: int | None = None) -> list[Path]:
    """Convert input thành danh sách ảnh PNG để OCR toàn bộ file."""
    dpi = int(dpi or OCR_PDF_DPI or 180)
    suffix = input_path.suffix.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if suffix in IMAGE_EXTS:
        out = output_dir / f"{input_path.stem}_page1.png"
        with Image.open(input_path) as img:
            return [_save_png(img.convert("RGB"), out)]
    if suffix in PDF_EXTS:
        images: list[Path] = []
        with fitz.open(input_path) as doc:
            if len(doc) == 0:
                raise ValueError("PDF không có trang nào")
            zoom = dpi / 72
            for idx, page in enumerate(doc, start=1):
                out = output_dir / f"{input_path.stem}_page{idx}.png"
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(_save_png(img, out))
        return images
    raise ValueError(f"Định dạng không hỗ trợ: {suffix}")


def ensure_image(input_path: Path, output_dir: Path, dpi: int | None = None) -> Path:
    """Convert input thành ảnh trang đầu. Giữ lại để tương thích code cũ."""
    return ensure_images(input_path, output_dir, dpi=dpi)[0]
