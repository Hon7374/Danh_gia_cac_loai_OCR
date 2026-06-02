from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def _page_profile(gray: np.ndarray) -> dict[str, float | bool]:
    vals = gray.reshape(-1)
    p5, p95 = np.percentile(vals, [5, 95])
    white_ratio = float(np.mean(vals > 235))
    dark_ratio = float(np.mean(vals < 80))
    mid_ratio = float(np.mean((vals >= 80) & (vals <= 235)))
    contrast = float(p95 - p5)
    clean_print_like = (
        white_ratio >= 0.82
        and dark_ratio >= 0.01
        and mid_ratio <= 0.14
        and contrast >= 110
    )
    low_contrast = contrast < 85
    uneven_background = white_ratio < 0.65 or mid_ratio > 0.24
    return {
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "mid_ratio": mid_ratio,
        "contrast": contrast,
        "clean_print_like": clean_print_like,
        "needs_threshold": (not clean_print_like) and (low_contrast or uneven_background),
    }


def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 245))
    if len(coords) < 100:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.2 or abs(angle) > 10:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot write output image: {path}")
    encoded.tofile(str(path))


def preprocess_image(input_image: Path, output_dir: Path) -> tuple[Path, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img = _imread_unicode(input_image)
    if img is None:
        raise ValueError("Cannot read input image")

    steps: list[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale")

    profile = _page_profile(gray)
    out = output_dir / f"{input_image.stem}_opencv_preprocessed.png"
    if profile["clean_print_like"]:
        steps.append("quality_gate_clean_raw_passthrough")
        _imwrite_unicode(out, img)
        return out, steps

    denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    steps.append("denoise_fastNlMeans")

    deskewed = _deskew(denoised)
    steps.append("deskew_minAreaRect")

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(deskewed)
    steps.append("CLAHE_contrast")

    if profile["needs_threshold"]:
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            15,
        )
        steps.append("adaptive_threshold_safe")
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        steps.append("morph_close")
    else:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
        cleaned = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
        steps.append("unsharp_mask_light")

    _imwrite_unicode(out, cleaned)
    return out, steps
