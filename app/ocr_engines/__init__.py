from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .paddle_vietocr_engine import PaddleVietOCREngine
from .paddleocr_vl_engine import PaddleOCRVLEngine

ENGINE_REGISTRY = {
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
    "paddle_vietocr": PaddleVietOCREngine,
    "paddleocr_vl": PaddleOCRVLEngine,
}
