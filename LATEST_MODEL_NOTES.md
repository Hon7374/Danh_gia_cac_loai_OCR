# Ghi chú cập nhật model/OCR mới nhất

File này tóm tắt cách bản demo lấy bản mới nhất của từng OCR engine tại thời điểm cài đặt.

## 1. Tesseract OCR

- Python wrapper: `pytesseract` đã nằm trong `requirements.txt`.
- Binary OCR engine: không thể cài bằng pip, phải có executable `tesseract` trên máy hoặc trong Docker.
- Windows: chạy `scripts/setup_latest_windows.ps1`, script sẽ thử cài `UB-Mannheim.TesseractOCR` bằng `winget`.
- Linux/Docker: đã cài `tesseract-ocr`, `tesseract-ocr-vie`, `tesseract-ocr-eng`.
- Nếu không muốn cài local, dùng Docker: `docker compose up --build`.

## 2. EasyOCR

- Cài latest stable bằng:

```bash
python -m pip install -U easyocr
```

## 3. PaddleOCR + VietOCR

- PaddleOCR latest stable:

```bash
python -m pip install -U paddleocr paddlepaddle
```

- VietOCR latest stable:

```bash
python -m pip install -U vietocr torch torchvision
```

## 4. GLM-OCR

- Chạy qua API/key trong `.env`:

```env
ZAI_API_KEY=your_key
GLM_OCR_ENDPOINT=https://api.z.ai/api/paas/v4/layout_parsing
```

## 5. PaddleOCR-VL

Có 2 cách:

### Cách A: Python API

```python
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(device="cpu")
output = pipeline.predict("input.png")
```

### Cách B: CLI adapter trong demo

Trong `.env`:

```env
PADDLEOCR_VL_CMD=paddleocr doc_parser -i "{input}" --device cpu --save_path "{output_dir}"
```

## 6. LayoutLMv3

- Cài dependencies:

```bash
python -m pip install -U transformers torch accelerate datasets
```

- Muốn hậu xử lý thật bằng LayoutLMv3 cần model token classification đã fine-tune trên dữ liệu công văn.
- Nếu chưa có model, demo tự fallback rule/regex để vẫn chạy được.

## 7. Lệnh setup nhanh

### Windows cơ bản + tự cài Tesseract nếu có winget

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_latest_windows.ps1
```

### Windows full OCR packages

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_latest_windows.ps1 -Full
```

### Linux full OCR packages

```bash
FULL=1 bash scripts/setup_latest_linux.sh
```

### Docker không cần cài Tesseract local

```bash
docker compose up --build
```
