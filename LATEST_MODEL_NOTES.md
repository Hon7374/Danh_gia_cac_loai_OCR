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

- Demo mặc định dùng checkpoint official được lưu cục bộ tại `models/vietocr-pretrained/vgg_transformer.pth`.
- Checkpoint fine-tune 120 iteration cũ đã bị loại khỏi mặc định sau khi thua trên validation sạch và tài liệu test khóa; số liệu gate nằm ở `reports/vietocr_model_gate_20260712.json`.
- Runtime là hybrid theo từng crop: VietOCR chỉ được nhận khi output hợp lệ; decoder loop, lặp bệnh lý, rỗng hoặc tăng/co độ dài bất thường sẽ giữ lại Paddle.
- Crop có tỷ lệ rộng/cao từ 18 trở lên được tách tại valley khoảng trắng trước khi nhận dạng để tránh co cả dòng vào 512 px; thứ tự text dùng recursive XY-cut theo bbox cho header/footer nhiều cột.
- Config và checkpoint official đều nằm trong `models/vietocr-pretrained/`; quá trình inference không cần tải config/backbone/model từ Internet.

- PaddleOCR latest stable:

```bash
python -m pip install -U paddleocr paddlepaddle
```

- VietOCR latest stable:

```bash
python -m pip install -U vietocr torch torchvision
```

## 4. GLM-OCR (adapter legacy, không kích hoạt trong demo)

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
