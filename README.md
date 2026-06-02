# OCR Full Benchmark Demo v4 — upload file text chuẩn + dashboard so sánh OCR

Demo này dùng để chạy thử trực tiếp pipeline OCR cho đề tài NCKH:

**PDF/ảnh công văn → tiền xử lý OpenCV → chạy nhiều OCR engine → so CER/WER → hậu xử lý LayoutLMv3-ready → trích xuất trường công văn.**

Bản v2 đã sửa theo yêu cầu:

- Không bắt buộc máy đã cài sẵn Tesseract binary; có script tự kiểm tra/cài hoặc chạy bằng Docker.
- Không ghim cứng version cũ của EasyOCR/PaddleOCR/VietOCR/Transformers; riêng PaddlePaddle/einops có pin tương thích để PaddleOCR-VL chạy ổn trên Windows CPU.
- Cập nhật adapter PaddleOCR-VL theo hướng `from paddleocr import PaddleOCRVL` và CLI `paddleocr doc_parser`.
- Cập nhật endpoint mặc định GLM-OCR/Z.ai layout parsing.
- Có `scripts/check_environment.py` để kiểm tra model/package/binary trước khi demo.

## 1. Các OCR engine có trong demo

| Engine | Chạy thế nào | Ghi chú |
|---|---|---|
| Tesseract OCR | Local/offline hoặc Docker | Nếu máy chưa có binary, script có thể cài qua winget; Docker thì có sẵn |
| EasyOCR | Optional local/offline | Cài bằng `pip install -U easyocr` |
| PaddleOCR + VietOCR | Optional local/offline | PaddleOCR detect/recognize; VietOCR hỗ trợ tiếng Việt |
| GLM-OCR | API/cloud hoặc endpoint riêng | Cấu hình `ZAI_API_KEY` hoặc `GLM_OCR_API_KEY` trong `.env` |
| PaddleOCR-VL | Optional local/API/CLI | Hỗ trợ Python API hoặc `PADDLEOCR_VL_CMD` |
| LayoutLMv3 | Optional post-processing | Có model fine-tuned thì chạy thật, chưa có thì fallback rule/regex |

## 2. Cách chạy nhanh nếu chưa cài Tesseract trên Windows

Mở PowerShell tại thư mục project:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_latest_windows.ps1
```

Sau đó chạy:

```bat
run_windows.bat
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Nếu muốn cài luôn các OCR optional mới nhất:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_latest_windows.ps1 -Full
```

Lưu ý: `-Full` có thể rất lâu vì cài PyTorch, PaddleOCR, EasyOCR, VietOCR, Transformers và PaddleOCR-VL. Lần đầu chạy model có thể tải thêm vài GB vào cache người dùng.

## 3. Cách chạy không cần cài Tesseract vào máy: Docker

Nếu máy có Docker Desktop:

```bash
docker compose up --build
```

Docker image đã cài:

- `tesseract-ocr`
- `tesseract-ocr-vie`
- `tesseract-ocr-eng`
- Python requirements cơ bản

Sau đó mở:

```text
http://127.0.0.1:8000
```

Nếu muốn Docker có cả optional OCR packages, dùng file `docker/Dockerfile.full-ocr` để build riêng. Image này rất nặng.

## 4. Cài các OCR optional mới nhất

```bash
python -m pip install -U -r requirements-optional-ocr.txt
python -m pip install -U "einops>=0.8.1" "chardet<6" "numpy==2.2.1" "pillow==10.2.0"
```

Nếu bị xung đột package, dùng bản ổn định hơn:

```bash
python -m pip install -r requirements-optional-stable.txt
python -m pip install -U "einops>=0.8.1" "chardet<6" "numpy==2.2.1" "pillow==10.2.0"
```

## 5. Kiểm tra môi trường trước khi demo

```bash
python scripts/check_environment.py
```

Script sẽ kiểm tra:

- Python version
- Tesseract binary có hay chưa
- Language data `vie`, `eng`
- Version các package: EasyOCR, PaddleOCR, VietOCR, Torch, Transformers
- Biến môi trường GLM-OCR, PaddleOCR-VL, LayoutLMv3

## 6. Cấu hình GLM-OCR

Copy file môi trường:

```bash
cp .env.example .env
```

Điền:

```env
ZAI_API_KEY=your_key_here
GLM_OCR_ENDPOINT=https://api.z.ai/api/paas/v4/layout_parsing
```

Hoặc dùng biến cũ:

```env
GLM_OCR_API_KEY=your_key_here
```

## 7. Cấu hình PaddleOCR-VL

### Cách 1: để app tự import Python API

Cài PaddleOCR mới, sau đó app sẽ thử:

```python
from paddleocr import PaddleOCRVL
```

### Cách 2: dùng CLI adapter bền hơn

Trong `.env`:

```env
PADDLEOCR_VL_CMD=paddleocr doc_parser -i "{input}" --device cpu --save_path "{output_dir}"
```

`{input}` và `{output_dir}` sẽ được app tự thay.

## 8. Hậu xử lý LayoutLMv3

LayoutLMv3 không tự biết trường công văn nếu chưa fine-tune. Muốn chạy thật cần model token classification đã huấn luyện trên nhãn như:

- `SO_KY_HIEU`
- `NGAY_BAN_HANH`
- `TRICH_YEU`
- `NOI_GUI`
- `NOI_NHAN`

Nếu đã có model:

```env
LAYOUTLMV3_MODEL_DIR=./models/layoutlmv3-congvan-token-classification
```

Nếu chưa có, demo tự fallback rule/regex để vẫn có kết quả.

## 9. File mẫu

```text
demo_samples/
├── sample_cong_van_scan.png
├── sample_cong_van_scan.pdf
├── ground_truth_text/sample_cong_van.txt
└── ground_truth_fields/sample_cong_van.json
```

Upload `sample_cong_van_scan.png` hoặc `sample_cong_van_scan.pdf`, rồi copy nội dung `ground_truth_text/sample_cong_van.txt` vào ô Ground Truth để tính CER/WER.

## 10. Kết quả đầu ra

Mỗi lần chạy tạo một job trong:

```text
jobs/<job_id>/
```

Gồm:

```text
report.json
benchmark_results.csv
images/*_page1.png
images/*_opencv_preprocessed.png
```

CSV dùng để đưa vào báo cáo/Excel:

| engine | variant | status | elapsed_sec | cer | wer | error | text_preview |
|---|---|---|---:|---:|---:|---|---|

## 11. Giới hạn cần nói thật khi bảo vệ

- Tesseract cần binary ngoài Python; bản v2 đã có script cài/check và Docker để tránh lỗi máy chưa cài.
- EasyOCR/PaddleOCR/VietOCR/Transformers là gói nặng, nên không ép cài trong lần chạy cơ bản.
- GLM-OCR cần API key hoặc môi trường triển khai riêng.
- PaddleOCR-VL model nặng; nên demo qua CLI/API adapter hoặc máy có GPU/RAM đủ.
- LayoutLMv3 muốn chính xác phải fine-tune bằng dữ liệu công văn đã gán nhãn; fallback rule-based chỉ dùng cho MVP.

## Bản v3 - Dashboard biểu đồ so sánh OCR

Bản v3 bổ sung giao diện hiển thị kết quả kỹ hơn sau khi chạy benchmark:

- Thẻ tổng quan: tổng số lượt OCR, số engine chạy thành công, số engine lỗi/thiếu cấu hình, trạng thái có/không có ground truth.
- Tự chọn engine tốt nhất theo CER, WER, thời gian xử lý và điểm chất lượng tổng hợp.
- Biểu đồ CER và WER để so sánh độ chính xác khi có ground truth text.
- Biểu đồ quality score, thời gian xử lý, độ dài text OCR và confidence trung bình.
- Bảng đánh giá tác động của tiền xử lý OpenCV: raw vs opencv_preprocessed.
- Ma trận trạng thái engine để biết engine nào chạy được, engine nào thiếu package/API key.
- Vẫn giữ bảng chi tiết OCR output và phần hậu xử lý LayoutLMv3/fallback rule để trích xuất trường.
- Xuất thêm `comparison_summary.json` bên cạnh `benchmark_results.csv` và `report.json`.

Cách dùng phần biểu đồ:

1. Upload PDF/ảnh công văn.
2. Tick các OCR engine cần so sánh.
3. Giữ tick `So sánh cả ảnh gốc và ảnh đã tiền xử lý OpenCV` để xem tác động preprocessing.
4. Dán ground truth text nếu muốn có biểu đồ CER/WER và quality score.
5. Bấm `Chạy benchmark OCR`.

Nếu không có ground truth, hệ thống vẫn hiển thị được thời gian xử lý, trạng thái engine, độ dài text OCR, confidence và text preview. Tuy nhiên, CER/WER và quality score cần text chuẩn để đánh giá đúng độ chính xác.

## Bản v4 - Upload file text chuẩn thay cho nhập tay

Bản v4 đổi phần Ground Truth từ nhập tay là chính sang **upload file text chuẩn**.

- Upload file công văn PDF/ảnh scan ở ô đầu tiên.
- Upload file text chuẩn ở ô **File text chuẩn để tính CER/WER**.
- Hỗ trợ: `.txt`, `.md`, `.csv`, `.json`, `.docx`.
- Nếu upload file text chuẩn, hệ thống tự đọc nội dung file để tính CER/WER và vẽ biểu đồ.
- Ô dán text vẫn giữ trong mục nâng cao để dùng nhanh khi chưa tạo file.

Cách dùng mẫu:

1. Upload `demo_samples/sample_cong_van_scan.png`.
2. Upload `demo_samples/ground_truth_text/sample_cong_van.txt` vào ô file text chuẩn.
3. Bấm `Chạy benchmark OCR`.
4. Xem CER/WER, quality score, biểu đồ thời gian và bảng so sánh.
