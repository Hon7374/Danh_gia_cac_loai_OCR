# LayoutLMv3 thật cho trích xuất trường công văn

App không còn dùng `tiny-random` làm mặc định. LayoutLMv3 chỉ được coi là nguồn trích xuất khi checkpoint token-classification có nhãn khớp schema công văn:

- `so_ky_hieu`
- `ngay_ban_hanh`
- `trich_yeu`
- `co_quan_ban_hanh`
- `noi_gui`
- `noi_nhan`
- `loai_van_ban`

Nếu model chạy được nhưng nhãn là kiểu FUNSD/invoice như `QUESTION`, `ANSWER`, `HEADER`, app vẫn hiển thị nhãn token để kiểm tra, nhưng không cho model đó ghi đè trường công văn.

Mặc định `.env` đang dùng checkpoint thật `welcomyou/layoutlmv3-vn-admin-kie`. Model này là LayoutLMv3 KIE cho tài liệu hành chính Việt Nam, phát hành dạng ONNX INT8, nên app chạy qua `onnxruntime`.

## Dữ liệu train

Tạo file JSONL, mỗi dòng là một trang:

```json
{"image":"page_001.png","words":["Số:","11/2026/TT-NHNN"],"boxes":[[80,120,120,140],[130,120,260,140]],"labels":["O","B-so_ky_hieu"]}
```

- `image`: đường dẫn ảnh trang, tương đối theo thư mục chứa JSONL hoặc tuyệt đối.
- `words`: token OCR theo thứ tự đọc.
- `boxes`: bbox `[x0,y0,x1,y1]`; mặc định là tọa độ pixel gốc. Nếu bbox đã chuẩn hóa 0-1000, thêm `--boxes-normalized` khi train.
- `labels`: `O` hoặc BIO label như `B-trich_yeu`, `I-trich_yeu`.

Có file mẫu tại `dataset_template/layoutlmv3_fields.example.jsonl`.

## Train checkpoint

```powershell
.\.venv\Scripts\python.exe scripts\train_layoutlmv3_fields.py `
  --train-jsonl dataset_template\layoutlmv3_fields.train.jsonl `
  --output-dir models\layoutlmv3-congvan-token-classification `
  --epochs 8 `
  --batch-size 1
```

Sau khi train xong, cấu hình:

```env
LAYOUTLMV3_MODEL_DIR=models/layoutlmv3-congvan-token-classification
LAYOUTLMV3_PROCESSOR_NAME=microsoft/layoutlmv3-base
```

Khởi động lại app. Khi checkpoint hợp lệ, trang kết quả sẽ báo:

```text
Mode: layoutlmv3_model
Field source: layoutlmv3+rule
Field label schema: compatible
```

## Folder dataset gom san

Tat ca du lieu train co the dat trong mot folder:

```text
dataset_template/layoutlmv3_training/
```

Trong do:

- `pages/`: bo anh tung trang vao day.
- `train.jsonl`: du lieu train chinh.
- `eval.jsonl`: du lieu validation neu co, co the de trong.
- `label_schema.json`: schema field/label hop le.
- `train.ps1`: chay train truc tiep tu folder nay.

Lenh train nhanh:

```powershell
.\dataset_template\layoutlmv3_training\train.ps1
```
