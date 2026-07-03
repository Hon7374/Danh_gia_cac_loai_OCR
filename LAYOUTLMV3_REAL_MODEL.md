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

Mặc định `.env` đang trỏ về checkpoint local đã fine-tune tại `models/layoutlmv3-congvan-token-classification`. App chạy checkpoint này bằng `transformers` và chỉ cho model ghi trường khi schema nhãn khớp bộ nhãn công văn ở trên.

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

Khuyến nghị cho dữ liệu thật trong project hiện tại: chạy một lệnh end-to-end từ folder `OCR`.

```powershell
.\scripts\train_layoutlmv3_from_ocr.ps1 `
  -SourceDir OCR `
  -MaxPages 1 `
  -Epochs 8 `
  -BatchSize 1 `
  -LearningRate 0.00002 `
  -Fp16
```

Lan train da chay tren folder `OCR`:

- 70 file nguon -> 70 record LayoutLMv3, khong skip file nao.
- Chia 60 record train / 10 record eval, khop duoc 41 file `.txt` ground truth.
- Checkpoint moi da luu tai `models/layoutlmv3-congvan-token-classification`.
- Eval cuoi: `token_accuracy=0.9072`, `non_o_precision=0.7961`, `non_o_recall=0.9206`, `non_o_f1=0.8539`.

Bo build dataset tu dong dat nhan tuong xung cho thong tin can nhan biet. No gom token OCR thanh cac dong theo bbox, nhan dien vi tri header/noi dung, uu tien nhan theo bo cuc cho `so_ky_hieu`, `ngay_ban_hanh`, `loai_van_ban`, `co_quan_ban_hanh`, `trich_yeu`, `noi_nhan`, roi moi fallback sang khop chuoi ground truth an toan. Cach nay giup tranh viec lay nham cac cau trich dan trong phan than van ban lam so ky hieu, ngay ban hanh hoac loai van ban.

Script này sẽ:

- quét đệ quy PDF/ảnh trong `OCR`;
- ghép file `.txt` cùng bộ dữ liệu làm ground truth nếu tên đủ khớp;
- render trang đầu, chạy Tesseract lấy `words` + `boxes`;
- gán nhãn BIO cho các trường công văn;
- chia train/eval và fine-tune checkpoint ở `models/layoutlmv3-congvan-token-classification`;
- ghi `dataset_template/layoutlmv3_training/build_manifest.json` và `models/layoutlmv3-congvan-token-classification/training_summary.json`.

Nếu muốn train thủ công từ JSONL đã có:

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
Field source: layoutlmv3_validated+rule
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
