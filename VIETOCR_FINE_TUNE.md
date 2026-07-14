# Fine-tune VietOCR cho công văn tiếng Việt

## Mục tiêu

Nhánh này dùng PaddleOCR để phát hiện vùng chữ, sau đó dùng VietOCR đã fine-tune trên crop dòng công văn để nhận dạng lại text tiếng Việt. Mục tiêu là giảm lỗi rụng dấu, mất nguyên âm và lỗi từ trong engine `paddle_vietocr`.

## Dữ liệu cũ (không còn dùng để promote model)

- Dataset: `dataset_template/vietocr_finetune`
- Số crop dòng chữ: `2043`
- Train: `1737`
- Validation: `306`
- Nhãn align từ ground-truth `.doc/.docx`: `1722`
- Pseudo-label OCR chất lượng cao: `321`

Dataset cũ bị trộn dòng của cùng tài liệu vào cả train/validation, chứa pseudo-label và mất cân bằng theo tài liệu. Vì vậy số validation cũ không đủ tin cậy để promote checkpoint.

## Checkpoint fine-tune cũ (đã loại khỏi mặc định)

- Model dir: `models/vietocr-congvan`
- Config: `models/vietocr-congvan/config.yml`
- Weights: `models/vietocr-congvan/transformerocr.pth`
- Thiết bị train: `cuda:0`
- Iterations đã chạy: `120`
- Validation full-sequence accuracy: `0.8088`
- Validation per-character accuracy: `0.9119`
- Eval 128 dòng validation:
  - CER: `2.43%`
  - WER: `3.70%`

Model gate ngày 2026-07-12 cho thấy checkpoint này thua checkpoint VietOCR chính thức:

- Validation sạch theo tài liệu: CER `1,71%` so với `1,01%`.
- Tài liệu lỗi được khóa khỏi train: CER `15,49%` so với `11,43%`; WER `27,91%` so với `17,94%`.

Vì vậy demo dùng `models/vietocr-pretrained/vgg_transformer.pth`. Checkpoint cũ chỉ được giữ lại để tái lập kết quả, không được tự động promote.

## Dataset v2 không leakage

- Dataset: `dataset_template/vietocr_finetune_v2`
- 1.475 crop có nhãn ground-truth, không pseudo-label.
- 6 tài liệu nguồn; chia train/validation theo SHA-256 tài liệu.
- Không overlap job, tài liệu, ảnh hoặc content hash giữa hai split.
- Tài liệu `695292828457` được khóa hoàn toàn khỏi train để làm test độc lập.

## Build lại dataset

```powershell
.\.venv\Scripts\python.exe scripts\build_vietocr_finetune_dataset.py --output-dir dataset_template\vietocr_finetune_v2 --test-job-id 695292828457 --max-samples 3000 --max-samples-per-document 600 --overwrite
```

## Train candidate mới

```powershell
.\.venv\Scripts\python.exe scripts\train_vietocr_finetune.py --dataset-dir dataset_template\vietocr_finetune_v2 --output-dir models\vietocr-congvan-candidate --init-weights models\vietocr-pretrained\vgg_transformer.pth --iters 3000 --batch-size 4 --image-max-width 1024 --max-lr 0.00005 --image-aug --print-every 20 --valid-every 250 --eval-samples 226 --rebuild-lmdb
```

## Cấu hình app dùng model

Trong `.env`:

```env
PADDLE_VIETOCR_REFINE=1
VIETOCR_MODEL_PROFILE=pretrained
VIETOCR_CONFIG_PATH=models\vietocr-pretrained\config.yml
VIETOCR_WEIGHTS_PATH=models\vietocr-pretrained\vgg_transformer.pth
VIETOCR_BATCH_SIZE=48
```

Chỉ đổi sang candidate sau khi candidate thắng cả validation sạch và test khóa:

```env
VIETOCR_MODEL_PROFILE=finetuned
VIETOCR_CONFIG_PATH=models\vietocr-congvan-candidate\config.yml
VIETOCR_WEIGHTS_PATH=models\vietocr-congvan-candidate\transformerocr.pth
```

Ngoài model gate, runtime còn áp dụng hybrid guard theo từng crop. Output VietOCR chạm trần decoder, lặp bệnh lý, tăng/co độ dài bất thường hoặc rỗng sẽ tự động dùng lại text Paddle. Dòng có aspect ratio từ 18 trở lên được tách tại khoảng trắng trước khi nhận dạng; recursive XY-cut sắp lại header/footer nhiều cột chỉ bằng bbox.

Sau khi sửa `.env`, restart server để app nhận model mới.

## Lưu ý kỹ thuật

Trên Windows, PaddleOCR và Torch có thể xung đột CUDA/cuDNN nếu nạp chung trong một process. Vì vậy project chạy VietOCR refinement trong subprocess riêng (`app/ocr_engines/vietocr_refine_worker.py`). Cách này chậm hơn một chút nhưng ổn định hơn khi bật GPU.
