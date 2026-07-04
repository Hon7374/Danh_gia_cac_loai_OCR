# Fine-tune VietOCR cho công văn tiếng Việt

## Mục tiêu

Nhánh này dùng PaddleOCR để phát hiện vùng chữ, sau đó dùng VietOCR đã fine-tune trên crop dòng công văn để nhận dạng lại text tiếng Việt. Mục tiêu là giảm lỗi rụng dấu, mất nguyên âm và lỗi từ trong engine `paddle_vietocr`.

## Dữ liệu đã tạo

- Dataset: `dataset_template/vietocr_finetune`
- Số crop dòng chữ: `2043`
- Train: `1737`
- Validation: `306`
- Nhãn align từ ground-truth `.doc/.docx`: `1722`
- Pseudo-label OCR chất lượng cao: `321`

Dataset được sinh từ các job benchmark hiện có trong `jobs/`, ưu tiên `tesseract / raw` vì đây là OCR có CER/WER tốt nhất trong dữ liệu demo hiện tại.

## Model đã train

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

## Build lại dataset

```powershell
.\.venv\Scripts\python.exe scripts\build_vietocr_finetune_dataset.py --overwrite --max-samples 3000
```

## Train tiếp

```powershell
.\.venv\Scripts\python.exe scripts\train_vietocr_finetune.py --iters 1000 --batch-size 8 --print-every 20 --valid-every 200 --eval-samples 128 --resume-existing --rebuild-lmdb
```

## Cấu hình app dùng model

Trong `.env`:

```env
PADDLE_VIETOCR_REFINE=1
VIETOCR_MODEL_PROFILE=finetuned
VIETOCR_CONFIG_PATH=D:\ocr_workspace\ocr_full_demo_v2_latest\models\vietocr-congvan\config.yml
VIETOCR_WEIGHTS_PATH=D:\ocr_workspace\ocr_full_demo_v2_latest\models\vietocr-congvan\transformerocr.pth
VIETOCR_BATCH_SIZE=48
```

Nếu muốn dùng VietOCR pretrained chính chủ thay vì model fine-tune nội bộ, đặt:

```env
VIETOCR_MODEL_PROFILE=pretrained
```

Trong smoke test `demo_samples/sample_cong_van_scan.png`, profile `pretrained` đang cho WER tốt hơn model fine-tune ngắn hiện tại, nên demo local mặc định dùng `pretrained`.

Sau khi sửa `.env`, restart server để app nhận model mới.

## Lưu ý kỹ thuật

Trên Windows, PaddleOCR và Torch có thể xung đột CUDA/cuDNN nếu nạp chung trong một process. Vì vậy project chạy VietOCR refinement trong subprocess riêng (`app/ocr_engines/vietocr_refine_worker.py`). Cách này chậm hơn một chút nhưng ổn định hơn khi bật GPU.
