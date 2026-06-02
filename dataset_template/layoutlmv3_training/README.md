# Dataset train LayoutLMv3 cong van

Day la mot folder duy nhat de nap du lieu train LayoutLMv3.

## Cau truc

```text
dataset_template/layoutlmv3_training/
+-- README.md
+-- label_schema.json
+-- example.jsonl
+-- train.jsonl
+-- eval.jsonl
+-- build_dataset.ps1
+-- convert_word_to_pdf.ps1
+-- train.ps1
+-- pages/
    +-- .gitkeep
```

## Cach nap du lieu

1. Bo anh tung trang vao `pages/`, vi du:

```text
pages/page_001.png
pages/page_002.png
```

2. Dien du lieu vao `train.jsonl`. Moi dong la 1 trang. Xem sample trong `example.jsonl`:

```json
{"image":"pages/page_001.png","words":["Số:","11/2026/TT-NHNN","THÔNG","TƯ","Phân","cấp"],"boxes":[[80,120,120,140],[130,120,260,140],[250,180,320,205],[325,180,360,205],[120,225,165,245],[170,225,205,245]],"labels":["O","B-so_ky_hieu","B-loai_van_ban","I-loai_van_ban","B-trich_yeu","I-trich_yeu"]}
```

3. Neu co tap validation rieng, dien vao `eval.jsonl`. Neu chua co thi de trong, script se tu chia mot phan train lam eval khi du so mau.

Neu da bo PDF/anh vao `pages/` nhung chua co JSONL, chay lenh nay de build `train.jsonl` tu dong bang OCR + rule label:

```powershell
.\dataset_template\layoutlmv3_training\build_dataset.ps1
```

DOC/DOCX khong co bbox anh. De dung chung, convert sang PDF truoc:

```powershell
.\dataset_template\layoutlmv3_training\convert_word_to_pdf.ps1
```

## Nhan hop le

```text
O
B-so_ky_hieu / I-so_ky_hieu
B-ngay_ban_hanh / I-ngay_ban_hanh
B-trich_yeu / I-trich_yeu
B-co_quan_ban_hanh / I-co_quan_ban_hanh
B-noi_gui / I-noi_gui
B-noi_nhan / I-noi_nhan
B-loai_van_ban / I-loai_van_ban
```

## Box

`boxes` la bbox cua tung token theo dang:

```text
[x0, y0, x1, y1]
```

Mac dinh script coi box la toa do pixel tren anh goc. Neu box da chuan hoa 0-1000, mo `train.ps1` va them tham so `--boxes-normalized`.

## Chay train

Tu PowerShell:

```powershell
cd D:\ocr_workspace\ocr_full_demo_v2_latest
.\dataset_template\layoutlmv3_training\train.ps1
```

Output model se nam o:

```text
models/layoutlmv3-congvan-token-classification
```

Sau khi train xong, cau hinh `.env`:

```env
LAYOUTLMV3_MODEL_DIR=models/layoutlmv3-congvan-token-classification
LAYOUTLMV3_MODEL_NAME=
LAYOUTLMV3_PROCESSOR_NAME=microsoft/layoutlmv3-base
```
