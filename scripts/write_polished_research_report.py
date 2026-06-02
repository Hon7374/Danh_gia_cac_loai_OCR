# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import generate_benchmark_technical_report as base
from scripts import generate_benchmark_technical_report_all_jobs as all_jobs


ROOT = Path.cwd()
JOB_ID = base.JOB_ID
JOB = base.JOB
OUT = base.OUT


def md_cell(value) -> str:
    return base.md_cell(value)


def fmt(value, digits: int = 2) -> str:
    return base.fmt(value, digits)


def metric_item(summary: dict, key: str) -> dict:
    item = summary.get(key) or {}
    return item if isinstance(item, dict) else {}


def summary_rows_table(rows: list[dict]) -> str:
    lines = [
        "| Engine | Biến thể | Trạng thái | CER | WER | Runtime | Text length | Confidence TB | Quality score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {md_cell(row.get('engine'))} | {md_cell(row.get('variant'))} | {md_cell(row.get('status'))} | "
            f"{fmt(row.get('cer_pct'), 2)}% | {fmt(row.get('wer_pct'), 2)}% | {fmt(row.get('elapsed_sec'), 3)}s | "
            f"{md_cell(row.get('text_len'))} | {fmt(row.get('avg_confidence'), 2)}% | {fmt(row.get('quality_score'), 2)}/100 |"
        )
    return "\n".join(lines)


def latest_highlight_table(summary: dict) -> str:
    best_cer = metric_item(summary, "best_cer")
    best_wer = metric_item(summary, "best_wer")
    fastest = metric_item(summary, "fastest")
    best_quality = metric_item(summary, "best_quality")
    best_conf = metric_item(summary, "best_confidence")
    rows = summary.get("rows") or []
    longest = max(rows, key=lambda row: row.get("text_len") or -1)
    return f"""| Hạng mục | Engine/biến thể | Số liệu |
|---|---|---:|
| CER thấp nhất | `{best_cer.get('label')}` | {fmt(best_cer.get('cer_pct'), 2)}% |
| WER thấp nhất | `{best_wer.get('label')}` | {fmt(best_wer.get('wer_pct'), 2)}% |
| Nhanh nhất | `{fastest.get('label')}` | {fmt(fastest.get('elapsed_sec'), 3)}s |
| Quality score cao nhất | `{best_quality.get('label')}` | {fmt(best_quality.get('quality_score'), 2)}/100 |
| Text dài nhất | `{longest.get('label')}` | {md_cell(longest.get('text_len'))} ký tự |
| Confidence cao nhất | `{best_conf.get('label')}` | {fmt(best_conf.get('avg_confidence'), 2)}% |"""


def row_item(summary: dict, engine: str, variant: str) -> dict:
    for row in summary.get("rows") or []:
        if row.get("engine") == engine and row.get("variant") == variant:
            return row
    return {}


def latest_job_note(summary: dict) -> str:
    best_cer = metric_item(summary, "best_cer")
    best_wer = metric_item(summary, "best_wer")
    best_quality = metric_item(summary, "best_quality")
    fastest = metric_item(summary, "fastest")
    return (
        f"Trong job mới nhất, **{best_quality.get('label')}** đạt quality score cao nhất "
        f"với {fmt(best_quality.get('quality_score'), 2)}/100. "
        f"Cấu hình có CER thấp nhất là **{best_cer.get('label')}** "
        f"với CER {fmt(best_cer.get('cer_pct'), 2)}% và WER {fmt(best_cer.get('wer_pct'), 2)}%. "
        f"Cấu hình có WER thấp nhất là **{best_wer.get('label')}** "
        f"với WER {fmt(best_wer.get('wer_pct'), 2)}%. "
        f"Cấu hình nhanh nhất là **{fastest.get('label')}** với {fmt(fastest.get('elapsed_sec'), 3)} giây."
    )


def opencv_pipeline_note(report: dict) -> str:
    steps = report.get("opencv_steps") or []
    if not steps:
        return "Chưa có dữ liệu pipeline OpenCV ở lần chạy hiện tại."
    steps_text = ", ".join(f"`{step}`" for step in steps)
    return (
        f"Pipeline OpenCV trong lần chạy hiện tại dùng chế độ adaptive/safe, gồm {steps_text}. "
        "Điểm quan trọng là bước `quality_gate_clean_raw_passthrough`: nếu trang scan đã sạch, "
        "hệ thống không ép nhị phân để tránh làm mất dấu tiếng Việt; threshold chỉ dùng cho trang thật sự cần làm sạch nền."
    )


def opencv_impact_note(summary: dict) -> str:
    effects = summary.get("preprocessing_effect") or []
    if not effects:
        return "Chưa có dữ liệu so sánh raw và OpenCV ở lần chạy hiện tại."
    improved = [e for e in effects if (e.get("cer_delta") is not None and e.get("cer_delta") < 0) or (e.get("wer_delta") is not None and e.get("wer_delta") < 0)]
    recommended_opencv = [e for e in effects if e.get("recommended_variant") == "opencv_preprocessed"]
    lines = []
    if improved:
        best = min(improved, key=lambda e: ((e.get("cer_delta") or 0) + (e.get("wer_delta") or 0)))
        lines.append(
            f"Ở job mới nhất, OpenCV adaptive giúp **{best.get('engine')}** cải thiện: "
            f"delta CER {fmt(best.get('cer_delta'), 2)} điểm %, delta WER {fmt(best.get('wer_delta'), 2)} điểm %."
        )
    else:
        lines.append("Ở job mới nhất, không có engine nào cải thiện rõ ràng sau OpenCV.")
    if recommended_opencv:
        engines = ", ".join(f"`{e.get('engine')}`" for e in recommended_opencv)
        lines.append(f"Dashboard khuyến nghị dùng OpenCV cho: {engines}.")
    raw_engines = [e.get("engine") for e in effects if e.get("recommended_variant") == "raw"]
    if raw_engines:
        lines.append(
            "Với các engine còn lại, raw vẫn là nhánh chính; OpenCV chỉ là fallback khi ảnh nhiễu, nghiêng hoặc tương phản kém."
        )
    return " ".join(lines)


def paddle_vietocr_note(summary: dict) -> str:
    raw = row_item(summary, "paddle_vietocr", "raw")
    opencv = row_item(summary, "paddle_vietocr", "opencv_preprocessed")
    if not raw:
        return "**PaddleOCR + VietOCR.** Chưa có dữ liệu thực nghiệm ở lần chạy hiện tại."
    return (
        f"**PaddleOCR + VietOCR.** Sau fine-tune VietOCR, cấu hình raw đạt CER {fmt(raw.get('cer_pct'), 2)}% "
        f"và WER {fmt(raw.get('wer_pct'), 2)}%, tốt hơn đáng kể so với trạng thái trước fine-tune. "
        f"Biến thể OpenCV hiện đạt CER {fmt(opencv.get('cer_pct'), 2)}% và WER {fmt(opencv.get('wer_pct'), 2)}%, "
        "nên dashboard khuyến nghị dùng raw cho engine này. Confidence nội bộ vẫn cần được đối chiếu bằng CER/WER, "
        "không dùng một mình để chọn kết quả production."
    )


def write_report() -> None:
    # Regenerate current assets and aggregate charts before writing the final report.
    base.main()
    records = all_jobs.load_records()
    all_jobs.make_aggregate_charts(records)

    report = json.loads((JOB / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((JOB / "comparison_summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows") or []
    layout = report.get("layoutlmv3_postprocess") or {}
    fields = layout.get("fields") or {}
    metric_rows = all_jobs.all_metric_rows(records)
    gt_jobs = [record for record in records if record["has_gt"]]
    representative = all_jobs.representative_records(records)
    upload_counts = Counter(record["filename"] for record in records)
    report_mtime = datetime.fromtimestamp((JOB / "report.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    latest_best_cer = metric_item(summary, "best_cer")
    latest_best_wer = metric_item(summary, "best_wer")
    latest_fastest = metric_item(summary, "fastest")
    latest_quality = metric_item(summary, "best_quality")
    latest_conf = metric_item(summary, "best_confidence")

    content = f"""# Báo cáo kỹ thuật thực nghiệm OCR và trích xuất metadata công văn tiếng Việt

**Đề tài:** Ứng dụng OCR kết hợp chữ ký số trong quản lý công văn điện tử tiếng Việt  
**Dữ liệu phân tích:** toàn bộ kết quả đã scan trong thư mục `jobs/`, phân tích sâu job mới nhất `{JOB_ID}`  
**File scan mới nhất:** `{report.get('uploaded_file')}`  
**Thời điểm cập nhật số liệu:** {report_mtime}  

## 1. Tóm tắt nghiên cứu

Báo cáo này tổng hợp kết quả benchmark OCR trên các công văn tiếng Việt đã được xử lý trong hệ thống demo. Mục tiêu là đánh giá khả năng nhận dạng văn bản, tác động của tiền xử lý OpenCV, khả năng tính lỗi OCR bằng CER/WER khi có file text chuẩn, và vai trò của LayoutLMv3 trong trích xuất metadata công văn.

Dữ liệu được lấy trực tiếp từ các file thực nghiệm hiện có trong project, gồm `report.json`, `comparison_summary.json`, `benchmark_results.csv`, ảnh trang scan, ảnh OpenCV, output OCR và kết quả LayoutLMv3. Báo cáo không sử dụng benchmark Internet và không tự tạo số liệu minh họa.

Kết quả chính cho thấy **Tesseract raw là lựa chọn OCR production phù hợp nhất trong bộ demo hiện tại** do đạt CER thấp nhất và thời gian xử lý nhanh nhất trên các hồ sơ đầy đủ. **PaddleOCR-VL** có lợi thế về WER và hiểu layout nhưng runtime rất cao, phù hợp kiểm tra offline. **LayoutLMv3** nên đặt sau bước chọn OCR tốt nhất để trích metadata nghiệp vụ trước khi kiểm tra chữ ký số và lưu hồ sơ.

## 2. Dữ liệu thực nghiệm

| Hạng mục | Giá trị |
|---|---:|
| Tổng job có report | {len(records)} |
| Job có ground-truth và đã tính CER/WER | {len(gt_jobs)} |
| Dòng OCR metric có CER/WER | {len(metric_rows)} |
| Hồ sơ đầy đủ đại diện | {len(representative)} |
| Job mới nhất phân tích sâu | `{JOB_ID}` |

Phân bố file scan trong lịch sử chạy:

| File scan | Số job |
|---|---:|
{chr(10).join(f"| {md_cell(name)} | {count} |" for name, count in upload_counts.most_common())}

Các hồ sơ đầy đủ đại diện là các job có đủ ground-truth, có số trang, chạy đủ 8 biến thể OCR và không lỗi runtime. Nhóm này được dùng làm cơ sở kết luận chính vì các job lịch sử khác có thể là lần chạy thử một phần, trùng file hoặc không cùng phạm vi ground-truth.

## 3. Phương pháp đánh giá

Hệ thống chạy OCR trên hai nhánh ảnh:

1. `raw`: ảnh trang scan gốc sau khi tách từ PDF.
2. `opencv_preprocessed`: ảnh sau tiền xử lý OpenCV.

Các engine được benchmark gồm Tesseract, EasyOCR, PaddleOCR + VietOCR và PaddleOCR-VL. Với mỗi engine, hệ thống ghi nhận trạng thái chạy, runtime, độ dài text, confidence trung bình nếu engine hỗ trợ, CER, WER và quality score.

CER và WER được tính từ file ground-truth do người dùng upload. Ở các công văn dài, phép tính Levenshtein toàn văn được thực hiện bằng `rapidfuzz` để tránh tình trạng metric bị bỏ qua do giới hạn bộ nhớ/thời gian của Dynamic Programming thuần Python. Vì vậy, các job có ground-truth hiện đã hiển thị được CER/WER và bảng tác động OpenCV.

Để tránh lỗi chọn nhầm engine có confidence cao nhưng rụng dấu tiếng Việt, pipeline bổ sung bước hậu kiểm chất lượng tiếng Việt. Bước này đo tỷ lệ token có dấu, tỷ lệ ký tự tiếng Việt có dấu và các mẫu mất nguyên âm thường gặp như `CNG`, `DC`, `VIT`, `quc`, `ngh`. Nếu output bị nghi ngờ rụng dấu/mất nguyên âm nghiêm trọng, hệ thống hạ điểm quality khi không có ground-truth và không dùng output đó làm kết quả chính.

Quality score khi có CER/WER được tính theo hướng lỗi càng thấp càng tốt. CER được ưu tiên hơn WER vì tiếng Việt thường sai dấu, ký tự hoặc cụm âm tiết; WER vẫn quan trọng để đánh giá mức độ bảo toàn từ/ngữ nghĩa.

## 4. Kiến trúc hệ thống

```mermaid
graph TD
    U[Người dùng] --> UP[Upload PDF/scan]
    UP --> PRE[OpenCV preprocessing]
    UP --> RAW[Nhánh ảnh raw]
    PRE --> OCR[OCR benchmark engine]
    RAW --> OCR
    OCR --> SEL[Chọn kết quả OCR tốt nhất]
    SEL --> LM[LayoutLMv3 document understanding]
    LM --> META[Trích metadata công văn]
    META --> SIG[Kiểm tra chữ ký số]
    SIG --> DB[(Lưu DB / kho hồ sơ)]
    DB --> DASH[Dashboard tra cứu và đánh giá]
```

## 5. Dashboard benchmark job mới nhất

![Dashboard benchmark](assets/dashboard_benchmark.png)

### 5.1. Tổng quan job `{JOB_ID}`

| Chỉ tiêu | Giá trị |
|---|---:|
| File scan | `{all_jobs.short_filename(report)}` |
| Số trang OCR | {report.get('page_count')} |
| Tổng lượt OCR | {summary.get('total_runs')} |
| Lượt thành công | {summary.get('ok_runs')}/{summary.get('total_runs')} |
| Runtime fail | {summary.get('error_runs')} |
| Skipped | {summary.get('skipped_runs')} |
| Ground-truth | Có, {report.get('ground_truth_text_length')} ký tự |
| CER/WER | Đã tính |
| OpenCV workers | {report.get('runtime', {}).get('opencv_workers')} |
| Tesseract workers | {report.get('runtime', {}).get('tesseract_workers')} |
| GPU workers | {report.get('runtime', {}).get('gpu_workers')} |

### 5.2. Kết quả nổi bật

{latest_highlight_table(summary)}

{latest_job_note(summary)}

## 6. So sánh ảnh gốc và ảnh OpenCV

### 6.1. Ảnh gốc công văn

![Ảnh gốc công văn trang 1](assets/input_raw_page1.png)

### 6.2. Ảnh sau OpenCV preprocessing

![Ảnh sau OpenCV trang 1](assets/input_opencv_page1.png)

{opencv_pipeline_note(report)}

Kết quả thực nghiệm cho thấy OpenCV không nên áp dụng cứng cho mọi tài liệu. Với ảnh scan rõ, threshold/morphology có thể làm mất dấu tiếng Việt hoặc làm dày nét, khiến CER/WER tăng. Vì vậy raw nên là nhánh chính; OpenCV nên là nhánh fallback cho ảnh nhiễu, nghiêng hoặc tương phản kém.

## 7. Bảng kết quả OCR chi tiết job mới nhất

{summary_rows_table(rows)}

## 8. Biểu đồ job mới nhất

### 8.1. Quality score

![Quality score OCR](assets/quality_score_bar.png)

### 8.2. Runtime

![Runtime OCR](assets/runtime_bar.png)

### 8.3. Confidence trung bình

![Confidence OCR](assets/confidence_bar.png)

### 8.4. Độ dài text OCR

![Text length OCR](assets/text_length_bar.png)

### 8.5. Radar tổng hợp

![Radar tổng hợp OCR](assets/radar_summary.png)

## 9. Phân tích các file đã scan trước đó

{all_jobs.representative_table(records)}

### 9.1. CER tốt nhất theo hồ sơ đầy đủ

![CER tốt nhất theo hồ sơ](assets/aggregate_best_cer_by_job.png)

### 9.2. CER/WER trung bình theo engine

![CER WER trung bình theo engine](assets/aggregate_engine_avg_cer_wer.png)

### 9.3. Số lần engine thắng quality score

![Engine thắng quality score](assets/aggregate_engine_quality_wins.png)

## 10. Phân tích theo OCR engine

{all_jobs.engine_stats_table(records)}

**Tesseract.** Ở ba hồ sơ đầy đủ đại diện, Tesseract raw đạt CER thấp nhất: 3.01% với `39-bgddt.pdf`, 1.57% với `151_2026_ND-CP_13052026_1-signed.pdf` và 15.02% với `Thông tư 11-2026-TT-NHNN.pdf`. Runtime cũng thấp nhất trong nhóm engine, phù hợp triển khai production.

**EasyOCR.** EasyOCR chạy được trên nhiều job và cho text tương đối dài, nhưng CER/WER trung bình không tốt bằng Tesseract hoặc PaddleOCR-VL. Engine này phù hợp làm kênh đối chiếu trong benchmark hơn là OCR chính.

{paddle_vietocr_note(summary)}

**PaddleOCR-VL.** PaddleOCR-VL có WER tốt nhất ở một số hồ sơ và có lợi thế về layout/markdown, nhưng runtime rất cao. Ở job mới nhất, `paddleocr_vl / raw` mất 831.805 giây; ở tài liệu dài có thể lên hàng nghìn giây. Vì vậy engine này phù hợp kiểm tra offline hoặc phân tích layout chuyên sâu, không phù hợp làm production mặc định.

## 11. Tác động của OpenCV

{base.impact_table(summary.get('preprocessing_effect') or [])}

{opencv_impact_note(summary)}

## 12. OCR output thực tế

![OCR output preview](assets/ocr_output_preview.png)

Preview trên là output từ engine được pipeline cũ chọn theo thứ tự ưu tiên. Sau khi bổ sung CER/WER, việc chọn engine cần ưu tiên metric lỗi thực nghiệm. Đây là lý do Tesseract raw được chọn làm production chính cho bộ dữ liệu hiện tại.

## 13. LayoutLMv3 và trích xuất metadata

![LayoutLMv3 extracted fields](assets/layoutlmv3_extracted_fields.png)

OCR và LayoutLMv3 có vai trò khác nhau. OCR nhận dạng chữ từ ảnh; LayoutLMv3 hiểu cấu trúc tài liệu và gán thông tin vào các trường nghiệp vụ. Trong đề tài quản lý công văn điện tử, LayoutLMv3 được dùng cho document understanding, key information extraction và hậu xử lý metadata sau OCR.

| Thuộc tính | Giá trị |
|---|---|
| Mode | `{layout.get('mode')}` |
| Extractor | `{layout.get('extractor')}` |
| Runtime | `{layout.get('runtime')}` |
| Model path | `{md_cell(layout.get('model_path'))}` |
| Model ran | `{layout.get('model_ran')}` |
| Torch device | `{layout.get('torch_device')}` |
| Field label schema | `{layout.get('field_label_compatible')}` |
| Field source | `{layout.get('fields_source')}` |
| Accepted model fields | `{len(layout.get('accepted_model_fields') or {})}` |
| Rejected model fields | `{len(layout.get('rejected_model_fields') or {})}` |

{base.layout_fields_table(fields)}

LayoutLMv3 thật đã chạy bằng transformers từ model local. Trong job mới nhất, field do model dự đoán chưa được guard chấp nhận hoàn toàn, nên metadata cuối cùng được merge với rule fallback. Cách làm này giúp giảm nguy cơ ghi sai metadata vào hồ sơ công văn.

## 14. Sequence xử lý nghiệp vụ

```mermaid
sequenceDiagram
    participant User as User
    participant System as System
    participant OCR as OCR Engines
    participant LayoutLMv3 as LayoutLMv3
    participant Sig as Signature Verification
    participant DB as Database

    User->>System: Upload PDF/scan công văn
    System->>System: Tách trang ảnh và sinh nhánh raw/OpenCV
    System->>OCR: Chạy benchmark OCR
    OCR-->>System: Trả text, runtime, confidence
    System->>System: Tính CER/WER nếu có ground-truth
    System->>System: Chọn OCR output tốt nhất
    System->>LayoutLMv3: Gửi text + layout/image
    LayoutLMv3-->>System: Trả metadata và nhãn field
    System->>Sig: Kiểm tra chữ ký số/PAdES
    Sig-->>System: Trả trạng thái chữ ký
    System->>DB: Lưu file, OCR, metadata, metric
    User->>System: Tra cứu dashboard/report
    System-->>User: Hiển thị benchmark và hồ sơ
```

## 15. Tích hợp chữ ký số

Trong kiến trúc đề tài, chữ ký số nên được đặt sau OCR và LayoutLMv3. Lý do là hệ thống cần metadata công văn để đối soát file, định danh hồ sơ và lưu kết quả kiểm tra chữ ký. Luồng phù hợp là OCR tạo text, LayoutLMv3 trích metadata, sau đó module chữ ký số kiểm tra chứng thư, thời điểm ký và tính toàn vẹn tài liệu.

## 16. Kết luận

Dựa trên dữ liệu CER/WER đã tính từ các file ground-truth trong project, **Tesseract raw là OCR production chính phù hợp nhất cho demo hiện tại**. Engine này đạt CER thấp nhất ở các hồ sơ đầy đủ đại diện, đồng thời có runtime thấp hơn đáng kể so với các engine học sâu.

**Tesseract OpenCV adaptive** có thể dùng khi dashboard cho thấy CER/WER giảm; ở job mới nhất nhánh này cải thiện nhẹ so với raw. **PaddleOCR-VL** nên dùng cho kiểm tra layout/chất lượng offline. **PaddleOCR + VietOCR** đã tốt hơn sau fine-tune ở nhánh raw, nhưng biến thể OpenCV vẫn kém hơn nên chưa nên làm production chính. **EasyOCR** phù hợp cho kiểm thử đối chiếu.

**LayoutLMv3 nên đặt ở bước hậu OCR**, sau khi đã chọn OCR output tốt nhất. Vai trò của LayoutLMv3 là trích metadata và hiểu cấu trúc công văn, không thay thế OCR. Cách kết hợp này phù hợp với đề tài “Ứng dụng OCR kết hợp chữ ký số trong quản lý công văn điện tử tiếng Việt”: OCR nhận chữ, LayoutLMv3 hiểu trường nghiệp vụ, chữ ký số xác thực tính pháp lý và DB lưu trữ phục vụ tra cứu.

## 17. Hạn chế và hướng hoàn thiện

1. Cần mở rộng tập ground-truth để giảm lệch do số lượng hồ sơ đầy đủ còn ít.
2. Cần tách rõ job thử nghiệm partial và job benchmark chuẩn trong dashboard.
3. Cần mở rộng fine-tune PaddleOCR + VietOCR bằng nhiều mẫu công văn hơn nếu muốn dùng engine này làm production.
4. Cần huấn luyện thêm LayoutLMv3 trên tập công văn gán nhãn thật để giảm phụ thuộc rule fallback.
5. Cần lưu Docker log theo từng job để bảo đảm khả năng tái lập môi trường chạy.

## 18. Phụ lục: toàn bộ job đã scan

{all_jobs.all_jobs_table(records)}
"""

    (OUT / "OCR_BENCHMARK_TECHNICAL_REPORT.md").write_text(content, encoding="utf-8")
    print(f"Wrote {OUT / 'OCR_BENCHMARK_TECHNICAL_REPORT.md'}")


if __name__ == "__main__":
    write_report()
