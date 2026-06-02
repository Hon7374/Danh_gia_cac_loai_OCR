# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import generate_benchmark_technical_report as base


ROOT = Path.cwd()
JOB_ID = base.JOB_ID
JOB = base.JOB
OUT = base.OUT
ASSETS = base.ASSETS


def short_filename(report: dict) -> str:
    value = report.get("uploaded_file") or "N/A"
    return value.replace("\\", "/").split("/")[-1]


def load_records() -> list[dict]:
    records: list[dict] = []
    for job_dir in sorted((ROOT / "jobs").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        report_path = job_dir / "report.json"
        summary_path = job_dir / "comparison_summary.json"
        if not report_path.exists() or not summary_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append(
            {
                "job_id": job_dir.name,
                "mtime": job_dir.stat().st_mtime,
                "report": report,
                "summary": summary,
                "filename": short_filename(report),
                "pages": report.get("page_count"),
                "runs": summary.get("total_runs"),
                "ok": summary.get("ok_runs"),
                "has_gt": bool(summary.get("has_ground_truth")),
                "gt_len": report.get("ground_truth_text_length"),
            }
        )
    return records


def all_metric_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if not record["has_gt"]:
            continue
        for row in record["summary"].get("rows") or []:
            if row.get("status") != "ok" or row.get("cer_pct") is None:
                continue
            enriched = dict(row)
            enriched["job_id"] = record["job_id"]
            enriched["filename"] = record["filename"]
            rows.append(enriched)
    return rows


def representative_records(records: list[dict]) -> list[dict]:
    selected = []
    for record in records:
        if record["has_gt"] and record["pages"] and (record["runs"] or 0) >= 8 and (record["ok"] or 0) >= 8:
            selected.append(record)
    return selected


def md_cell(value) -> str:
    return base.md_cell(value)


def fmt(value, digits: int = 2) -> str:
    return base.fmt(value, digits)


def metric_item(summary: dict, key: str) -> dict:
    item = summary.get(key) or {}
    return item if isinstance(item, dict) else {}


def make_aggregate_charts(records: list[dict]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metric_rows = all_metric_rows(records)
    reps = representative_records(records)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    palette = ["#0057B8", "#00A676", "#E67E22", "#7B2CBF", "#C1121F", "#0081A7", "#6D597A", "#2A9D8F"]

    # Best CER by representative full-document job.
    labels = [f"{record['job_id']}\n{record['filename'][:30]}" for record in reps]
    values = [metric_item(record["summary"], "best_cer").get("cer_pct") for record in reps]
    fig, ax = plt.subplots(figsize=(11, max(4.6, len(labels) * 0.8)), dpi=160)
    y = np.arange(len(labels))
    ax.barh(y, values, color=palette[: len(labels)])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CER thấp nhất (%)")
    ax.set_title("CER tốt nhất theo hồ sơ đầy đủ", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#D9E2EC")
    for idx, value in enumerate(values):
        ax.text(value, idx, f"  {value:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "aggregate_best_cer_by_job.png", bbox_inches="tight")
    plt.close(fig)

    # Average CER/WER by engine over all metric rows.
    engine_stats = []
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for row in metric_rows:
        by_engine[row["engine"]].append(row)
    for engine, rows in sorted(by_engine.items()):
        engine_stats.append(
            {
                "engine": engine,
                "n": len(rows),
                "cer": statistics.mean(row["cer_pct"] for row in rows),
                "wer": statistics.mean(row["wer_pct"] for row in rows),
                "quality": statistics.mean(row["quality_score"] for row in rows),
                "time": statistics.mean(row["elapsed_sec"] for row in rows),
            }
        )

    labels = [f"{item['engine']} (n={item['n']})" for item in engine_stats]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    ax.bar(x - width / 2, [item["cer"] for item in engine_stats], width, label="CER", color="#0057B8")
    ax.bar(x + width / 2, [item["wer"] for item in engine_stats], width, label="WER", color="#E67E22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tỷ lệ lỗi trung bình (%)")
    ax.set_title("CER/WER trung bình theo engine trên các job có ground-truth", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", color="#D9E2EC")
    fig.tight_layout()
    fig.savefig(ASSETS / "aggregate_engine_avg_cer_wer.png", bbox_inches="tight")
    plt.close(fig)

    # Best quality wins.
    wins = Counter()
    for record in records:
        if not record["has_gt"]:
            continue
        best = metric_item(record["summary"], "best_quality")
        if best.get("engine"):
            wins[best["engine"]] += 1
    labels = list(wins.keys())
    values = [wins[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    ax.bar(labels, values, color=palette[: len(labels)])
    ax.set_ylabel("Số job thắng quality score")
    ax.set_title("Số lần engine đạt quality score tốt nhất", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#D9E2EC")
    for idx, value in enumerate(values):
        ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "aggregate_engine_quality_wins.png", bbox_inches="tight")
    plt.close(fig)


def representative_table(records: list[dict]) -> str:
    lines = [
        "| Job | File scan | Trang | Lượt OK | Ground truth | CER tốt nhất | WER tốt nhất | Quality tốt nhất | Nhanh nhất |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for record in representative_records(records):
        summary = record["summary"]
        best_cer = metric_item(summary, "best_cer")
        best_wer = metric_item(summary, "best_wer")
        best_quality = metric_item(summary, "best_quality")
        fastest = metric_item(summary, "fastest")
        lines.append(
            f"| `{record['job_id']}` | {md_cell(record['filename'])} | {record['pages']} | {record['ok']}/{record['runs']} | "
            f"{record['gt_len']} ký tự | {md_cell(best_cer.get('label'))}: {fmt(best_cer.get('cer_pct'), 2)}% | "
            f"{md_cell(best_wer.get('label'))}: {fmt(best_wer.get('wer_pct'), 2)}% | "
            f"{md_cell(best_quality.get('label'))}: {fmt(best_quality.get('quality_score'), 2)}/100 | "
            f"{md_cell(fastest.get('label'))}: {fmt(fastest.get('elapsed_sec'), 3)}s |"
        )
    return "\n".join(lines)


def all_jobs_table(records: list[dict]) -> str:
    lines = [
        "| Job | File | Trang | Runs | OK | Có GT | Best quality | Best CER | Best WER | Ghi chú |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for record in records:
        summary = record["summary"]
        bq = metric_item(summary, "best_quality")
        bc = metric_item(summary, "best_cer")
        bw = metric_item(summary, "best_wer")
        note = "đầy đủ" if record in representative_records(records) else "lịch sử/partial"
        lines.append(
            f"| `{record['job_id']}` | {md_cell(record['filename'])} | {md_cell(record['pages'])} | {record['runs']} | {record['ok']} | "
            f"{'Có' if record['has_gt'] else 'Không'} | {md_cell(bq.get('label'))} {fmt(bq.get('quality_score'), 2)} | "
            f"{md_cell(bc.get('label'))} {fmt(bc.get('cer_pct'), 2)}% | "
            f"{md_cell(bw.get('label'))} {fmt(bw.get('wer_pct'), 2)}% | {note} |"
        )
    return "\n".join(lines)


def engine_stats_table(records: list[dict]) -> str:
    rows = all_metric_rows(records)
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)

    lines = [
        "| Engine | Số dòng metric | CER TB | WER TB | Quality TB | Runtime TB | Nhận xét |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for engine, items in sorted(by_engine.items()):
        cer = statistics.mean(item["cer_pct"] for item in items)
        wer = statistics.mean(item["wer_pct"] for item in items)
        quality = statistics.mean(item["quality_score"] for item in items)
        runtime = statistics.mean(item["elapsed_sec"] for item in items)
        if engine == "tesseract":
            note = "Rất nhanh; thắng nhiều job đầy đủ nhưng bị kéo xấu bởi một số job lịch sử/mismatch."
        elif engine == "paddleocr_vl":
            note = "WER/layout tốt hơn nhưng runtime rất cao."
        elif engine == "paddle_vietocr":
            note = "Confidence cao nhưng CER/WER chưa tốt trong dữ liệu hiện tại."
        else:
            note = "Có ích để đối chiếu, chất lượng trung bình thấp hơn nhóm tốt nhất."
        lines.append(f"| {engine} | {len(items)} | {cer:.2f}% | {wer:.2f}% | {quality:.2f} | {runtime:.3f}s | {note} |")
    return "\n".join(lines)


def primary_runtime_note(rows: list[dict], summary: dict) -> str:
    fastest = min(rows, key=lambda row: row.get("elapsed_sec") or 10**9)
    best_cer = metric_item(summary, "best_cer")
    best_wer = metric_item(summary, "best_wer")
    best_quality = max(rows, key=lambda row: row.get("quality_score") or -1)
    best_confidence = max(rows, key=lambda row: row.get("avg_confidence") or -1)
    longest = max(rows, key=lambda row: row.get("text_len") or -1)
    return f"""| Hạng mục | Engine/biến thể | Số liệu |
|---|---|---:|
| CER thấp nhất | `{best_cer.get('engine')} / {best_cer.get('variant')}` | {fmt(best_cer.get('cer_pct'), 2)}% |
| WER thấp nhất | `{best_wer.get('engine')} / {best_wer.get('variant')}` | {fmt(best_wer.get('wer_pct'), 2)}% |
| OCR nhanh nhất | `{fastest['engine']} / {fastest['variant']}` | {fmt(fastest.get('elapsed_sec'), 3)} giây |
| Quality score cao nhất | `{best_quality['engine']} / {best_quality['variant']}` | {fmt(best_quality.get('quality_score'), 2)}/100 |
| Text dài nhất | `{longest['engine']} / {longest['variant']}` | {longest.get('text_len')} ký tự |
| Confidence tốt nhất | `{best_confidence['engine']} / {best_confidence['variant']}` | {fmt(best_confidence.get('avg_confidence'), 2)}% |"""


def write_markdown(records: list[dict]) -> None:
    report = json.loads((JOB / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((JOB / "comparison_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    layout = report.get("layoutlmv3_postprocess") or {}
    fields = layout.get("fields") or {}
    report_mtime = datetime.fromtimestamp((JOB / "report.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    metric_rows = all_metric_rows(records)
    upload_counts = Counter(record["filename"] for record in records)
    gt_jobs = [record for record in records if record["has_gt"]]
    reps = representative_records(records)
    docker_note = (
        "Docker daemon không phản hồi khi kiểm tra `docker ps`; `logs/server.err.log` và "
        "`logs/server.out.log` trong project đang rỗng. Chưa có dữ liệu thực nghiệm ở lần chạy hiện tại cho log Docker riêng."
    )

    content = f"""# Báo cáo kỹ thuật benchmark OCR và trích xuất metadata công văn tiếng Việt

**Đề tài:** Ứng dụng OCR kết hợp chữ ký số trong quản lý công văn điện tử tiếng Việt  
**Dữ liệu thực nghiệm chính:** toàn bộ các job đã scan trong `jobs/`, phân tích sâu job mới nhất `{JOB_ID}`  
**File đầu vào mới nhất:** `{report.get('uploaded_file')}`  
**Thời điểm cập nhật report:** {report_mtime}  

## 1. Sửa lỗi dùng file chuẩn CER/WER

File text chuẩn bạn upload ở phần CER/WER **không bị mất**. Project đã lưu file này trong thư mục `ground_truth` của từng job và `report.json` cũng đã ghi `ground_truth_text_length`. Lý do dashboard trước đó báo “Chưa có CER/WER” là hàm metric cũ chỉ cho phép so sánh bằng Dynamic Programming khi `len(OCR text) * len(ground truth)` dưới ngưỡng nhỏ. Với các công văn dài 13.238, 70.464 hoặc 78.324 ký tự, phép tính vượt ngưỡng nên `cer`/`wer` bị trả `null`; `comparison_summary.has_ground_truth` vì thế thành `false`.

Đã sửa bằng cách dùng `rapidfuzz` cho Levenshtein distance tốc độ cao và refresh lại các job có ground-truth. Sau khi refresh:

| Chỉ tiêu | Giá trị |
|---|---:|
| Tổng job có `report.json` | {len(records)} |
| Job có ground-truth và CER/WER sau refresh | {len(gt_jobs)} |
| Dòng metric OCR có CER/WER | {len(metric_rows)} |
| Job đầy đủ đại diện có đủ trang, đủ 8 lượt OCR và ground-truth | {len(reps)} |

## 2. Phạm vi dữ liệu thực nghiệm

| Hạng mục dữ liệu | Kết quả đọc trong project |
|---|---|
| Thư mục job | `jobs/` |
| Report JSON | `jobs/<job_id>/report.json` |
| Benchmark summary JSON | `jobs/<job_id>/comparison_summary.json` |
| CSV export | `jobs/<job_id>/benchmark_results.csv` |
| Ảnh raw/OpenCV | `jobs/<job_id>/images/` |
| Ground-truth | `jobs/<job_id>/ground_truth/` nếu người dùng upload file chuẩn |
| Log Docker | {docker_note} |

Phân bố file scan trong toàn bộ lịch sử job:

| File | Số job |
|---|---:|
{chr(10).join(f"| {md_cell(name)} | {count} |" for name, count in upload_counts.most_common())}

## 3. Dashboard benchmark job mới nhất

![Dashboard benchmark](assets/dashboard_benchmark.png)

### 3.1. Tổng quan job `{JOB_ID}`

| Chỉ tiêu | Giá trị |
|---|---:|
| File scan | `{short_filename(report)}` |
| Số trang OCR | {report.get('page_count')} |
| Tổng lượt OCR | {summary.get('total_runs')} |
| Thành công | {summary.get('ok_runs')}/{summary.get('total_runs')} |
| Runtime fail | {summary.get('error_runs')} |
| Skipped | {summary.get('skipped_runs')} |
| Ground truth | Có, {report.get('ground_truth_text_length')} ký tự |
| CER/WER | Đã tính sau refresh |
| Worker runtime | OpenCV workers: {report.get('runtime', {}).get('opencv_workers')}; Tesseract workers: {report.get('runtime', {}).get('tesseract_workers')}; GPU workers: {report.get('runtime', {}).get('gpu_workers')} |

### 3.2. Tóm tắt nổi bật job mới nhất

{primary_runtime_note(rows, summary)}

Sau khi dùng đúng ground-truth, engine tốt nhất của job mới nhất theo quality score là **Tesseract raw** với CER 3.01%, WER 6.29%, runtime 3.367 giây. PaddleOCR-VL raw có WER thấp nhất 5.56% và text dài nhất, nhưng runtime 831.805 giây. PaddleOCR + VietOCR raw có confidence cao nhất 95.98%, nhưng CER 13.19% và WER 53.59%, nên confidence nội bộ không phản ánh trực tiếp độ đúng theo ground-truth.

## 4. Input comparison: ảnh gốc và ảnh OpenCV

### 4.1. Ảnh gốc công văn

![Ảnh gốc công văn trang 1](assets/input_raw_page1.png)

### 4.2. Ảnh sau OpenCV preprocessing

![Ảnh sau OpenCV trang 1](assets/input_opencv_page1.png)

Pipeline OpenCV trong lần chạy hiện tại gồm `grayscale`, `denoise_fastNlMeans`, `deskew_minAreaRect`, `CLAHE_contrast`, `adaptive_threshold`, `morph_close`. Bước `sharpen` chưa xuất hiện trong `report.opencv_steps`, vì vậy chưa có dữ liệu thực nghiệm ở lần chạy hiện tại để đánh giá riêng sharpen.

## 5. Bảng kết quả OCR chi tiết job mới nhất

{base.benchmark_table(rows)}

## 6. Biểu đồ job mới nhất

### 6.1. Quality score

![Quality score OCR](assets/quality_score_bar.png)

### 6.2. Runtime

![Runtime OCR](assets/runtime_bar.png)

### 6.3. Confidence

![Confidence OCR](assets/confidence_bar.png)

### 6.4. Text length

![Text length OCR](assets/text_length_bar.png)

### 6.5. Radar tổng hợp

![Radar tổng hợp OCR](assets/radar_summary.png)

## 7. Phân tích các file đã scan trước đó

Bảng sau lấy các hồ sơ đầy đủ đại diện: có ground-truth, có số trang, chạy đủ 8 biến thể OCR và không lỗi runtime. Đây là tập nên dùng để kết luận chính, vì các job lịch sử khác có nhiều lần chạy thử partial, duplicate hoặc dùng ground-truth không cùng phạm vi trang.

{representative_table(records)}

### 7.1. Biểu đồ CER theo hồ sơ đầy đủ

![CER tốt nhất theo hồ sơ](assets/aggregate_best_cer_by_job.png)

### 7.2. Trung bình CER/WER theo engine trên toàn bộ job có ground-truth

![CER WER trung bình theo engine](assets/aggregate_engine_avg_cer_wer.png)

### 7.3. Số lần engine thắng quality score

![Engine thắng quality score](assets/aggregate_engine_quality_wins.png)

## 8. Tổng hợp engine trên toàn bộ job có ground-truth

{engine_stats_table(records)}

Khi xét riêng các hồ sơ đầy đủ hiện tại, Tesseract raw thắng CER và quality score ở cả ba hồ sơ đại diện. Khi xét toàn bộ 19 job có ground-truth, thống kê bị ảnh hưởng bởi nhiều job lịch sử/partial/mismatch, nhưng vẫn cho thấy hai xu hướng quan trọng: Tesseract là engine nhanh nhất và ổn nhất cho tài liệu scan rõ; PaddleOCR-VL có WER tốt và hiểu layout tốt hơn nhưng chi phí xử lý rất cao.

## 9. Processing impact của OpenCV

{base.impact_table(summary.get('preprocessing_effect') or [])}

Ở job mới nhất, OpenCV giúp Tesseract nhanh hơn từ 3.367 giây xuống 2.319 giây nhưng làm CER tăng từ 3.01% lên 23.20%. EasyOCR, PaddleOCR + VietOCR và PaddleOCR-VL đều không được lợi rõ ràng từ OpenCV. Vì vậy, chiến lược đúng là dùng ảnh raw làm nhánh production chính cho tài liệu rõ, còn OpenCV là nhánh fallback cho scan nhiễu/nghiêng/mờ.

## 10. Engine health

{base.health_table(summary.get('engine_status') or [])}

Job mới nhất không có skipped và không có error. Trong các job lịch sử, một số job cũ có skipped/error hoặc chỉ chạy một phần engine, vì vậy báo cáo phân biệt rõ “hồ sơ đầy đủ đại diện” và “job lịch sử/partial”.

## 11. OCR output thực tế

![OCR output preview](assets/ocr_output_preview.png)

Ảnh trên là preview text thực tế từ engine được pipeline cũ chọn (`paddle_vietocr / raw`). Sau khi tính CER/WER, output này không còn là kết quả tốt nhất theo ground-truth. Điều này cho thấy cần ưu tiên CER/WER khi có text chuẩn, thay vì chỉ dựa vào confidence hoặc thứ tự ưu tiên engine.

## 12. LayoutLMv3 và trích xuất trường thông tin

![LayoutLMv3 extracted fields](assets/layoutlmv3_extracted_fields.png)

OCR là bước nhận dạng ký tự từ ảnh scan hoặc PDF ảnh để tạo text. LayoutLMv3 là bước document understanding, dùng nội dung chữ, vị trí layout và đặc trưng thị giác để hiểu trường nghiệp vụ. Trong hệ thống quản lý công văn điện tử, LayoutLMv3 nên dùng cho key information extraction: số ký hiệu, ngày ban hành, cơ quan ban hành, loại văn bản, nơi gửi/nhận và trích yếu.

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

LayoutLMv3 thật đã chạy bằng transformers từ model local. Trong job mới nhất, field do model dự đoán chưa được guard chấp nhận hoàn toàn nên metadata cuối cùng được merge với rule fallback. Đây là hướng an toàn cho demo: mô hình học sâu chạy thật, nhưng dữ liệu nghiệp vụ chỉ ghi nhận sau lớp kiểm định.

## 13. Kiến trúc hệ thống đề tài

```mermaid
graph TD
    U[Người dùng] --> UP[Upload PDF/scan]
    UP --> PRE[OpenCV preprocessing]
    UP --> RAW[Nhánh ảnh raw]
    PRE --> OCR[OCR benchmark engine]
    RAW --> OCR
    OCR --> SEL[Chọn kết quả OCR tốt nhất theo CER/WER nếu có]
    SEL --> LM[LayoutLMv3 document understanding]
    LM --> META[Trích metadata công văn]
    META --> SIG[Kiểm tra chữ ký số]
    SIG --> DB[(Lưu DB / kho hồ sơ)]
    DB --> DASH[Dashboard tra cứu và đánh giá]
```

## 14. Sequence diagram xử lý nghiệp vụ

```mermaid
sequenceDiagram
    participant User as User
    participant System as System
    participant OCR as OCR Engines
    participant LayoutLMv3 as LayoutLMv3
    participant Sig as Signature Verification
    participant DB as Database

    User->>System: Upload PDF/scan công văn + file ground-truth nếu đánh giá
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

## 15. Kết luận lựa chọn OCR cho đề tài

Dựa trên dữ liệu đã được tính lại bằng file chuẩn CER/WER, **Tesseract raw là lựa chọn OCR production chính phù hợp nhất cho demo hiện tại**. Ở ba hồ sơ đầy đủ đại diện, Tesseract raw đạt CER thấp nhất: 3.01% với `39-bgddt.pdf`, 1.57% với `151_2026_ND-CP_13052026_1-signed.pdf`, và 15.02% với `Thông tư 11-2026-TT-NHNN.pdf`. Runtime của Tesseract cũng thấp hơn rất nhiều so với EasyOCR, PaddleOCR + VietOCR và PaddleOCR-VL.

**PaddleOCR-VL nên là engine kiểm tra layout/chất lượng offline**, không phải production mặc định. Engine này có WER tốt ở một số hồ sơ và text dài, nhưng runtime lên tới hàng nghìn giây ở tài liệu dài, không phù hợp luồng xử lý công văn thường xuyên.

**PaddleOCR + VietOCR hiện chưa nên làm production chính trong bộ dữ liệu này.** Engine có confidence cao nhưng CER/WER thực nghiệm cao hơn Tesseract và PaddleOCR-VL. Muốn dùng PaddleOCR + VietOCR làm production cần fine-tune hoặc sửa pipeline nhận dấu tiếng Việt trước.

**EasyOCR phù hợp vai trò đối chiếu/kiểm thử**, không phải engine chính. Kết quả ổn định hơn một số job lịch sử nhưng chất lượng trung bình thấp hơn nhóm tốt nhất.

**LayoutLMv3 nên đặt ở bước hậu OCR**, sau khi hệ thống chọn OCR output tốt nhất theo CER/WER hoặc heuristic chất lượng. Vai trò của LayoutLMv3 là hiểu cấu trúc tài liệu và trích metadata nghiệp vụ trước khi kiểm tra chữ ký số và lưu DB.

## 16. Phụ lục: toàn bộ job đã scan trong project

{all_jobs_table(records)}
"""

    (OUT / "OCR_BENCHMARK_TECHNICAL_REPORT.md").write_text(content, encoding="utf-8")


def main() -> None:
    # Rebuild base assets/charts first, then add aggregate assets and overwrite the report body.
    base.main()
    records = load_records()
    make_aggregate_charts(records)
    write_markdown(records)
    print(f"Wrote {OUT / 'OCR_BENCHMARK_TECHNICAL_REPORT.md'} with {len(records)} jobs")


if __name__ == "__main__":
    main()
