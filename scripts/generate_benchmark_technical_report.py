# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
JOB_ID = "2bbba632396d"
JOB = ROOT / "jobs" / JOB_ID
OUT = ROOT / "reports" / "benchmark_technical_report"
ASSETS = OUT / "assets"


def fmt(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_cell(value) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def ground_truth_desc(report: dict) -> str:
    ground_truth = report.get("ground_truth_file")
    if not isinstance(ground_truth, dict):
        return md_cell(ground_truth)
    filename = ground_truth.get("filename") or "N/A"
    relative_path = ground_truth.get("relative_path") or "N/A"
    size_bytes = ground_truth.get("size_bytes")
    reader = ground_truth.get("reader")
    return f"`{filename}`; path `{relative_path}`; size {size_bytes} bytes; reader `{reader}`"


def row_label(row: dict) -> str:
    variant = row["variant"].replace("opencv_preprocessed", "OpenCV")
    return f"{row['engine']}\n{variant}"


def find_font(name: str) -> str | None:
    candidates = [
        f"C:/Windows/Fonts/{name}",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def load_font(filename: str, size: int):
    font_path = find_font(filename)
    if font_path:
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def copy_input_images() -> None:
    shutil.copy2(JOB / "images" / "39-bgddt_page1.png", ASSETS / "input_raw_page1.png")
    shutil.copy2(
        JOB / "images" / "39-bgddt_page1_opencv_preprocessed.png",
        ASSETS / "input_opencv_page1.png",
    )


def make_bar_chart(rows: list[dict], filename: str, values: list, title: str, xlabel: str, suffix: str = "", log: bool = False) -> None:
    labels = [row_label(row) for row in rows]
    numeric_values = [0 if value is None else value for value in values]
    palette = ["#0057B8", "#00A676", "#E67E22", "#7B2CBF", "#C1121F", "#0081A7", "#6D597A", "#2A9D8F"]

    fig_height = max(5.2, len(numeric_values) * 0.58)
    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=160)
    y_pos = np.arange(len(numeric_values))
    ax.barh(y_pos, numeric_values, color=palette[: len(numeric_values)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    ax.set_xlabel(xlabel + (" (log scale)" if log else ""))
    if log:
        ax.set_xscale("log")

    max_value = max(numeric_values) if numeric_values else 1
    for idx, (draw_value, original_value) in enumerate(zip(numeric_values, values)):
        text = "N/A" if original_value is None else f"{original_value:.2f}{suffix}"
        x = draw_value if draw_value > 0 else max(max_value * 0.01, 0.5)
        ax.text(x, idx, "  " + text, va="center", fontsize=9, color="#0B1F33")

    ax.grid(axis="x", color="#D9E2EC", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / filename, bbox_inches="tight")
    plt.close(fig)


def make_radar_chart(rows: list[dict]) -> None:
    palette = ["#0057B8", "#00A676", "#E67E22", "#7B2CBF", "#C1121F"]
    radar_rows = [row for row in rows if row["variant"] == "raw"]
    radar_rows.append(next(row for row in rows if row["engine"] == "tesseract" and row["variant"] == "opencv_preprocessed"))
    metrics = ["Quality", "Tốc độ", "Confidence", "Text length"]
    fastest = min(row["elapsed_sec"] for row in rows if row.get("elapsed_sec"))
    max_text = max(row["text_len"] for row in rows if row.get("text_len"))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"polar": True}, dpi=160)
    for idx, row in enumerate(radar_rows):
        values = [
            row.get("quality_score") or 0,
            min(100, (fastest / (row.get("elapsed_sec") or fastest)) * 100),
            row.get("avg_confidence") or 0,
            ((row.get("text_len") or 0) / max_text) * 100,
        ]
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=palette[idx % len(palette)],
            linewidth=2,
            label=f"{row['engine']} / {row['variant'].replace('opencv_preprocessed', 'OpenCV')}",
        )
        ax.fill(angles, values, color=palette[idx % len(palette)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("Radar tổng hợp OCR\n(tốc độ được chuẩn hóa theo engine nhanh nhất)", fontsize=14, weight="bold", pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "radar_summary.png", bbox_inches="tight")
    plt.close(fig)


def make_layout_table_image(fields: dict) -> None:
    field_rows = [
        ("Số ký hiệu", fields.get("so_ky_hieu", "")),
        ("Ngày ban hành", fields.get("ngay_ban_hanh", "")),
        ("Cơ quan ban hành", fields.get("co_quan_ban_hanh", "")),
        ("Loại văn bản", fields.get("loai_van_ban", "")),
        ("Nơi gửi", fields.get("noi_gui", "")),
        ("Nơi nhận", fields.get("noi_nhan", "")),
        ("Trích yếu", fields.get("trich_yeu", "")),
        ("Ghi chú độ tin cậy", fields.get("confidence_note", "")),
    ]
    wrapped = [[key, "\n".join(textwrap.wrap(str(value) if value else "N/A", width=78))] for key, value in field_rows]

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=160)
    ax.axis("off")
    ax.set_title("LayoutLMv3 - trường thông tin trích xuất", fontsize=15, weight="bold", loc="left", pad=12)
    table = ax.table(
        cellText=wrapped,
        colLabels=["Trường", "Giá trị"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D8E2EF")
        if row_idx == 0:
            cell.set_facecolor("#EAF2FF")
            cell.set_text_props(weight="bold", color="#004A99")
        elif col_idx == 0:
            cell.set_facecolor("#F3F6FA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")
    for row_idx in range(len(wrapped) + 1):
        table[(row_idx, 0)].set_height(0.085)
        table[(row_idx, 1)].set_height(0.085)
    fig.tight_layout()
    fig.savefig(ASSETS / "layoutlmv3_extracted_fields.png", bbox_inches="tight")
    plt.close(fig)


def make_ocr_preview_image(report: dict) -> None:
    best = report.get("best_engine") or {"engine": "paddle_vietocr", "variant": "raw"}
    best_result = None
    for item in report.get("results", []):
        if item.get("engine") == best.get("engine") and item.get("variant") == best.get("variant"):
            best_result = item
            break
    best_result = best_result or report.get("results", [{}])[0]
    preview_text = (best_result.get("text") or "").strip().replace("\r", "")[:2200]

    title_font = load_font("arialbd.ttf", 34)
    sub_font = load_font("arial.ttf", 20)
    body_font = load_font("arial.ttf", 22)

    width, height = 1500, 950
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 110], fill="#EAF2FF")
    draw.text((40, 28), f"OCR output preview - {best.get('engine')} / {best.get('variant')}", font=title_font, fill="#003B73")
    draw.text((40, 76), f"Nguồn: jobs/{JOB_ID}/report.json, text thực tế của engine được chọn", font=sub_font, fill="#40566B")

    y = 145
    for paragraph in preview_text.split("\n"):
        lines = textwrap.wrap(paragraph, width=118) if paragraph.strip() else [""]
        for line in lines:
            if y > height - 55:
                draw.text((40, y), "...", font=body_font, fill="#0B1F33")
                y = height
                break
            draw.text((40, y), line, font=body_font, fill="#0B1F33")
            y += 32
        if y >= height:
            break
        y += 10
    image.save(ASSETS / "ocr_output_preview.png")


def benchmark_table(rows: list[dict]) -> str:
    lines = [
        "| Engine | Biến thể | Trạng thái | Runtime (s) | Text length | Confidence TB | Quality score | CER | WER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {md_cell(row['engine'])} | {md_cell(row['variant'])} | {md_cell(row['status'])} | "
            f"{fmt(row.get('elapsed_sec'), 3)} | {md_cell(row.get('text_len'))} | {fmt(row.get('avg_confidence'), 2)} | "
            f"{fmt(row.get('quality_score'), 2)} | {md_cell(row.get('cer'))} | {md_cell(row.get('wer'))} |"
        )
    return "\n".join(lines)


def impact_table(preprocessing_effect: list[dict]) -> str:
    lines = [
        "| Engine | Raw status | OpenCV status | Delta time OpenCV - raw (s) | Delta text length | Delta CER | Delta WER | Khuyến nghị | Nhận xét |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for effect in preprocessing_effect:
        engine = effect["engine"]
        recommended = effect.get("recommended_variant") or "raw"
        recommendation = "OpenCV" if recommended == "opencv_preprocessed" else "raw"
        if effect.get("cer_delta") is not None or effect.get("wer_delta") is not None:
            improved = (effect.get("cer_delta") or 0) < 0 or (effect.get("wer_delta") or 0) < 0
            note = "OpenCV cải thiện ít nhất một metric lỗi." if improved else "OpenCV làm metric lỗi xấu hơn."
        elif recommended == "opencv_preprocessed":
            note = "OpenCV được chọn theo quality score/heuristic."
        else:
            note = "Raw được ưu tiên, OpenCV chỉ dùng fallback."
        lines.append(
            f"| {engine} | {effect.get('raw_status')} | {effect.get('pre_status')} | {fmt(effect.get('time_delta'), 3)} | "
            f"{md_cell(effect.get('text_len_delta'))} | {md_cell(effect.get('cer_delta'))} | {md_cell(effect.get('wer_delta'))} | {recommendation} | {note} |"
        )
    return "\n".join(lines)


def health_table(engine_status: list[dict]) -> str:
    lines = [
        "| Engine | OK | Skipped | Error | Tổng lượt | Tỷ lệ OK |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in engine_status:
        total = item.get("total") or 0
        ok = item.get("ok") or 0
        ratio = (ok / total * 100) if total else 0
        lines.append(f"| {item.get('engine')} | {ok} | {item.get('skipped')} | {item.get('error')} | {total} | {ratio:.0f}% |")
    return "\n".join(lines)


def layout_fields_table(fields: dict) -> str:
    lines = [
        "| Trường | Giá trị trích xuất | Nguồn/nhận xét |",
        "|---|---|---|",
    ]
    for name, key in [
        ("Số ký hiệu", "so_ky_hieu"),
        ("Ngày ban hành", "ngay_ban_hanh"),
        ("Cơ quan ban hành", "co_quan_ban_hanh"),
        ("Loại văn bản", "loai_van_ban"),
        ("Nơi gửi", "noi_gui"),
        ("Nơi nhận", "noi_nhan"),
        ("Trích yếu", "trich_yeu"),
    ]:
        lines.append(f"| {name} | {md_cell(fields.get(key))} | {md_cell(fields.get('confidence_note'))} |")
    return "\n".join(lines)


def write_report(report: dict, summary: dict, rows: list[dict]) -> None:
    layout = report.get("layoutlmv3_postprocess") or {}
    fields = layout.get("fields") or {}
    preprocessing_effect = summary.get("preprocessing_effect") or []
    engine_status = summary.get("engine_status") or []

    fastest = min(rows, key=lambda row: row.get("elapsed_sec") or 10**9)
    best_quality = summary.get("best_quality") or max(rows, key=lambda row: row.get("quality_score") or -1)
    best_confidence = summary.get("best_confidence") or max(rows, key=lambda row: row.get("avg_confidence") or -1)
    longest = max(rows, key=lambda row: row.get("text_len") or -1)
    report_mtime = datetime.fromtimestamp((JOB / "report.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    docker_note = (
        "Docker daemon không phản hồi khi kiểm tra `docker ps`; `logs/server.err.log` và "
        "`logs/server.out.log` trong project đang rỗng. Chưa có dữ liệu thực nghiệm ở lần chạy hiện tại cho log Docker riêng."
    )

    content = f"""# Báo cáo kỹ thuật benchmark OCR và trích xuất metadata công văn tiếng Việt

**Đề tài:** Ứng dụng OCR kết hợp chữ ký số trong quản lý công văn điện tử tiếng Việt  
**Dữ liệu thực nghiệm chính:** job `{JOB_ID}` trong project local  
**File đầu vào:** `{report.get('uploaded_file')}`  
**Số trang scan:** {report.get('page_count')}  
**Thời điểm ghi nhận report:** {report_mtime}  

## 1. Phạm vi và nguồn dữ liệu thực nghiệm

Báo cáo này được lập trực tiếp từ dữ liệu có trong project, ưu tiên các file JSON/CSV/ảnh đã sinh sau lần chạy demo. Không sử dụng benchmark Internet và không tự tạo số liệu minh họa.

| Hạng mục dữ liệu | Kết quả đọc trong project |
|---|---|
| Job benchmark chính | `jobs/{JOB_ID}` |
| Report JSON | `jobs/{JOB_ID}/report.json` |
| Benchmark summary JSON | `jobs/{JOB_ID}/comparison_summary.json` |
| CSV export | `jobs/{JOB_ID}/benchmark_results.csv` |
| Ảnh trang gốc | `jobs/{JOB_ID}/images/39-bgddt_page1.png` đến `page7.png` |
| Ảnh OpenCV preprocessing | `jobs/{JOB_ID}/images/39-bgddt_page1_opencv_preprocessed.png` đến `page7_opencv_preprocessed.png` |
| Ground truth file | {ground_truth_desc(report)} |
| Ground truth text length trong report | `{report.get('ground_truth_text_length')}` ký tự |
| `output/`, `results/` ở project root | Chưa có thư mục dữ liệu thực nghiệm riêng ở lần chạy hiện tại; dữ liệu thực tế nằm trong `jobs/`. |
| Log Docker | {docker_note} |
| Kết quả LayoutLMv3 | Có trong `report.json`, mục `layoutlmv3_postprocess`. |

> Lưu ý về Accuracy: `report.json` có ghi nhận ground-truth `.docx`, nhưng `comparison_summary.json` hiện đặt `has_ground_truth=false` và các trường `cer`, `wer` đều là `null`. Vì vậy dashboard hiện chưa tính CER/WER cho lần chạy này; báo cáo không tự tính lại CER/WER ngoài pipeline để tránh làm lệch kết quả demo.

## 2. Dashboard benchmark hiện tại

![Dashboard benchmark](assets/dashboard_benchmark.png)

### 2.1. Tổng quan benchmark

| Chỉ tiêu | Giá trị |
|---|---:|
| Số engine logic | 4 engine: Tesseract, EasyOCR, PaddleOCR + VietOCR, PaddleOCR-VL |
| Số cấu hình benchmark | {summary.get('total_runs')} lượt: mỗi engine chạy `raw` và `opencv_preprocessed` |
| Số trang OCR | {report.get('page_count')} trang |
| Lượt thành công | {summary.get('ok_runs')}/{summary.get('total_runs')} |
| Runtime fail | {summary.get('error_runs')} |
| Skipped | {summary.get('skipped_runs')} |
| Issue runs | {summary.get('issue_runs')} |
| Text chuẩn/CER-WER | Có file ground-truth trong job, nhưng dashboard summary báo `has_ground_truth=false`, nên CER/WER chưa được tính. |
| Worker runtime | OpenCV workers: {report.get('runtime', {}).get('opencv_workers')}; Tesseract workers: {report.get('runtime', {}).get('tesseract_workers')}; GPU workers: {report.get('runtime', {}).get('gpu_workers')} |

### 2.2. Tóm tắt nổi bật

| Hạng mục | Engine/biến thể | Số liệu |
|---|---|---:|
| OCR nhanh nhất | `{fastest['engine']} / {fastest['variant']}` | {fmt(fastest.get('elapsed_sec'), 3)} giây |
| Quality score cao nhất | `{best_quality['engine']} / {best_quality['variant']}` | {fmt(best_quality.get('quality_score'), 2)}/100 |
| Text dài nhất | `{longest['engine']} / {longest['variant']}` | {longest.get('text_len')} ký tự |
| Confidence tốt nhất | `{best_confidence['engine']} / {best_confidence['variant']}` | {fmt(best_confidence.get('avg_confidence'), 2)}% |
| Engine được report chọn | `{report.get('best_engine', {}).get('engine')} / {report.get('best_engine', {}).get('variant')}` | Theo `report.json` |

Nhìn tổng thể, toàn bộ 8 lượt OCR đều chạy thành công. PaddleOCR-VL tạo văn bản dài nhất và đạt quality score tương đối cao nhất theo tiêu chí độ dài text, nhưng runtime rất lớn. PaddleOCR + VietOCR được pipeline chọn làm engine tốt nhất và có confidence cao nhất. Tesseract là engine nhanh nhất, đặc biệt khi chạy trên ảnh OpenCV.

## 3. Input comparison: ảnh gốc và ảnh sau OpenCV

### 3.1. Ảnh gốc công văn

![Ảnh gốc công văn trang 1](assets/input_raw_page1.png)

### 3.2. Ảnh sau OpenCV preprocessing

![Ảnh sau OpenCV trang 1](assets/input_opencv_page1.png)

Pipeline OpenCV trong lần chạy hiện tại gồm:

1. `grayscale`: chuyển ảnh sang mức xám để giảm nhiễu màu.
2. `denoise_fastNlMeans`: khử nhiễu bằng Non-local Means.
3. `deskew_minAreaRect`: hiệu chỉnh nghiêng dựa trên vùng chữ.
4. `CLAHE_contrast`: tăng tương phản cục bộ.
5. `adaptive_threshold`: nhị phân hóa thích nghi theo vùng ảnh.
6. `morph_close`: đóng hình thái học để liền nét ký tự.

Đối chiếu với pipeline xử lý ảnh mục tiêu của đề tài: grayscale, denoise, deskew, CLAHE, threshold và morphology đều đã có trong job hiện tại. Bước `sharpen` chưa xuất hiện trong `report.opencv_steps`, vì vậy chưa có dữ liệu thực nghiệm ở lần chạy hiện tại để đánh giá tác động riêng của sharpen.

Về mặt thị giác, ảnh OpenCV có nền sạch hơn và chữ nổi bật hơn, nhưng preprocessing không phải lúc nào cũng cải thiện OCR. Với tài liệu scan đã tương đối rõ, việc threshold/morphology có thể làm mất dấu tiếng Việt hoặc làm dày nét, khiến một số engine deep learning mất thông tin ảnh gốc.

## 4. Bảng kết quả OCR chi tiết

{benchmark_table(rows)}

### 4.1. Nhận xét theo engine

**Tesseract raw.** Runtime 3.367 giây, text length 13.303 ký tự, confidence 95.00% và quality score 94.97. Đây là cấu hình rất nhanh, ổn định và cho text dài. Với tài liệu hành chính scan rõ, Tesseract vẫn là baseline mạnh.

**Tesseract OpenCV.** Runtime giảm xuống 2.319 giây, là kết quả nhanh nhất toàn bộ benchmark. Tuy nhiên text length giảm còn 10.654 ký tự và quality score giảm còn 89.39. Điều này cho thấy OpenCV giúp tốc độ nhưng làm mất một phần thông tin chữ.

**EasyOCR raw.** Runtime 69.827 giây, text length 13.236 ký tự, confidence 87.16%, quality score 88.97. EasyOCR nhận được nhiều text nhưng confidence thấp hơn Tesseract và PaddleOCR + VietOCR.

**EasyOCR OpenCV.** Runtime tăng lên 89.012 giây, confidence giảm còn 80.89%, quality score giảm còn 84.17. OpenCV không có lợi cho EasyOCR trong lần chạy này.

**PaddleOCR + VietOCR raw.** Runtime 88.654 giây, text length 11.881 ký tự, confidence 95.98%, quality score 93.17. Đây là engine được `report.json` chọn làm tốt nhất. Ưu điểm chính là confidence cao nhất và phù hợp định hướng nhận dạng tiếng Việt trong công văn.

**PaddleOCR + VietOCR OpenCV.** Runtime 98.409 giây, text length 11.929 ký tự, confidence 94.63%, quality score 92.24. OpenCV làm tăng nhẹ số ký tự nhưng giảm confidence và tăng thời gian xử lý.

**PaddleOCR-VL raw.** Runtime 831.805 giây, text length 14.023 ký tự, quality score 100.00 theo tiêu chí relative text length. Engine này có lợi về biểu diễn layout/text dài, nhưng không có confidence trong dashboard và thời gian xử lý quá cao cho production thông thường.

**PaddleOCR-VL OpenCV.** Runtime 957.386 giây, text length 14.021 ký tự, quality score 99.99. OpenCV không cải thiện đáng kể text và còn làm runtime tăng thêm 125.581 giây.

## 5. Biểu đồ từ dữ liệu hiện tại

### 5.1. Quality score

![Quality score OCR](assets/quality_score_bar.png)

### 5.2. Runtime

![Runtime OCR](assets/runtime_bar.png)

### 5.3. Confidence

![Confidence OCR](assets/confidence_bar.png)

### 5.4. Text length

![Text length OCR](assets/text_length_bar.png)

### 5.5. Radar tổng hợp

![Radar tổng hợp OCR](assets/radar_summary.png)

Radar sử dụng các trục: quality score, speed chuẩn hóa theo engine nhanh nhất, confidence và text length. PaddleOCR-VL nổi bật ở text length/quality nhưng rất yếu ở speed và thiếu confidence; Tesseract nổi bật ở speed; PaddleOCR + VietOCR cân bằng tốt giữa confidence và chất lượng tiếng Việt theo lựa chọn của pipeline.

## 6. Accuracy: CER/WER

Dashboard hiện chưa có ground truth hợp lệ trong `comparison_summary.json` nên chưa tính CER/WER. Cụ thể:

| Trường | Giá trị |
|---|---|
| `report.ground_truth_file` | {ground_truth_desc(report)} |
| `report.ground_truth_text_length` | `{report.get('ground_truth_text_length')}` |
| `comparison_summary.has_ground_truth` | `{summary.get('has_ground_truth')}` |
| CER trong từng row | `null` |
| WER trong từng row | `null` |

Do CER/WER chưa được pipeline ghi nhận, đánh giá ở lần chạy này dựa trên các chỉ số có thật trong dashboard: runtime, text length, confidence và quality score. Với báo cáo NCKH chính thức, cần bổ sung bước chuẩn hóa ground-truth `.docx` để dashboard tính CER/WER tự động ở lần chạy kế tiếp.

## 7. Processing impact của OpenCV

{impact_table(preprocessing_effect)}

Kết quả cho thấy OpenCV không nên áp dụng cứng cho mọi engine. Tesseract được lợi về tốc độ nhưng giảm text length. EasyOCR và PaddleOCR-VL bị chậm hơn. PaddleOCR + VietOCR chỉ tăng nhẹ text length nhưng giảm confidence/quality. Vì vậy, chiến lược phù hợp là giữ ảnh raw làm nhánh chính cho engine học sâu, còn OpenCV là nhánh fallback khi ảnh scan nhiễu, nghiêng hoặc tương phản kém.

## 8. Engine health

{health_table(engine_status)}

Tất cả engine đều đạt 2/2 lượt thành công, không có skipped và không có error. Về độ ổn định runtime trong dashboard hiện tại, Tesseract là engine nhẹ nhất và có thể dùng làm fallback nhanh. Về engine production cho bài toán tiếng Việt, PaddleOCR + VietOCR được ưu tiên vì pipeline đã chọn cấu hình này và confidence cao nhất trong lần chạy.

## 9. OCR output thực tế

![OCR output preview](assets/ocr_output_preview.png)

Ảnh trên là phần preview text thực tế lấy từ engine được `report.json` chọn: `{report.get('best_engine', {}).get('engine')} / {report.get('best_engine', {}).get('variant')}`. Text vẫn còn lỗi mất dấu ở một số cụm, ví dụ các ký tự tiếng Việt trong phần đầu văn bản, nhưng metadata quan trọng vẫn có thể được hậu xử lý bằng rule và LayoutLMv3.

## 10. LayoutLMv3 và trích xuất trường thông tin

![LayoutLMv3 extracted fields](assets/layoutlmv3_extracted_fields.png)

### 10.1. Phân biệt OCR và LayoutLMv3

OCR là bước nhận dạng ký tự từ ảnh scan hoặc PDF ảnh để tạo text. OCR trả lời câu hỏi: “trên ảnh có chữ gì?”.

LayoutLMv3 là mô hình document understanding, kết hợp nội dung chữ, vị trí layout và biểu diễn thị giác của tài liệu. LayoutLMv3 trả lời câu hỏi: “đoạn chữ đó thuộc trường nghiệp vụ nào?”. Trong đề tài quản lý công văn điện tử, LayoutLMv3 phù hợp cho key information extraction, trích metadata, hỗ trợ kiểm tra hồ sơ và làm bước hậu xử lý sau OCR trước khi lưu trữ/kiểm tra chữ ký số.

### 10.2. Kết quả LayoutLMv3 trong lần chạy

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
| Model word count | `{layout.get('model_word_count')}` |
| Accepted model fields | `{len(layout.get('accepted_model_fields') or {})}` |
| Rejected model fields | `{len(layout.get('rejected_model_fields') or {})}` |

{layout_fields_table(fields)}

LayoutLMv3 thật đã chạy bằng transformers từ model local. Tuy nhiên, trong lần chạy này trường `trich_yeu` do model dự đoán bị guard kiểm định loại bỏ vì chưa đủ tin cậy; metadata cuối cùng được merge từ rule fallback trên OCR tốt nhất. Đây là thiết kế hợp lý cho demo nghiên cứu: mô hình học sâu được chạy thật, nhưng output nghiệp vụ chỉ nhận field qua lớp kiểm định để tránh ghi metadata sai vào hồ sơ công văn.

## 11. Kiến trúc hệ thống đề tài

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

## 12. Sequence diagram xử lý nghiệp vụ

```mermaid
sequenceDiagram
    participant User as User
    participant System as System
    participant OCR as OCR Engines
    participant LayoutLMv3 as LayoutLMv3
    participant Sig as Signature Verification
    participant DB as Database

    User->>System: Upload PDF/scan công văn
    System->>System: Tách trang ảnh và sinh nhánh OpenCV
    System->>OCR: Chạy benchmark OCR raw/OpenCV
    OCR-->>System: Trả text, runtime, confidence, quality score
    System->>System: Chọn OCR output tốt nhất
    System->>LayoutLMv3: Gửi text + layout/image để trích field
    LayoutLMv3-->>System: Trả metadata và nhãn field
    System->>Sig: Kiểm tra chữ ký số/PAdES nếu có
    Sig-->>System: Trả trạng thái chữ ký và chứng thư
    System->>DB: Lưu file, OCR output, metadata, kết quả đánh giá
    User->>System: Tra cứu dashboard/report
    System-->>User: Hiển thị hồ sơ, OCR benchmark và metadata
```

## 13. Định hướng tích hợp chữ ký số

Theo kế hoạch triển khai, module chữ ký số nên nằm sau bước OCR + LayoutLMv3. Lý do là hệ thống cần biết metadata công văn trước khi kiểm tra, đối soát và lưu trạng thái ký số. Luồng xử lý phù hợp là:

1. OCR tạo text từ công văn scan/PDF.
2. LayoutLMv3 trích `số ký hiệu`, `ngày ban hành`, `cơ quan ban hành`, `loại văn bản`, `trích yếu`.
3. Module chữ ký số kiểm tra chữ ký, chứng thư, thời điểm ký và tính toàn vẹn file.
4. Database lưu đồng thời file gốc, text OCR, metadata, trạng thái chữ ký và log xử lý.

Cách đặt này giúp chữ ký số không chỉ là kiểm tra file, mà còn liên kết được với nghiệp vụ quản lý công văn đi/đến.

## 14. Kết luận lựa chọn OCR cho đề tài

Dựa trên dữ liệu benchmark hiện tại, **PaddleOCR + VietOCR raw là lựa chọn phù hợp nhất để làm OCR production chính cho đề tài**. Lý do: pipeline thực nghiệm đã chọn engine này trong `report.json`, confidence trung bình cao nhất đạt 95.98%, runtime 88.654 giây cho 7 trang vẫn chấp nhận được trong bài toán quản lý công văn, và hướng tiếp cận PaddleOCR + VietOCR phù hợp hơn với nhận dạng tiếng Việt hành chính so với các baseline tổng quát.

**Tesseract nên được dùng làm OCR backup/fallback nhanh.** Tesseract raw đạt quality score 94.97 và runtime chỉ 3.367 giây; Tesseract OpenCV là cấu hình nhanh nhất với 2.319 giây. Khi cần phản hồi nhanh, kiểm tra sơ bộ hoặc khi GPU không sẵn sàng, Tesseract là phương án dự phòng tốt.

**EasyOCR nên dùng cho kiểm thử đối chiếu, không nên là production chính.** Engine này chạy thành công và nhận được text dài, nhưng confidence thấp hơn và OpenCV làm giảm chất lượng trong lần chạy hiện tại.

**PaddleOCR-VL nên dùng cho phân tích layout chuyên sâu hoặc batch offline, không nên làm OCR production mặc định.** Engine này đạt quality score 100.00 và text length cao nhất 14.023 ký tự, nhưng runtime 831.805 giây ở raw và 957.386 giây ở OpenCV là quá cao cho luồng xử lý công văn thông thường.

**LayoutLMv3 nên đặt ở bước hậu OCR**, sau khi hệ thống đã chọn OCR output tốt nhất và trước bước kiểm tra chữ ký số/lưu DB. Vai trò của LayoutLMv3 là document understanding và trích metadata nghiệp vụ, không thay thế OCR. Với thiết kế này, OCR đảm nhiệm nhận chữ, LayoutLMv3 hiểu cấu trúc công văn, còn chữ ký số đảm nhiệm xác thực tính pháp lý và toàn vẹn tài liệu.

## 15. Khuyến nghị hoàn thiện lần chạy tiếp theo

1. Sửa pipeline ground-truth để `comparison_summary.has_ground_truth=true` khi đã có file `.docx`, từ đó tính CER/WER tự động.
2. Lưu docker logs thành file trong job để báo cáo tái lập được môi trường container.
3. Với ảnh scan rõ, đặt raw branch làm mặc định; OpenCV branch dùng cho ảnh nhiễu/nghiêng/mờ.
4. Fine-tune thêm LayoutLMv3 trên tập công văn đã gán nhãn để tăng số field được model chấp nhận thay vì phải dựa nhiều vào rule fallback.
5. Lưu cả preview OCR và metadata sau kiểm định vào hồ sơ để phục vụ kiểm toán kết quả xử lý.
"""

    (OUT / "OCR_BENCHMARK_TECHNICAL_REPORT.md").write_text(content, encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    report = json.loads((JOB / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((JOB / "comparison_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    layout = report.get("layoutlmv3_postprocess") or {}
    fields = layout.get("fields") or {}

    copy_input_images()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    make_bar_chart(rows, "quality_score_bar.png", [row.get("quality_score") for row in rows], "Quality score theo OCR engine", "Quality score (0-100)", suffix="/100")
    make_bar_chart(rows, "runtime_bar.png", [row.get("elapsed_sec") for row in rows], "Thời gian xử lý theo OCR engine", "Giây", suffix="s", log=True)
    make_bar_chart(rows, "confidence_bar.png", [row.get("avg_confidence") for row in rows], "Confidence trung bình theo OCR engine", "Confidence (%)", suffix="%")
    make_bar_chart(rows, "text_length_bar.png", [row.get("text_len") for row in rows], "Độ dài text OCR theo engine", "Số ký tự", suffix=" ký tự")
    make_radar_chart(rows)
    make_layout_table_image(fields)
    make_ocr_preview_image(report)
    write_report(report, summary, rows)

    print(f"Wrote {OUT / 'OCR_BENCHMARK_TECHNICAL_REPORT.md'}")
    for asset in sorted(ASSETS.glob("*")):
        print(f"Asset {asset.name}: {asset.stat().st_size} bytes")


if __name__ == "__main__":
    main()
