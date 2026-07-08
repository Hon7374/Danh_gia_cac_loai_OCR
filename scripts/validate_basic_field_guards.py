from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.field_extract import extract_fields_rule_based
from app.services.layoutlmv3_postprocess import _merge_model_fields, stabilize_layout_fields_from_rows


def _assert_equal(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _assert_not_contains(value: str, needles: tuple[str, ...], label: str) -> None:
    lowered = value.lower()
    bad = [needle for needle in needles if needle.lower() in lowered]
    if bad:
        raise AssertionError(f"{label}: contains forbidden body marker(s) {bad}: {value!r}")


def test_flattened_nghi_dinh_header_wins_over_body_thong_bao() -> None:
    text = (
        "CHÍNH PHỦ CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM "
        "Số: 151/2026/NĐ-CP Hà Nội, ngày 13 tháng 5 năm 2026 "
        "NGHỊ ĐỊNH Quy định về tổ chức và hoạt động của văn phòng thi hành án dân sự, Thừa hành viên "
        "Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15; "
        "Điều 2. Giải thích từ ngữ 1. Tổng đạt là việc thông báo, giao nhận giấy tờ, hồ sơ, tài liệu."
    )
    fields = extract_fields_rule_based(text).to_dict()
    _assert_equal(fields["loai_van_ban"], "NGHỊ ĐỊNH", "doc type")
    _assert_equal(
        fields["trich_yeu"],
        "Quy định về tổ chức và hoạt động của văn phòng thi hành án dân sự, Thừa hành viên",
        "subject",
    )
    _assert_not_contains(fields["trich_yeu"], ("Tổng đạt", "thông báo, giao nhận", "Điều 2"), "subject")


def test_damaged_ocr_still_does_not_pick_body_thong_bao() -> None:
    text = (
        "CHÍNH PH CNG HÒA X HI CH NGH VIT NAM "
        "Số: 151/2026/NĐ-CP Hà Nội, ngày 13 tháng 5 năm 2026 "
        "NGH DINH Quy đnh v t chc và hot đng ca văn phòng thi hành án dân s, Tha hành viên "
        "Căn c Lut T chc Chính ph s 63/2025/QH15 "
        "Trong Ngh đnh này, các t ng dưi đây đưc hiu như sau "
        "1. Tng đt là vic thông báo, giao nhn giy t, h so, tài liu do Tha hành viên."
    )
    fields = extract_fields_rule_based(text).to_dict()
    _assert_equal(fields["loai_van_ban"], "NGHỊ ĐỊNH", "damaged doc type")
    _assert_not_contains(fields["trich_yeu"], ("Tng đt", "giao nhn", "giy t", "h so"), "damaged subject")


def test_layoutlmv3_model_fields_are_fail_closed_without_rule_support() -> None:
    fallback = {"so_ky_hieu": "", "ngay_ban_hanh": "", "trich_yeu": "", "co_quan_ban_hanh": "", "loai_van_ban": ""}
    bad_model = {
        "loai_van_ban": "THÔNG BÁO",
        "trich_yeu": "1. Tng đt là vic thông báo, giao nhn giy t, h so theo quy đnh",
        "co_quan_ban_hanh": "Số: 151/2026/NĐ-CP",
    }
    merged = _merge_model_fields(fallback, bad_model)
    _assert_equal(merged["loai_van_ban"], "", "model-only doc type")
    _assert_equal(merged["trich_yeu"], "", "model-only subject")
    _assert_equal(merged["co_quan_ban_hanh"], "", "model-only issuer")


def test_layoutlmv3_stabilizer_corrects_bad_model_with_ocr_candidates() -> None:
    layout_result = {
        "fields": {
            "so_ky_hieu": "151/2026/NĐ-CP",
            "ngay_ban_hanh": "13/05/2026",
            "trich_yeu": "1. Tng đt là vic thông báo, giao nhn giy t, h so, tài liu",
            "co_quan_ban_hanh": "Số: 151/2026/NĐ-CP",
            "loai_van_ban": "THÔNG BÁO",
        }
    }
    rows = [
        {
            "engine": "easyocr",
            "variant": "raw",
            "status": "ok",
            "raw": {
                "first_page_text": (
                    "CHÍNH PHỦ CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM "
                    "Số: 151/2026/NĐ-CP Hà Nội, ngày 13 tháng 5 năm 2026 "
                    "NGHỊ ĐỊNH Quy định về tổ chức và hoạt động của văn phòng thi hành án dân sự, Thừa hành viên "
                    "Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15"
                )
            },
            "text": "unused",
        }
    ]
    stabilized = stabilize_layout_fields_from_rows(layout_result, rows)
    fields = stabilized["fields"]
    _assert_equal(fields["loai_van_ban"], "NGHỊ ĐỊNH", "stabilized doc type")
    _assert_equal(fields["co_quan_ban_hanh"], "CHÍNH PHỦ", "stabilized issuer")
    _assert_equal(
        fields["trich_yeu"],
        "Quy định về tổ chức và hoạt động của văn phòng thi hành án dân sự, Thừa hành viên",
        "stabilized subject",
    )


def test_ministry_plan_name_is_not_document_type() -> None:
    text = (
        "BỘ\n"
        "KẾ HOẠCH VÀ ĐẦU TƯ\n"
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Số: 06\n"
        "2023/TT-BKHĐT\n"
        "Hà Nội, ngày 02 tháng 10 năm 2023\n"
        "VĂN PHÒNG CHÍNH PHỦ\n"
        "THÔNG TƯ\n"
        "CÔNG VĂN ĐẾN\n"
        "Ngày 03/10/2023\n"
        "Quy định hệ thống chỉ tiêu thống kê ngành Thống kê\n"
        "Giờ.... Ngày\n"
        "Kinh chuyển\n"
        "Căn cứ Luật Thống kê ngày 23 tháng 11 năm 2015"
    )
    fields = extract_fields_rule_based(text).to_dict()
    _assert_equal(fields["co_quan_ban_hanh"], "BỘ KẾ HOẠCH VÀ ĐẦU TƯ", "ministry issuer")
    _assert_equal(fields["loai_van_ban"], "THÔNG TƯ", "ministry doc type")
    _assert_equal(fields["so_ky_hieu"], "06/2023/TT-BKHĐT", "split document number")
    _assert_equal(fields["ngay_ban_hanh"], "02/10/2023", "ministry issued date")
    _assert_equal(fields["trich_yeu"], "Quy định hệ thống chỉ tiêu thống kê ngành Thống kê", "ministry subject")


def test_layout_stabilizer_prefers_clean_ministry_header() -> None:
    layout_result = {
        "fields": {
            "co_quan_ban_hanh": "#################, BỘ KẾ HOẠCH VÀ ĐẦU TỪ",
            "loai_van_ban": "KẾ HOẠCH",
            "ngay_ban_hanh": "01/10/2023",
            "so_ky_hieu": "06/2023/TT-BKHĐT",
            "trich_yeu": "ĐẾN, THONG y định hệ thông chi tiêu thong kê ngành Thong? Kinh chuyển Căn cứ Luật Thống kê ngày 23 tháng 11 năm 2015",
        }
    }
    rows = [
        {
            "engine": "paddle_vietocr",
            "variant": "raw",
            "status": "ok",
            "raw": {
                "first_page_text": (
                    "BỘ\nKẾ HOẠCH VÀ ĐẦU TƯ\nSố: 06\n2023/TT-BKHĐT\n"
                    "Hà Nội, ngày 02 tháng 10 năm 2023\nTHÔNG TƯ\nCÔNG VĂN ĐẾN\n"
                    "Quy định hệ thống chỉ tiêu thống kê ngành Thống kê\nKinh chuyển"
                )
            },
            "text": "unused",
        },
        {
            "engine": "paddleocr_vl",
            "variant": "raw",
            "status": "ok",
            "raw": {
                "first_page_text": (
                    "BỘ\nKẾ HOẠCH VÀ ĐẦU TỪ\nSố: 05/2023/TT-BKHDT\n"
                    "Hà Nội, ngày 01 tháng 10 năm 2023\nTHÔNG TỪ"
                )
            },
            "text": "unused",
        },
    ]
    fields = stabilize_layout_fields_from_rows(layout_result, rows)["fields"]
    _assert_equal(fields["co_quan_ban_hanh"], "BỘ KẾ HOẠCH VÀ ĐẦU TƯ", "clean stabilized issuer")
    _assert_equal(fields["loai_van_ban"], "THÔNG TƯ", "clean stabilized doc type")
    _assert_equal(fields["ngay_ban_hanh"], "02/10/2023", "clean stabilized date")
    _assert_equal(fields["so_ky_hieu"], "06/2023/TT-BKHĐT", "clean stabilized number")
    _assert_equal(fields["trich_yeu"], "Quy định hệ thống chỉ tiêu thống kê ngành Thống kê", "clean stabilized subject")


def test_real_regression_job_if_present() -> None:
    report_path = Path("jobs/4eb0cda1c107/report.json")
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stabilized = stabilize_layout_fields_from_rows(report.get("layoutlmv3_postprocess"), report.get("results") or [])
    fields = stabilized["fields"]
    _assert_equal(fields["loai_van_ban"], "NGHỊ ĐỊNH", "real job doc type")
    _assert_equal(fields["co_quan_ban_hanh"], "CHÍNH PHỦ", "real job issuer")
    _assert_not_contains(fields["trich_yeu"], ("Tng đt", "giao nhn giy", "thông báo, giao"), "real job subject")


def main() -> None:
    tests = [
        test_flattened_nghi_dinh_header_wins_over_body_thong_bao,
        test_damaged_ocr_still_does_not_pick_body_thong_bao,
        test_layoutlmv3_model_fields_are_fail_closed_without_rule_support,
        test_layoutlmv3_stabilizer_corrects_bad_model_with_ocr_candidates,
        test_ministry_plan_name_is_not_document_type,
        test_layout_stabilizer_prefers_clean_ministry_header,
        test_real_regression_job_if_present,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")


if __name__ == "__main__":
    main()
