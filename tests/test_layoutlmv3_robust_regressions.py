from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from app.ocr_engines.base import OCRBox
from app.services.layoutlmv3_postprocess import (
    _bio_to_fields,
    _clean_field_value,
    _field_score,
    _prepare_layout_words,
    finalize_layout_result,
    stabilize_layout_fields_from_rows,
)


EXPECTED_695_SUBJECT = (
    "Bãi bỏ một phần Thông tư số 71/2011/TT-BNNPTNT ngày 25/10/2011 "
    "của Bộ trưởng Bộ Nông nghiệp và Phát triển nông thôn ban hành quy "
    "chuẩn kỹ thuật quốc gia lĩnh vực thú y"
)


class LayoutRobustRegressionTests(unittest.TestCase):
    def test_subject_cleaning_never_cuts_legitimate_internal_verbs(self) -> None:
        values = [
            "Quy định về giờ làm việc và thời giờ nghỉ ngơi",
            "Bãi bỏ quy chế cũ và ban hành quy định mới",
            "Thông tư sửa đổi, bổ sung một số điều của quy định hiện hành",
            "Kế hoạch kinh doanh căn cứ theo nhu cầu thực tế",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(_clean_field_value("trich_yeu", value), value)

    def test_bioes_terminal_token_is_not_dropped(self) -> None:
        fields = _bio_to_fields(
            ["Phê", "duyệt", "hồ", "sơ"],
            ["B-trich_yeu", "I-trich_yeu", "I-trich_yeu", "E-trich_yeu"],
        )
        self.assertEqual(fields["trich_yeu"], "Phê duyệt hồ sơ")

    def test_invalid_calendar_dates_fail_closed(self) -> None:
        self.assertLess(_field_score("ngay_ban_hanh", "31/02/2026"), 0)
        self.assertLess(_field_score("ngay_ban_hanh", "25/09/2033"), 0)

    def test_line_boxes_are_split_and_one_bad_box_does_not_drop_the_page(self) -> None:
        words, boxes, diagnostics = _prepare_layout_words(
            [
                OCRBox(text="BỘ NÔNG NGHIỆP", bbox=[10, 10, 310, 50]),
                OCRBox(text="bad", bbox=[float("nan"), 1, 10, 20]),
                OCRBox(text="outside", bbox=[-100, -100, -10, -10]),
            ],
            width=1000,
            height=1000,
        )
        self.assertEqual(words, ["BỘ", "NÔNG", "NGHIỆP"])
        self.assertEqual(len(boxes), 3)
        self.assertEqual(diagnostics["accepted_source_box_count"], 1)
        self.assertGreaterEqual(diagnostics["rejected_box_count"], 2)
        self.assertEqual(diagnostics["word_box_adapter"], "character_width_interpolation")

    def test_single_ocr_engine_is_visible_as_unverified_not_silently_trusted(self) -> None:
        result = finalize_layout_result(
            {
                "mode": "layoutlmv3_model",
                "fields": {
                    "so_ky_hieu": "12/2026/QĐ-UBND",
                    "ngay_ban_hanh": "14/07/2026",
                    "co_quan_ban_hanh": "ỦY BAN NHÂN DÂN TỈNH BẮC NINH",
                    "loai_van_ban": "QUYẾT ĐỊNH",
                    "trich_yeu": "Về việc phê duyệt chương trình chuyển đổi số năm 2026",
                },
                "stabilization": {
                    "candidate_count": 2,
                    "independent_engine_count": 1,
                    "field_sources": {},
                },
            }
        )
        self.assertTrue(result["review_required"])
        self.assertEqual(
            result["final_field_provenance"]["trich_yeu"]["status"],
            "unverified_single_engine",
        )

    def test_real_695_subject_and_full_issuer_do_not_regress(self) -> None:
        report_path = Path("jobs/695292828457/report.json")
        if not report_path.exists():
            self.skipTest("demo regression report is not available")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stabilized = stabilize_layout_fields_from_rows(
            copy.deepcopy(report.get("layoutlmv3_postprocess")),
            report.get("results") or [],
        )
        fields = stabilized["fields"]
        self.assertEqual(fields["trich_yeu"], EXPECTED_695_SUBJECT)
        self.assertEqual(
            fields["co_quan_ban_hanh"],
            "BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN",
        )


if __name__ == "__main__":
    unittest.main()
