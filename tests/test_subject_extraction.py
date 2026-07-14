from __future__ import annotations

import unittest

from app.main import _layout_postprocess_needs_refresh
from app.services.field_extract import extract_fields_rule_based
from app.services.layoutlmv3_postprocess import _bio_to_fields


EXPECTED_SUBJECT = (
    "Bãi bỏ một phần Thông tư số 71/2011/TT-BNNPTNT ngày 25/10/2011 "
    "của Bộ trưởng Bộ Nông nghiệp và Phát triển nông thôn ban hành quy "
    "chuẩn kỹ thuật quốc gia lĩnh vực thú y"
)


class SubjectExtractionTests(unittest.TestCase):
    def test_normative_title_keeps_inner_ban_hanh_phrase(self) -> None:
        text = f"""\
BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN
THÔNG TƯ
{EXPECTED_SUBJECT}
Căn cứ Nghị định số 105/2022/NĐ-CP ngày 22 tháng 12 năm 2022;
Điều 1. Bãi bỏ khoản 2, khoản 3 Điều 1.
"""

        fields = extract_fields_rule_based(text)

        self.assertEqual(fields.trich_yeu, EXPECTED_SUBJECT)

    def test_bio_subject_prefers_long_title_span_over_separate_date_span(self) -> None:
        title_lines = [
            "Bãi bỏ một phần Thông tư số 71/2011/TT-BNNPTNT ngày 25/10/2011",
            "của Bộ trưởng Bộ Nông nghiệp và Phát triển nông thôn ban hành quy",
            "chuẩn kỹ thuật quốc gia lĩnh vực thú y",
        ]
        words = [
            "Hà Nội, ngày 27 tháng 9 năm 2023",
            "THÔNG TƯ",
            *title_lines,
            "Căn cứ Nghị định số 105/2022/NĐ-CP",
        ]
        labels = [
            "B-trich_yeu",
            "B-loai_van_ban",
            "B-trich_yeu",
            "I-trich_yeu",
            "I-trich_yeu",
            "O",
        ]

        fields = _bio_to_fields(words, labels)

        self.assertEqual(fields["trich_yeu"], EXPECTED_SUBJECT)
        self.assertNotIn("Hà Nội, ngày", fields["trich_yeu"])

    def test_enactment_sentence_with_ocr_thong_hu_recovers_canonical_subject(self) -> None:
        text = """\
BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN
THÔNG TƯ
Căn cứ Nghị định số 105/2022/NĐ-CP ngày 22 tháng 12 năm 2022;
Theo đề nghị của Cục trưởng Cục Thú y;
Bộ trưởng Bộ Nông nghiệp và Phát triển nông thôn ban hành Thông hư
Bãi bỏ một phần Thông hư số 71/2011/TT-BNNPTNT ngày 25/10/2011
của Bộ trưởng Bộ Nông nghiệp và Phát triển nông thôn ban hành quy
chuẩn kỹ thuật quốc gia lĩnh vực thú y.
Điều 1. Bãi bỏ khoản 2, khoản 3 Điều 1.
"""

        fields = extract_fields_rule_based(text)

        self.assertEqual(fields.trich_yeu, EXPECTED_SUBJECT)

    def test_legacy_layout_without_version_is_refreshed_even_when_source_matches(self) -> None:
        report = {
            "input_mode": "scan",
            "results": [
                {
                    "engine": "paddle_vietocr",
                    "variant": "raw",
                    "status": "ok",
                    "text": EXPECTED_SUBJECT,
                    "raw": {"first_page_text": EXPECTED_SUBJECT},
                    "boxes": [
                        {
                            "page": 1,
                            "text": EXPECTED_SUBJECT,
                            "confidence": 0.95,
                            "bbox": [100, 200, 900, 260],
                        }
                    ],
                }
            ],
            "layoutlmv3_postprocess": {
                "fields": {"trich_yeu": "stale subject"},
                "ocr_source": {
                    "engine": "paddle_vietocr",
                    "variant": "raw",
                },
            },
        }

        self.assertTrue(_layout_postprocess_needs_refresh(report))


if __name__ == "__main__":
    unittest.main()
