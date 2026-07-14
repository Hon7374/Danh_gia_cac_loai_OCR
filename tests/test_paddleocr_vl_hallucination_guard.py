from __future__ import annotations

import unittest

from app.ocr_engines.base import OCRBox
from app.ocr_engines.paddleocr_vl_engine import _sanitize_vl_hallucinations


class PaddleOCRVLHallucinationGuardTests(unittest.TestCase):
    def test_removes_dense_consecutive_number_hallucination_box(self) -> None:
        runaway = " ".join(f"Dấu {number}." for number in range(2, 602)) + " Dấu"
        normal = "Thông tư này có hiệu lực từ ngày 11 tháng 11 năm 2023."
        text = f"Mở đầu\n\n{runaway}\n\n{normal}"
        boxes = [
            OCRBox(text=runaway, bbox=[296, 466, 730, 511], label="text"),
            OCRBox(text=normal, bbox=[100, 700, 900, 760], label="text"),
        ]

        cleaned, kept, diagnostics = _sanitize_vl_hallucinations(text, boxes)

        self.assertNotIn("Dấu 600.", cleaned)
        self.assertIn(normal, cleaned)
        self.assertEqual([box.text for box in kept], [normal])
        self.assertEqual(diagnostics["removed_box_count"], 1)
        self.assertEqual(diagnostics["removed"][0]["sequences"][0]["count"], 600)

    def test_keeps_normal_legal_numbered_list_with_content(self) -> None:
        legal = "\n".join(
            f"Điều {number}. Nội dung hợp lệ của điều khoản số {number}."
            for number in range(1, 61)
        )
        boxes = [OCRBox(text=legal, bbox=[10, 10, 1000, 1800], label="text")]

        cleaned, kept, diagnostics = _sanitize_vl_hallucinations(legal, boxes)

        self.assertEqual(cleaned, legal)
        self.assertEqual(kept, boxes)
        self.assertEqual(diagnostics["removed_box_count"], 0)
        self.assertEqual(diagnostics["removed_sequence_count"], 0)

    def test_sanitizes_plain_text_when_layout_boxes_are_unavailable(self) -> None:
        runaway = " ".join(f"Mục {number}." for number in range(1, 81))
        text = f"Nội dung trước. {runaway} Nội dung sau."

        cleaned, kept, diagnostics = _sanitize_vl_hallucinations(text, [])

        self.assertEqual(kept, [])
        self.assertIn("Nội dung trước.", cleaned)
        self.assertIn("Nội dung sau.", cleaned)
        self.assertNotIn("Mục 79.", cleaned)
        self.assertEqual(diagnostics["removed_sequence_count"], 1)


if __name__ == "__main__":
    unittest.main()
