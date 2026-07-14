from __future__ import annotations

import json
import unittest

from PIL import Image, ImageDraw

from app.ocr_engines.base import OCRBox
from app.ocr_engines.paddle_vietocr_engine import (
    _apply_hybrid_refinement,
    _combine_segment_predictions,
    _evaluate_vietocr_candidate,
    _has_pathological_repetition,
    _split_long_crop_at_whitespace,
)


class PaddleVietOCRHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = {
            "bbox": [10, 20, 210, 52],
            "polygon": [[10, 20], [210, 20], [210, 52], [10, 52]],
            "label": "text",
        }

    def box(self, text: str, confidence: float | None = 0.9, **geometry) -> OCRBox:
        return OCRBox(text=text, confidence=confidence, **(geometry or self.geometry))

    def test_normal_refinement_is_preferred_but_paddle_geometry_is_preserved(self) -> None:
        paddle = self.box("Cong hoa xa hoi chu nghia Viet Nam", 0.72)
        candidate = self.box(
            "Cộng hòa xã hội chủ nghĩa Việt Nam",
            0.97,
            bbox=[999, 999, 1000, 1000],
            polygon=None,
            label="wrong",
        )

        output, stats = _apply_hybrid_refinement([paddle], [candidate], {"weights": "fine_tuned.pth"})

        self.assertEqual(output[0].text, candidate.text)
        self.assertEqual(output[0].confidence, 0.97)
        self.assertEqual(output[0].bbox, paddle.bbox)
        self.assertEqual(output[0].polygon, paddle.polygon)
        self.assertEqual(output[0].label, paddle.label)
        self.assertEqual(stats["accepted_refinements"], 1)
        self.assertEqual(stats["fallback_to_paddle"], 0)
        self.assertTrue(stats["paddle_geometry_preserved"])
        self.assertEqual(stats["model"]["weights"], "fine_tuned.pth")

    def test_decoder_cap_and_repetition_fall_back_to_paddle(self) -> None:
        paddle = self.box("Quyền và nghĩa vụ của công dân", 0.88)
        capped = self.box("x" * 128, 0.99)
        repeated = self.box("năm 2007 của năm 2007 của năm 2007 của", 0.99)

        output, stats = _apply_hybrid_refinement([paddle, paddle], [capped, repeated])

        self.assertEqual([box.text for box in output], [paddle.text, paddle.text])
        self.assertEqual(stats["fallback_to_paddle"], 2)
        self.assertEqual(stats["fallback_reason_counts"]["decoder_cap_or_missing_eos"], 1)
        self.assertGreaterEqual(stats["fallback_reason_counts"]["pathological_repetition"], 1)

    def test_unreasonable_length_growth_and_shrinkage_fall_back(self) -> None:
        original = self.box("Căn cứ Luật tổ chức chính quyền địa phương", 0.8)
        growth = self.box("Nội dung không đúng " * 8, 0.95)
        shrink = self.box("Luật", 0.95)

        growth_ok, growth_reasons, _ = _evaluate_vietocr_candidate(original, growth)
        shrink_ok, shrink_reasons, _ = _evaluate_vietocr_candidate(original, shrink)

        self.assertFalse(growth_ok)
        self.assertIn("excessive_length_growth", growth_reasons)
        self.assertFalse(shrink_ok)
        self.assertIn("excessive_length_shrinkage", shrink_reasons)

    def test_empty_missing_placeholder_and_low_confidence_fall_back(self) -> None:
        originals = [self.box("Văn bản hành chính", 0.85) for _ in range(4)]
        candidates = [
            self.box("", None),
            self.box("null", 0.99),
            self.box("Văn bản hành chính", 0.01),
        ]

        output, stats = _apply_hybrid_refinement(originals, candidates)

        self.assertEqual([box.text for box in output], [box.text for box in originals])
        self.assertEqual(stats["fallback_to_paddle"], 4)
        self.assertEqual(stats["fallback_reason_counts"]["empty_prediction"], 1)
        self.assertEqual(stats["fallback_reason_counts"]["placeholder_output"], 1)
        self.assertEqual(stats["fallback_reason_counts"]["very_low_confidence"], 1)
        self.assertEqual(stats["fallback_reason_counts"]["missing_prediction"], 1)

    def test_repetition_detector_avoids_normal_administrative_text(self) -> None:
        self.assertTrue(_has_pathological_repetition("Quy Quyền Quy Quyền Quy Quyền"))
        self.assertTrue(_has_pathological_repetition("abcabcabc"))
        self.assertFalse(
            _has_pathological_repetition(
                "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015"
            )
        )

    def test_audit_metadata_is_json_serializable(self) -> None:
        paddle = self.box("Nơi nhận: Như trên", 0.9)
        output, stats = _apply_hybrid_refinement([paddle], [self.box("!!!", 0.9)], {"device": "cpu"})

        payload = json.loads(json.dumps({"boxes": [box.__dict__ for box in output], "stats": stats}))
        self.assertEqual(payload["boxes"][0]["text"], paddle.text)
        self.assertEqual(payload["stats"]["fallback_reason_counts"]["no_alphanumeric_content"], 1)
        self.assertEqual(payload["stats"]["model"]["device"], "cpu")

    def test_long_crop_splits_only_at_a_whitespace_valley(self) -> None:
        image = Image.new("RGB", (800, 40), "white")
        draw = ImageDraw.Draw(image)
        for x0, x1 in ((10, 90), (120, 210), (240, 350), (450, 540), (570, 660), (690, 790)):
            draw.rectangle((x0, 10, x1, 30), fill="black")

        segments, diagnostics = _split_long_crop_at_whitespace(image)

        self.assertIsNotNone(segments)
        self.assertEqual(len(segments or []), 2)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["segment_count"], 2)
        self.assertGreater(diagnostics["cut_columns"][0], 350)
        self.assertLess(diagnostics["cut_columns"][0], 450)
        self.assertEqual(sum(segment.width for segment in segments or []), image.width)

    def test_normal_width_crop_is_not_split(self) -> None:
        segments, diagnostics = _split_long_crop_at_whitespace(Image.new("RGB", (500, 40), "white"))

        self.assertIsNone(segments)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "below_aspect_threshold")

    def test_segment_predictions_join_and_weight_confidence_by_text(self) -> None:
        text, confidence = _combine_segment_predictions(
            [("Cộng hòa", 0.8), ("xã hội chủ nghĩa Việt Nam", 0.95)]
        )

        self.assertEqual(text, "Cộng hòa xã hội chủ nghĩa Việt Nam")
        self.assertIsNotNone(confidence)
        self.assertGreater(float(confidence), 0.88)
        self.assertLess(float(confidence), 0.96)


if __name__ == "__main__":
    unittest.main()
