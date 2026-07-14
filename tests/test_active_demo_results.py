from __future__ import annotations

import unittest

from app.main import _is_active_demo_result, build_comparison_summary


ACTIVE_ENGINES = ("tesseract", "easyocr", "paddle_vietocr", "paddleocr_vl")
ACTIVE_VARIANTS = ("raw", "opencv_preprocessed")


def _result_row(engine: str, variant: str, index: int) -> dict:
    return {
        "engine": engine,
        "variant": variant,
        "status": "ok",
        "text": f"Ket qua OCR {engine} {variant}",
        "elapsed_sec": float(index + 1),
        "cer": (index + 1) / 100,
        "wer": (index + 2) / 100,
        "ground_truth_compare_length": 100,
        "boxes": [{"confidence": 0.9}],
        "raw": {},
    }


class ActiveDemoResultTests(unittest.TestCase):
    def test_all_supported_engines_allow_raw_and_opencv_variants(self) -> None:
        for engine in ACTIVE_ENGINES:
            for variant in ACTIVE_VARIANTS:
                with self.subTest(engine=engine, variant=variant):
                    self.assertTrue(
                        _is_active_demo_result({"engine": engine, "variant": variant})
                    )

    def test_glm_is_rejected_for_both_variants(self) -> None:
        for variant in ACTIVE_VARIANTS:
            with self.subTest(variant=variant):
                self.assertFalse(
                    _is_active_demo_result({"engine": "glm_ocr", "variant": variant})
                )

    def test_filtered_summary_contains_eight_rows_and_four_opencv_comparisons(self) -> None:
        rows = []
        index = 0
        for engine in (*ACTIVE_ENGINES, "glm_ocr"):
            for variant in ACTIVE_VARIANTS:
                rows.append(_result_row(engine, variant, index))
                index += 1

        active_rows = [row for row in rows if _is_active_demo_result(row)]
        summary = build_comparison_summary(active_rows)

        expected_pairs = {
            (engine, variant)
            for engine in ACTIVE_ENGINES
            for variant in ACTIVE_VARIANTS
        }
        actual_pairs = {
            (row["engine"], row["variant"])
            for row in summary["rows"]
        }

        self.assertEqual(summary["total_runs"], 8)
        self.assertEqual(len(summary["rows"]), 8)
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertNotIn("glm_ocr", {row["engine"] for row in summary["rows"]})

        effects = summary["preprocessing_effect"]
        self.assertEqual(len(effects), 4)
        self.assertEqual({effect["engine"] for effect in effects}, set(ACTIVE_ENGINES))


if __name__ == "__main__":
    unittest.main()
