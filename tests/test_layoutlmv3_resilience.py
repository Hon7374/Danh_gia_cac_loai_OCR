from __future__ import annotations

import copy
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import main
from app.ocr_engines.base import OCRBox
from app.services import layoutlmv3_postprocess as layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _document_text(
    *,
    doc_type: str = "QUYẾT ĐỊNH",
    issued_date: tuple[int, int, int] = (14, 7, 2026),
    subject: str = "Về việc phê duyệt chương trình chuyển đổi số năm 2026",
) -> str:
    day, month, year = issued_date
    return (
        "ỦY BAN NHÂN DÂN TỈNH BẮC NINH\n"
        "Số: 12/2026/QĐ-UBND\n"
        f"Bắc Ninh, ngày {day:02d} tháng {month:02d} năm {year}\n"
        f"{doc_type}\n"
        f"{subject}\n"
        "Căn cứ Luật Tổ chức chính quyền địa phương;"
    )


def _row(
    engine: str,
    text: str,
    *,
    variant: str = "raw",
    first_page_text: str | None = None,
    boxes: list | None = None,
    status: str = "ok",
) -> dict:
    return {
        "engine": engine,
        "variant": variant,
        "status": status,
        "text": text,
        "raw": {"first_page_text": first_page_text or text},
        "boxes": boxes or [],
    }


def _layout_result(fields: dict | None = None) -> dict:
    return {
        "postprocess_version": layout.POSTPROCESS_VERSION,
        "pipeline_fingerprint": layout.layout_pipeline_fingerprint(),
        "fields": fields
        or {
            "so_ky_hieu": "",
            "ngay_ban_hanh": "",
            "co_quan_ban_hanh": "",
            "loai_van_ban": "",
            "trich_yeu": "",
            "noi_gui": "",
            "noi_nhan": "",
        },
    }


class LayoutInputResilienceTests(unittest.TestCase):
    def test_missing_image_and_missing_boxes_fail_closed_to_rules(self) -> None:
        with patch.object(layout, "LAYOUTLMV3_MODEL_DIR", str(PROJECT_ROOT)):
            result = layout.layoutlmv3_postprocess(
                PROJECT_ROOT / "does-not-exist.png",
                _document_text(),
                None,
            )

        self.assertEqual(result["mode"], "rule_based_no_boxes")
        self.assertFalse(result["model_ran"])
        self.assertEqual(result["fields"]["loai_van_ban"], "QUYẾT ĐỊNH")

    def test_unreadable_image_with_boxes_returns_error_fallback_not_exception(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForTokenClassification = object
        fake_transformers.AutoProcessor = object
        box = OCRBox(text="QUYẾT ĐỊNH", confidence=0.9, bbox=[10, 10, 100, 30])

        with (
            patch.object(layout, "LAYOUTLMV3_MODEL_DIR", str(PROJECT_ROOT)),
            patch.object(layout, "LAYOUTLMV3_MODEL_NAME", ""),
            patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": fake_transformers},
            ),
        ):
            result = layout.layoutlmv3_postprocess(
                PROJECT_ROOT / "does-not-exist.png",
                _document_text(),
                [box],
            )

        self.assertEqual(result["mode"], "rule_based_layoutlmv3_error")
        self.assertFalse(result["model_ran"])
        self.assertEqual(result["fields"]["loai_van_ban"], "QUYẾT ĐỊNH")

    def test_layout_boxes_drop_only_malformed_and_non_first_page_entries(self) -> None:
        row = {
            "boxes": [
                {"page": 1, "text": "valid", "confidence": 0.95, "bbox": [1, 2, 30, 40]},
                {"page": 2, "text": "page two", "confidence": 0.9, "bbox": [1, 2, 30, 40]},
                {"page": "first", "text": "bad page", "bbox": [1, 2, 30, 40]},
                {"page": 1, "text": "short bbox", "bbox": [1, 2, 30]},
                {"page": 1, "text": "string bbox", "bbox": ["x", 2, 30, 40]},
                {"page": 1, "text": "zero area", "bbox": [1, 2, 1, 40]},
                {"page": 1, "text": "", "bbox": [1, 2, 30, 40]},
                None,
                "not-a-box",
            ]
        }

        boxes = main._layout_boxes_from_row(row)

        self.assertEqual([box.text for box in boxes], ["valid"])
        self.assertEqual(boxes[0].bbox, [1, 2, 30, 40])

    def test_normalize_bbox_clamps_coordinates_to_layoutlm_range(self) -> None:
        self.assertEqual(
            layout._normalize_bbox([-10, -20, 110, 220], width=100, height=200),
            [0, 0, 1000, 1000],
        )


class LayoutOverflowInferenceTests(unittest.TestCase):
    def test_overlapping_windows_cover_all_words_and_keep_highest_confidence(self) -> None:
        words = [f"w{index}" for index in range(5)]
        windows = [
            (
                [None, 0, 1, 2, None],
                [0, 0, 1, 1, 0],
                [0.9, 0.7, 0.6, 0.4, 0.9],
            ),
            (
                [None, 2, 3, 4, None],
                [0, 2, 2, 0, 0],
                [0.9, 0.95, 0.8, 0.9, 0.9],
            ),
        ]

        selected, labels, confidences, diagnostics = layout._merge_overflow_word_predictions(
            words,
            windows,
            {0: "O", 1: "B-trich_yeu", 2: "I-trich_yeu"},
            max_length=5,
            stride=2,
        )

        self.assertEqual(selected, words)
        self.assertEqual(labels, ["O", "B-trich_yeu", "I-trich_yeu", "I-trich_yeu", "O"])
        self.assertEqual(confidences[2], 0.95)
        self.assertEqual(diagnostics["covered_word_count"], 5)
        self.assertEqual(diagnostics["tokenizer_truncated_word_count"], 0)
        self.assertEqual(diagnostics["inference_window_count"], 2)
        self.assertEqual(diagnostics["overlap_word_count"], 1)
        self.assertEqual(diagnostics["overlap_prediction_replacement_count"], 1)

    def test_transformers_runtime_executes_overflow_windows_sequentially(self) -> None:
        import torch

        class FakeEncoding(dict):
            def __init__(self) -> None:
                super().__init__(
                    input_ids=torch.zeros((2, 5), dtype=torch.long),
                    attention_mask=torch.ones((2, 5), dtype=torch.long),
                    bbox=torch.zeros((2, 5, 4), dtype=torch.long),
                    pixel_values=[
                        torch.zeros((3, 8, 8), dtype=torch.float32),
                        torch.ones((3, 8, 8), dtype=torch.float32),
                    ],
                    overflow_to_sample_mapping=torch.tensor([0, 0], dtype=torch.long),
                )
                self._word_ids = [
                    [None, 0, 1, 2, None],
                    [None, 2, 3, 4, None],
                ]

            def word_ids(self, batch_index: int) -> list[int | None]:
                return self._word_ids[batch_index]

        class FakeProcessor:
            def __init__(self) -> None:
                self.kwargs: dict = {}

            def __call__(self, *_args, **kwargs):
                self.kwargs = kwargs
                return FakeEncoding()

        class FakeModel:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    id2label={0: "O", 1: "B-trich_yeu", 2: "I-trich_yeu"}
                )
                self.calls: list[dict[str, tuple[int, ...]]] = []

            def __call__(self, **inputs):
                self.calls.append(
                    {
                        key: tuple(value.shape)
                        for key, value in inputs.items()
                        if hasattr(value, "shape")
                    }
                )
                logits = torch.zeros((1, 5, 3), dtype=torch.float32)
                if len(self.calls) == 1:
                    logits[0, 1, 0] = 4.0
                    logits[0, 2, 1] = 4.0
                    logits[0, 3, 1] = 1.0
                else:
                    logits[0, 1, 2] = 6.0
                    logits[0, 2, 2] = 4.0
                    logits[0, 3, 0] = 4.0
                return types.SimpleNamespace(logits=logits)

        processor = FakeProcessor()
        model = FakeModel()
        boxes = [
            OCRBox(
                text="w0 w1 w2 w3 w4",
                confidence=0.9,
                bbox=[5, 5, 95, 25],
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            with (
                patch.object(layout, "LAYOUTLMV3_MODEL_DIR", temp_dir),
                patch.object(layout, "LAYOUTLMV3_MODEL_NAME", ""),
                patch.object(layout, "LAYOUTLMV3_PROCESSOR_NAME", "fake-processor"),
                patch.object(layout, "_load_transformers_runtime", return_value=(processor, model)),
            ):
                result = layout.layoutlmv3_postprocess(image_path, _document_text(), boxes)

        self.assertTrue(result["model_ran"])
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(processor.kwargs["return_overflowing_tokens"])
        self.assertEqual(processor.kwargs["stride"], 64)
        self.assertEqual(processor.kwargs["padding"], "max_length")
        for call in model.calls:
            self.assertNotIn("overflow_to_sample_mapping", call)
            self.assertEqual(call["input_ids"][0], 1)
            self.assertEqual(call["pixel_values"], (1, 3, 8, 8))
        diagnostics = result["prediction_diagnostics"]
        self.assertEqual(diagnostics["covered_word_count"], 5)
        self.assertEqual(diagnostics["tokenizer_truncated_word_count"], 0)
        self.assertEqual(diagnostics["inference_window_count"], 2)
        preview = {item["token"]: item["label"] for item in result["labels_preview_rows"]}
        self.assertEqual(preview["w2"], "I-trich_yeu")


class LayoutModelSchemaTests(unittest.TestCase):
    def test_unknown_model_labels_fail_closed_and_keep_rule_fields(self) -> None:
        text = _document_text(doc_type="NGHỊ ĐỊNH")
        result = layout._model_result(
            text=text,
            model_ref="bad-label-model",
            runtime="unit-test",
            processor_ref="unit-test",
            selected_words=["THÔNG", "BÁO", "Điều", "1"],
            labels_by_word=["LABEL_0", "LABEL_1", "B-random", "I-random"],
            id2label={0: "LABEL_0", 1: "LABEL_1", 2: "B-random", 3: "I-random"},
        )

        self.assertEqual(result["mode"], "layoutlmv3_model_incompatible_labels")
        self.assertFalse(result["field_label_compatible"])
        self.assertEqual(result["model_fields"], {})
        self.assertEqual(result["fields_source"], "rule_based")
        self.assertEqual(result["fields"]["loai_van_ban"], "NGHỊ ĐỊNH")

    def test_conflicting_compatible_model_label_cannot_replace_rule_doc_type(self) -> None:
        text = _document_text(doc_type="NGHỊ ĐỊNH")
        result = layout._model_result(
            text=text,
            model_ref="compatible-but-wrong-model",
            runtime="unit-test",
            processor_ref="unit-test",
            selected_words=["THÔNG BÁO"],
            labels_by_word=["B-document_type"],
            id2label={0: "O", 1: "B-document_type"},
        )

        self.assertTrue(result["field_label_compatible"])
        self.assertEqual(result["model_fields"]["loai_van_ban"], "THÔNG BÁO")
        self.assertEqual(result["fields"]["loai_van_ban"], "NGHỊ ĐỊNH")
        self.assertIn("loai_van_ban", result["rejected_model_fields"])

    def test_rule_fallback_supports_all_configured_document_types(self) -> None:
        document_types = [
            "THÔNG TƯ",
            "CÔNG VĂN",
            "QUYẾT ĐỊNH",
            "NGHỊ ĐỊNH",
            "THÔNG BÁO",
            "KẾ HOẠCH",
            "TỜ TRÌNH",
            "GIẤY MỜI",
            "BÁO CÁO",
        ]
        for document_type in document_types:
            with self.subTest(document_type=document_type):
                result = layout._model_result(
                    text=_document_text(doc_type=document_type),
                    model_ref="generic-label-model",
                    runtime="unit-test",
                    processor_ref="unit-test",
                    selected_words=["noise"],
                    labels_by_word=["LABEL_0"],
                    id2label={0: "LABEL_0"},
                )
                self.assertEqual(result["fields"]["loai_van_ban"], document_type)


class LayoutMultiPageAndCandidateTests(unittest.TestCase):
    def test_multi_page_normative_document_keeps_page_one_header_and_last_page_recipient(self) -> None:
        first_page = _document_text(doc_type="QUYẾT ĐỊNH")
        full_text = (
            first_page
            + "\nĐiều 1. Phê duyệt chương trình.\n"
            + "Điều 2. Quyết định này có hiệu lực kể từ ngày ký.\n"
            + "Nơi nhận:\n- Sở Nội vụ;\n- Lưu: VT."
        )
        candidate = layout._candidate_fields_from_row(
            _row("easyocr", full_text, first_page_text=first_page)
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["so_ky_hieu"], "12/2026/QĐ-UBND")
        self.assertEqual(candidate["loai_van_ban"], "QUYẾT ĐỊNH")
        self.assertEqual(candidate["noi_nhan"], "Sở Nội vụ")
        self.assertNotIn("Điều 1", candidate["trich_yeu"])

    def test_malformed_candidate_rows_are_ignored_without_losing_valid_row(self) -> None:
        valid = _row("tesseract", _document_text(doc_type="THÔNG BÁO"))
        malformed_rows = [
            None,
            "not-a-row",
            {"engine": "easyocr", "status": "ok", "text": "x", "raw": "not-a-dict"},
            {"engine": "easyocr", "status": "ok", "text": 123, "raw": {}},
            {"engine": "glm_ocr", "status": "ok", "text": _document_text()},
            {"engine": "easyocr", "status": "error", "text": _document_text()},
        ]

        stabilized = layout.stabilize_layout_fields_from_rows(
            _layout_result(),
            [*malformed_rows, valid],
        )

        self.assertEqual(stabilized["fields"]["loai_van_ban"], "THÔNG BÁO")
        self.assertEqual(stabilized["fields"]["so_ky_hieu"], "12/2026/QĐ-UBND")

    def test_two_independent_ocr_engines_outvote_one_conflicting_date(self) -> None:
        correct = _document_text(issued_date=(14, 7, 2026))
        conflicting = _document_text(issued_date=(1, 1, 2025))
        rows = [
            _row("easyocr", correct),
            _row("tesseract", correct),
            _row("paddle_vietocr", conflicting),
        ]

        stabilized = layout.stabilize_layout_fields_from_rows(_layout_result(), rows)

        self.assertEqual(stabilized["fields"]["ngay_ban_hanh"], "14/07/2026")

    def test_two_independent_ocr_engines_outvote_one_plausible_wrong_subject(self) -> None:
        correct_subject = "Về việc phê duyệt chương trình chuyển đổi số năm 2026"
        wrong_subject = "Về việc điều chỉnh kế hoạch kiểm tra và tổng hợp báo cáo định kỳ năm 2026"
        rows = [
            _row("tesseract", _document_text(subject=correct_subject)),
            _row("paddle_vietocr", _document_text(subject=correct_subject)),
            _row("easyocr", _document_text(subject=wrong_subject)),
        ]

        stabilized = layout.stabilize_layout_fields_from_rows(_layout_result(), rows)

        self.assertEqual(stabilized["fields"]["trich_yeu"], correct_subject)


class LayoutFingerprintRefreshTests(unittest.TestCase):
    @staticmethod
    def _page_boxes(page_two_text: str = "page two") -> list[dict]:
        return [
            {"page": 1, "text": "header", "confidence": 0.9, "bbox": [1, 2, 30, 40]},
            {"page": 2, "text": page_two_text, "confidence": 0.8, "bbox": [1, 50, 30, 80]},
        ]

    def test_fingerprint_uses_only_page_one_and_is_stable_when_later_pages_change(self) -> None:
        row_before = _row("easyocr", _document_text(), boxes=self._page_boxes("old page two"))
        row_after = copy.deepcopy(row_before)
        row_after["boxes"][1]["text"] = "new page two"

        self.assertEqual(
            main._layout_input_fingerprint(row_before),
            main._layout_input_fingerprint(row_after),
        )

    def test_fingerprint_tolerates_malformed_raw_and_box_payloads(self) -> None:
        row = _row("easyocr", _document_text())
        row["raw"] = "not-a-dict"
        row["boxes"] = [
            None,
            "bad",
            {"page": "first", "text": "bad page", "bbox": [1, 2, 3, 4]},
            {"page": 1, "text": "valid", "confidence": 0.9, "bbox": [1, 2, 30, 40]},
        ]

        fingerprint = main._layout_input_fingerprint(row)

        self.assertRegex(fingerprint, re.compile(r"^[0-9a-f]{64}$"))

    def test_changed_page_one_fingerprint_forces_refresh_but_page_two_does_not(self) -> None:
        original = _row("easyocr", _document_text(), boxes=self._page_boxes("old"))
        layout_result = _layout_result()
        layout_result["ocr_source"] = {
            "engine": "easyocr",
            "variant": "raw",
            "input_fingerprint": main._layout_input_fingerprint(original),
        }
        report = {
            "input_mode": "scan",
            "results": [copy.deepcopy(original)],
            "layoutlmv3_postprocess": layout_result,
        }

        page_two_changed = copy.deepcopy(report)
        page_two_changed["results"][0]["boxes"][1]["text"] = "changed page two"
        self.assertFalse(main._layout_postprocess_needs_refresh(page_two_changed))

        page_one_changed = copy.deepcopy(report)
        page_one_changed["results"][0]["raw"]["first_page_text"] += " corrected"
        self.assertTrue(main._layout_postprocess_needs_refresh(page_one_changed))

    def test_corrupt_cached_layout_metadata_is_treated_as_stale_not_as_fatal(self) -> None:
        row = _row("easyocr", _document_text())
        reports = [
            {
                "input_mode": "scan",
                "results": [row],
                "layoutlmv3_postprocess": "not-a-dict",
            },
            {
                "input_mode": "scan",
                "results": [row],
                "layoutlmv3_postprocess": {
                    "postprocess_version": layout.POSTPROCESS_VERSION,
                    "ocr_source": "not-a-dict",
                    "fields": {},
                },
            },
        ]

        for report in reports:
            with self.subTest(layout_payload=report["layoutlmv3_postprocess"]):
                self.assertTrue(main._layout_postprocess_needs_refresh(report))

    def test_malformed_result_rows_do_not_crash_refresh_check(self) -> None:
        report = {
            "input_mode": "scan",
            "results": [None, "not-a-row", {}, {"status": "error"}],
            "layoutlmv3_postprocess": _layout_result(),
        }

        self.assertFalse(main._layout_postprocess_needs_refresh(report))


if __name__ == "__main__":
    unittest.main()
