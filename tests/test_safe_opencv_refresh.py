from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts import refresh_opencv_safe_preprocessing as refresh


def _row(engine: str, variant: str, text: str, **extra) -> dict:
    row = {
        "engine": engine,
        "variant": variant,
        "status": "ok",
        "text": text,
        "boxes": [],
        "elapsed_sec": 1.0,
        "raw": {"marker": f"{engine}-{variant}-old"},
    }
    row.update(extra)
    return row


class SafeOpenCVRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_temp = Path(__file__).resolve().parents[1] / "tmp_preprocess_test"
        workspace_temp.mkdir(parents=True, exist_ok=True)
        self.root = workspace_temp / f"safe_opencv_refresh_{uuid.uuid4().hex}"
        self.jobs = self.root / "jobs"
        self.job_dir = self.jobs / "job-1"
        self.images = self.job_dir / "images"
        self.ground_truth = self.job_dir / "ground_truth" / "truth.txt"
        self.images.mkdir(parents=True)
        self.ground_truth.parent.mkdir(parents=True)
        (self.images / "page1.png").write_bytes(b"raw-image")
        self.ground_truth.write_text("abc", encoding="utf-8")
        self.jobs_patch = patch.object(refresh, "JOBS_DIR", self.jobs)
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_report(self, opencv_text: str) -> dict:
        old_opencv = _row(
            "paddleocr_vl",
            "opencv_preprocessed",
            opencv_text,
            custom_payload={"must": "survive"},
        )
        report = {
            "job_id": "job-1",
            "raw_images": ["images/page1.png"],
            "raw_image": "images/page1.png",
            "ground_truth_file": {"relative_path": "ground_truth/truth.txt"},
            "results": [
                _row("paddleocr_vl", "raw", "abc"),
                old_opencv,
                _row("glm_ocr", "raw", "abc"),
            ],
        }
        (self.job_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False),
            encoding="utf-8",
        )
        return old_opencv

    @staticmethod
    def _fake_preprocess(raw_page: Path, image_dir: Path) -> tuple[Path, list[str]]:
        return image_dir / f"{raw_page.stem}_opencv_preprocessed.png", ["test_preprocess"]

    def _run(self, candidate: dict) -> dict:
        with (
            patch.object(refresh, "preprocess_image", side_effect=self._fake_preprocess),
            patch.object(refresh, "_run_engine_on_pages", return_value=candidate),
            patch.object(refresh, "_refresh_layout_postprocess", return_value=False),
            patch.object(refresh, "_refresh_document_archive", return_value=False),
        ):
            return refresh.refresh_job(
                "job-1",
                engines={"paddleocr_vl"},
                refresh_layout=False,
            )

    def test_better_candidate_replaces_row_and_preserves_complete_history(self) -> None:
        old_opencv = self._write_report("axc")
        candidate = _row("paddleocr_vl", "opencv_preprocessed", "abc")

        result = self._run(candidate)
        saved = json.loads((self.job_dir / "report.json").read_text(encoding="utf-8"))
        saved_opencv = next(
            row
            for row in saved["results"]
            if row["engine"] == "paddleocr_vl" and row["variant"] == "opencv_preprocessed"
        )

        self.assertEqual(result["engines"], ["paddleocr_vl"])
        self.assertTrue(result["attempts"][0]["accepted"])
        self.assertEqual(saved_opencv["text"], "abc")
        self.assertEqual(saved_opencv["cer"], 0.0)
        self.assertEqual(saved["result_history"], [old_opencv])
        self.assertEqual(
            saved["result_history_metadata"][0]["reason"],
            "replaced_by_opencv_safe_preprocessing_refresh",
        )
        self.assertEqual(
            saved["result_history_metadata"][0]["replacement_mode"],
            "opencv_safe_preprocessing_refresh",
        )
        self.assertNotIn(
            "glm_ocr",
            {row["engine"] for row in saved["comparison_summary"]["rows"]},
        )
        self.assertNotEqual(saved["best_engine"]["engine"], "glm_ocr")

    def test_worse_candidate_is_rejected_without_losing_old_row(self) -> None:
        old_opencv = self._write_report("abc")
        candidate = _row("paddleocr_vl", "opencv_preprocessed", "zzz")

        result = self._run(candidate)
        saved = json.loads((self.job_dir / "report.json").read_text(encoding="utf-8"))
        saved_opencv = next(
            row
            for row in saved["results"]
            if row["engine"] == "paddleocr_vl" and row["variant"] == "opencv_preprocessed"
        )

        self.assertEqual(result["engines"], [])
        self.assertFalse(result["attempts"][0]["accepted"])
        self.assertEqual(saved_opencv["text"], old_opencv["text"])
        self.assertEqual(saved_opencv["custom_payload"], {"must": "survive"})
        self.assertNotIn("result_history", saved)
        self.assertFalse(saved["refresh_history"][-1]["rows"][0]["accepted"])

    def test_failed_candidate_is_rejected_without_replacing_old_row(self) -> None:
        old_opencv = self._write_report("abc")
        candidate = {
            "engine": "paddleocr_vl",
            "variant": "opencv_preprocessed",
            "status": "error",
            "text": "",
            "error": "worker failed",
        }

        result = self._run(candidate)
        saved = json.loads((self.job_dir / "report.json").read_text(encoding="utf-8"))
        saved_opencv = next(
            row
            for row in saved["results"]
            if row["engine"] == "paddleocr_vl" and row["variant"] == "opencv_preprocessed"
        )

        self.assertEqual(result["engines"], [])
        self.assertFalse(result["attempts"][0]["accepted"])
        self.assertEqual(saved_opencv, old_opencv | {"cer": 0.0, "wer": 0.0, "ground_truth_compare_length": 3, "ground_truth_metric_version": 2})
        self.assertNotIn("result_history", saved)


if __name__ == "__main__":
    unittest.main()
