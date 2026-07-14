from __future__ import annotations

import json
import shutil
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

from app.services import document_archive as archive


class DocumentArchiveOCRSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_temp = Path(__file__).resolve().parents[1] / "tmp_preprocess_test"
        workspace_temp.mkdir(parents=True, exist_ok=True)
        # tempfile.mkdtemp applies a Windows 0o700 ACL that managed sandbox
        # identities cannot traverse.  A normal inherited workspace directory
        # remains isolated while working in both local and sandboxed runs.
        self.root = workspace_temp / f"archive_sync_test_{uuid.uuid4().hex}"
        self.root.mkdir()
        self.storage = self.root / "storage"
        self.jobs = self.root / "jobs"
        self.storage.mkdir(parents=True)
        self.jobs.mkdir(parents=True)
        self.storage_patch = patch.object(archive, "STORAGE_DIR", self.storage)
        self.jobs_patch = patch.object(archive, "JOBS_DIR", self.jobs)
        self.storage_patch.start()
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        self.storage_patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_archive(self) -> tuple[Path, Path, dict, dict]:
        archive_root = self.storage / "documents" / "2026" / "07" / "doc-1"
        old_row = {
            "engine": "paddle_vietocr",
            "variant": "raw",
            "status": "ok",
            "text": "Nội dung OCR cũ",
            "boxes": [{"text": "Nội dung OCR cũ", "bbox": [1, 2, 100, 20]}],
            "elapsed_sec": 10.0,
            "raw": {"page_count": 1, "model": "old"},
        }
        base_name = "paddle_vietocr__raw"
        text_path = archive_root / "04_ocr_text" / f"{base_name}.txt"
        json_path = archive_root / "05_ocr_json" / f"{base_name}.json"
        word_path = archive_root / "08_word_outputs" / f"{base_name}.docx"
        archive._write_text(text_path, old_row["text"])
        archive._write_json(json_path, old_row)
        word_file = archive._write_editable_ocr_docx(
            word_path,
            "Old OCR",
            old_row,
            old_row["text"],
        )
        output = {
            "engine": old_row["engine"],
            "variant": old_row["variant"],
            "status": old_row["status"],
            "elapsed_sec": old_row["elapsed_sec"],
            "text_length": len(old_row["text"]),
            "text_path": archive._rel(text_path),
            "json_path": archive._rel(json_path),
            "word_path": archive._rel(word_path),
            "word_file": word_file,
            "word_kind": "editable_ocr_layout",
            "word_layout_version": archive.WORD_LAYOUT_VERSION,
            "page_count": 1,
        }
        manifest = {
            "document_id": "doc-1",
            "job_id": "job-1",
            "created_at": "2026-07-12T10:00:00+07:00",
            "source_filename": "scan.pdf",
            "page_count": 1,
            "status": "scanned",
            "storage_root": archive._rel(archive_root),
            "ocr_outputs": [output],
            "successful_ocr_output_count": 1,
            "issue_ocr_output_count": 0,
            "exports": {},
        }
        manifest_path = archive_root / "manifest.json"
        archive._write_json(manifest_path, manifest)
        return archive_root, manifest_path, old_row, manifest

    def test_changed_row_snapshots_old_files_then_updates_all_artifacts(self) -> None:
        archive_root, manifest_path, old_row, _ = self._make_archive()
        new_row = {
            **old_row,
            "text": "Nội dung OCR mới đã qua hybrid guard",
            "boxes": [{"text": "Nội dung OCR mới", "bbox": [1, 2, 100, 20]}],
            "elapsed_sec": 11.5,
            "raw": {"page_count": 1, "model": "new", "hybrid_refinement": {"fallback_to_paddle": 2}},
        }
        report = {"job_id": "job-1", "page_count": 1, "results": [new_row]}

        self.assertTrue(archive.sync_archive_metadata(manifest_path, report))

        base_name = "paddle_vietocr__raw"
        current_text = archive_root / "04_ocr_text" / f"{base_name}.txt"
        current_json = archive_root / "05_ocr_json" / f"{base_name}.json"
        current_word = archive_root / "08_word_outputs" / f"{base_name}.docx"
        self.assertEqual(current_text.read_text(encoding="utf-8"), new_row["text"])
        self.assertEqual(json.loads(current_json.read_text(encoding="utf-8")), new_row)
        self.assertTrue(zipfile.is_zipfile(current_word))
        with zipfile.ZipFile(current_word) as docx:
            document_xml = docx.read("word/document.xml").decode("utf-8")
        self.assertIn("Nội dung OCR mới", document_xml)

        history_versions = list((archive_root / "09_history").glob("*"))
        self.assertEqual(len(history_versions), 1)
        history_row_root = history_versions[0] / base_name
        self.assertEqual(
            (history_row_root / "04_ocr_text" / f"{base_name}.txt").read_text(encoding="utf-8"),
            old_row["text"],
        )
        self.assertEqual(
            json.loads((history_row_root / "05_ocr_json" / f"{base_name}.json").read_text(encoding="utf-8")),
            old_row,
        )
        self.assertTrue(zipfile.is_zipfile(history_row_root / "08_word_outputs" / f"{base_name}.docx"))
        history_manifest = json.loads((history_row_root / "history_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(history_manifest["reason"], "row_changed")
        self.assertEqual(set(history_manifest["artifacts"]), {"04_ocr_text", "05_ocr_json", "08_word_outputs"})

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = manifest["ocr_outputs"][0]
        self.assertEqual(output["text_length"], len(new_row["text"]))
        self.assertEqual(output["elapsed_sec"], 11.5)
        self.assertEqual(len(output["history_versions"]), 1)
        self.assertEqual(len(manifest["ocr_history"]), 1)
        self.assertEqual(output["word_file"]["sha256"], archive._sha256(current_word))

        self.assertFalse(archive.sync_archive_metadata(manifest_path, report))
        self.assertEqual(len(list((archive_root / "09_history").glob("*"))), 1)

    def test_new_report_row_is_added_without_deleting_or_versioning_existing_row(self) -> None:
        archive_root, manifest_path, old_row, _ = self._make_archive()
        new_row = {
            "engine": "tesseract",
            "variant": "raw",
            "status": "ok",
            "text": "Một engine mới",
            "boxes": [],
            "elapsed_sec": 1.2,
            "raw": {"page_count": 1},
        }
        report = {"job_id": "job-1", "page_count": 1, "results": [new_row]}

        self.assertTrue(archive.sync_archive_metadata(manifest_path, report))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = {(row["engine"], row["variant"]) for row in manifest["ocr_outputs"]}
        self.assertEqual(keys, {("paddle_vietocr", "raw"), ("tesseract", "raw")})
        self.assertEqual(
            (archive_root / "04_ocr_text" / "paddle_vietocr__raw.txt").read_text(encoding="utf-8"),
            old_row["text"],
        )
        self.assertEqual(
            (archive_root / "04_ocr_text" / "tesseract__raw.txt").read_text(encoding="utf-8"),
            new_row["text"],
        )
        self.assertFalse((archive_root / "09_history").exists())


if __name__ == "__main__":
    unittest.main()
