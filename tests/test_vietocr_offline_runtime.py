from __future__ import annotations

import unittest
from pathlib import Path

from vietocr.tool.config import Cfg

from app.config import OCR_TEMP_DIR, VIETOCR_CONFIG_PATH, VIETOCR_WEIGHTS_PATH
from app.services.tempdirs import workspace_temporary_directory


class VietOCROfflineRuntimeTests(unittest.TestCase):
    def test_local_config_and_checkpoint_are_self_contained(self) -> None:
        config_path = Path(VIETOCR_CONFIG_PATH)
        weights_path = Path(VIETOCR_WEIGHTS_PATH)

        self.assertTrue(config_path.is_file())
        self.assertTrue(weights_path.is_file())
        self.assertGreater(weights_path.stat().st_size, 100_000_000)
        config = Cfg.load_config_from_file(str(config_path))
        self.assertFalse(config["cnn"]["pretrained"])
        self.assertEqual(config["dataset"]["image_max_width"], 512)
        self.assertFalse(str(config["weights"]).startswith("http"))

    def test_worker_temp_directory_is_writable_and_cleaned(self) -> None:
        created: Path | None = None
        with workspace_temporary_directory("runtime_test_") as raw_path:
            created = Path(raw_path)
            self.assertEqual(created.parent.resolve(), OCR_TEMP_DIR.resolve())
            marker = created / "marker.txt"
            marker.write_text("ok", encoding="utf-8")
            self.assertEqual(marker.read_text(encoding="utf-8"), "ok")

        self.assertIsNotNone(created)
        self.assertFalse(created.exists())


if __name__ == "__main__":
    unittest.main()
