from __future__ import annotations

import unittest

from app.ocr_engines.base import OCRBox
from app.services.reading_order import order_boxes_xy_cut


class ReadingOrderTests(unittest.TestCase):
    @staticmethod
    def box(
        text: str,
        bbox: list[int] | None,
        confidence: float | None = 0.9,
    ) -> OCRBox:
        return OCRBox(text=text, bbox=bbox, confidence=confidence)

    def test_header_uses_columns_inside_independent_horizontal_zones(self) -> None:
        boxes = [
            self.box("authority-1", [100, 80, 400, 112]),
            self.box("motto-1", [600, 80, 900, 112]),
            self.box("authority-2", [110, 108, 390, 140]),
            self.box("motto-2", [610, 108, 890, 140]),
            self.box("document-number", [100, 200, 400, 232]),
            self.box("document-date", [600, 200, 900, 232]),
            self.box("body-1", [100, 310, 900, 345]),
            self.box("body-2", [100, 343, 900, 378]),
        ]

        ordered, diagnostics = order_boxes_xy_cut(boxes, page_width=1000, page_height=1400)

        self.assertEqual(
            [box.text for box in ordered],
            [
                "authority-1",
                "authority-2",
                "motto-1",
                "motto-2",
                "document-number",
                "document-date",
                "body-1",
                "body-2",
            ],
        )
        self.assertTrue(diagnostics["applied"])
        self.assertGreaterEqual(diagnostics["horizontal_cuts"], 2)
        self.assertGreaterEqual(diagnostics["vertical_cuts"], 2)

    def test_footer_reads_recipient_column_before_signature_column(self) -> None:
        boxes = [
            self.box("body-1", [80, 200, 920, 235]),
            self.box("body-2", [80, 233, 920, 268]),
            self.box("recipients", [80, 800, 430, 832]),
            self.box("minister", [650, 800, 920, 832]),
            self.box("recipient-1", [80, 830, 430, 862]),
            self.box("deputy", [670, 830, 900, 862]),
            # A weak seal/signature detection spans the whitespace channel. It
            # must not prevent column discovery and must not be dropped.
            self.box("seal-artifact", [450, 840, 700, 1000], confidence=0.2),
            self.box("recipient-2", [80, 860, 500, 892]),
            self.box("signer-name", [680, 890, 920, 922]),
        ]

        ordered, diagnostics = order_boxes_xy_cut(boxes, page_width=1000, page_height=1400)

        self.assertEqual(
            [box.text for box in ordered],
            [
                "body-1",
                "body-2",
                "recipients",
                "recipient-1",
                "recipient-2",
                "minister",
                "deputy",
                "seal-artifact",
                "signer-name",
            ],
        )
        self.assertEqual(len(ordered), len(boxes))
        self.assertEqual(diagnostics["vertical_cuts"], 1)

    def test_one_column_connected_body_keeps_detector_order(self) -> None:
        boxes = [
            self.box("first", [100, 100, 900, 140]),
            self.box("second", [110, 137, 890, 177]),
            self.box("third", [90, 174, 920, 214]),
        ]

        ordered, diagnostics = order_boxes_xy_cut(boxes, page_width=1000, page_height=1400)

        self.assertEqual(ordered, boxes)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["vertical_cuts"], 0)

    def test_invalid_geometry_stays_anchored_and_no_box_is_lost(self) -> None:
        right = self.box("right", [650, 100, 900, 135])
        invalid = self.box("no-geometry", None)
        left = self.box("left", [100, 100, 400, 135])

        ordered, diagnostics = order_boxes_xy_cut(
            [right, invalid, left],
            page_width=1000,
            page_height=1400,
        )

        self.assertEqual([box.text for box in ordered], ["left", "no-geometry", "right"])
        self.assertIs(ordered[1], invalid)
        self.assertEqual(diagnostics["invalid_geometry_boxes"], 1)
        self.assertEqual(len(ordered), 3)


if __name__ == "__main__":
    unittest.main()
