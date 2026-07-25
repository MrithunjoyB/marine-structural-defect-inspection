from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from structvision import (
    CLASSICAL_METHOD,
    DEFAULT_METHOD,
    HYBRID_METHOD,
    METHOD_STATUSES,
    PATCHCORE_METHOD,
    DemonstrationInputError,
    analyse_demonstration_image,
    analysis_json_bytes,
    annotated_png_bytes,
    binary_mask_png_bytes,
    candidate_crop,
    candidate_mask,
    candidate_rows,
    decode_image_bytes,
    demonstration_fixture,
    method_availability,
    proposal_csv_bytes,
    render_overlay,
)


def encoded_image(mode: str, image_format: str, suffix: str = "") -> tuple[bytes, str]:
    if mode == "L":
        array = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    elif mode == "RGBA":
        array = np.zeros((24, 32, 4), dtype=np.uint8)
        array[..., :3] = (220, 80, 20)
        array[..., 3] = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]
    else:
        array = np.zeros((24, 32, 3), dtype=np.uint8)
        array[..., 0] = 30
        array[..., 1] = 90
        array[..., 2] = 180
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    extension = suffix or {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tiff"}[image_format]
    return buffer.getvalue(), f"sample{extension}"


class DemonstrationInputTests(unittest.TestCase):
    def test_png_jpeg_and_tiff_decode_in_memory(self):
        for image_format, mode in (("PNG", "RGB"), ("JPEG", "RGB"), ("TIFF", "L")):
            with self.subTest(image_format=image_format):
                payload, filename = encoded_image(mode, image_format)
                decoded = decode_image_bytes(payload, filename=filename)
                self.assertEqual(decoded.image_bgr.shape, (24, 32, 3))
                self.assertEqual(decoded.source_format, image_format)

    def test_grayscale_rgb_and_rgba_contract(self):
        gray, gray_name = encoded_image("L", "PNG")
        rgb, rgb_name = encoded_image("RGB", "PNG")
        rgba, rgba_name = encoded_image("RGBA", "PNG")
        self.assertIn("grayscale", decode_image_bytes(gray, filename=gray_name).colour_handling)
        self.assertIn("RGB", decode_image_bytes(rgb, filename=rgb_name).colour_handling)
        with self.assertRaisesRegex(DemonstrationInputError, "explicit alpha handling"):
            decode_image_bytes(rgba, filename=rgba_name)
        decoded = decode_image_bytes(
            rgba,
            filename=rgba_name,
            alpha_handling="composite_white",
        )
        self.assertIn("composite_white", decoded.colour_handling)

    def test_malformed_and_type_mismatch_rejected(self):
        with self.assertRaisesRegex(DemonstrationInputError, "malformed"):
            decode_image_bytes(b"not an image", filename="bad.png")
        payload, _ = encoded_image("RGB", "PNG")
        with self.assertRaisesRegex(DemonstrationInputError, "does not match"):
            decode_image_bytes(payload, filename="wrong.jpg")

    def test_encoded_and_decoded_limits_are_fail_closed(self):
        payload, filename = encoded_image("RGB", "PNG")
        with self.assertRaisesRegex(DemonstrationInputError, "Encoded image exceeds"):
            decode_image_bytes(payload, filename=filename, max_encoded_bytes=4)
        with self.assertRaisesRegex(DemonstrationInputError, "Decoded image exceeds"):
            decode_image_bytes(payload, filename=filename, max_decoded_pixels=100)


class DemonstrationExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoded = demonstration_fixture("thin structural indication")
        cls.analysis = analyse_demonstration_image(cls.decoded)

    def test_default_and_status_policy(self):
        self.assertEqual(DEFAULT_METHOD, CLASSICAL_METHOD)
        by_id = {item.method_id: item for item in METHOD_STATUSES}
        self.assertTrue(by_id[CLASSICAL_METHOD].recommended_default)
        self.assertEqual(by_id[PATCHCORE_METHOD].status, "protected development baseline")
        self.assertEqual(by_id[HYBRID_METHOD].status, "rejected development candidate")
        self.assertTrue(method_availability(CLASSICAL_METHOD).available)
        self.assertFalse(method_availability(PATCHCORE_METHOD).available)

    def test_classical_result_preserves_method_and_coordinate_identity(self):
        self.assertEqual(self.analysis.method_id, CLASSICAL_METHOD)
        mapping = dict(self.analysis.coordinate_mapping)
        self.assertEqual(mapping["analysed_width"], self.analysis.image_shape[1])
        self.assertEqual(mapping["analysed_height"], self.analysis.image_shape[0])
        self.assertEqual(
            render_overlay(self.analysis).shape,
            self.analysis.image_shape,
        )

    def test_candidate_geometry_and_render_alignment(self):
        rows = candidate_rows(self.analysis)
        self.assertGreater(len(rows), 0)
        ranks = [row["rank"] for row in rows]
        self.assertEqual(ranks, list(range(1, len(rows) + 1)))
        for row in rows:
            mask = candidate_mask(self.analysis, str(row["proposal_id"]))
            ys, xs = np.where(mask > 0)
            self.assertEqual(
                (row["x_min"], row["y_min"], row["x_max"], row["y_max"]),
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
            crop = candidate_crop(self.analysis, str(row["proposal_id"]))
            self.assertEqual(
                crop.shape[:2],
                (row["y_max"] - row["y_min"], row["x_max"] - row["x_min"]),
            )

    def test_exports_have_explicit_semantics_and_no_absolute_paths(self):
        payload = json.loads(analysis_json_bytes(self.analysis))
        self.assertEqual(payload["schema_version"], "structvision-professor-analysis-v1")
        self.assertEqual(payload["method"]["method_id"], CLASSICAL_METHOD)
        self.assertFalse(payload["input"]["absolute_path_recorded"])
        self.assertEqual(
            payload["analysis"]["bounding_box_convention"],
            "half-open:x_min,y_min,x_max,y_max",
        )
        serialised = json.dumps(payload)
        self.assertNotIn(str(Path.cwd()), serialised)
        self.assertNotIn("probability_score", serialised)
        csv_text = proposal_csv_bytes(self.analysis).decode("utf-8")
        self.assertIn("configuration_hash", csv_text)
        self.assertIn("N/A", csv_text)

    def test_png_exports_use_exact_dimensions_and_binary_masks(self):
        overlay = cv2.imdecode(
            np.frombuffer(annotated_png_bytes(self.analysis), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(overlay.shape, self.analysis.image_shape)
        first = str(candidate_rows(self.analysis)[0]["proposal_id"])
        mask = cv2.imdecode(
            np.frombuffer(binary_mask_png_bytes(self.analysis, first), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertEqual(mask.shape, self.analysis.image_shape[:2])
        self.assertTrue(np.all((mask == 0) | (mask == 255)))

    def test_analysis_path_creates_no_caller_files(self):
        with tempfile.TemporaryDirectory() as directory:
            before = set(Path(directory).iterdir())
            original = Path.cwd()
            try:
                import os

                os.chdir(directory)
                analyse_demonstration_image(demonstration_fixture("clean textured surface"))
            finally:
                os.chdir(original)
            self.assertEqual(before, set(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
