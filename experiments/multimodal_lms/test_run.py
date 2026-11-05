"""Check that sampling preserves temporal order and invalid model answers fail."""
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from .run import contact_sheets, parse_answer, sample_frames


class RunnerTests(unittest.TestCase):
    def test_sampling_includes_both_ends_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for index in range(21):
                writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
            inputs, meta = sample_frames(path, 4, 48, 64)
        self.assertEqual(meta["sample_frame_indices"], [0, 2, 5, 8, 10, 12, 15, 18, 20])
        self.assertEqual(len(inputs), 18)
        self.assertEqual(inputs[0]["content"], "t=0.000 s")
        self.assertEqual(inputs[-2]["content"], "t=2.000 s")

    def test_unknown_is_not_converted_to_negative(self):
        response = {"output": [{"type": "message", "content":
            '{"circling":null,"direction":"uncertain","circle_count":null,"confidence":"low"}'}]}
        self.assertIsNone(parse_answer(response)["circling"])

    def test_null_direction_is_preserved_as_uncertain(self):
        response = {"output": [{"type": "message", "content":
            '{"circling":null,"direction":null,"circle_count":null,"confidence":"low"}'}]}
        self.assertEqual(parse_answer(response)["direction"], "uncertain")

    def test_contact_sheets_keep_all_frames(self):
        import base64
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)
        image = {"type": "image", "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}
        inputs = []
        for index in range(5):
            inputs.extend([{"type": "text", "content": f"t={index} s"}, image])
        result = contact_sheets(inputs)
        self.assertEqual(len(result), 3)  # ordering instructions plus two sheets
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result[2]["data_url"].split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (152, 128, 3))
        self.assertGreater(decoded[:28, :64].max(), 100)  # fifth frame has a timestamp
        self.assertLess(decoded[76:, :].max(), 5)  # unused lower tiles remain black

    def test_rejects_inconsistent_negative(self):
        response = {"output": [{"type": "message", "content":
            '{"circling":false,"direction":"clockwise","circle_count":2,"confidence":"high"}'}]}
        with self.assertRaises(ValueError):
            parse_answer(response)

    def test_rejects_truncated_output(self):
        response = {"output": [{"type": "message", "content": '{"circling":true'}]}
        with self.assertRaises(ValueError):
            parse_answer(response)

    def test_accepts_fenced_json_after_prose(self):
        response = {"output": [{"type": "message", "content":
            'Model explanation.\n```json\n{"circling":true,"direction":"clockwise","circle_count":1.5,"confidence":"high"}\n```'}]}
        self.assertEqual(parse_answer(response)["circle_count"], 1.5)


if __name__ == "__main__":
    unittest.main()
