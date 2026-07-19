"""Fast unit tests for experiment record helpers."""

import tempfile
import unittest
from pathlib import Path

from src.experiment import make_run_id, save_run_record


class ExperimentTests(unittest.TestCase):
    def test_run_id_contains_config_fingerprint(self):
        first = make_run_id({"seed": 1}).split("-")[-1]
        second = make_run_id({"seed": 2}).split("-")[-1]
        self.assertNotEqual(first, second)

    def test_record_is_written_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_run_record({"run_id": "test", "results": []}, directory)
            self.assertTrue(Path(path).is_file())
            self.assertIn('"run_id": "test"', Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
