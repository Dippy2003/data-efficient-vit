"""Fast unit tests for study aggregation and validation."""

from argparse import Namespace
import unittest

from src.run_study import aggregate_results, validate_args


class StudyTests(unittest.TestCase):
    def test_aggregate_results_calculates_mean(self):
        rows = [
            {"model": "cnn", "fraction": 0.1, "accuracy": 0.4, "macro_f1": 0.3},
            {"model": "cnn", "fraction": 0.1, "accuracy": 0.6, "macro_f1": 0.5},
        ]
        summary = aggregate_results(rows)[0]
        self.assertAlmostEqual(summary["accuracy_mean"], 0.5)
        self.assertEqual(summary["runs"], 2)

    def test_invalid_fraction_is_rejected(self):
        args = Namespace(fractions=[0], seeds=[42], epochs=1, batch_size=1, patience=0)
        with self.assertRaises(SystemExit):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
