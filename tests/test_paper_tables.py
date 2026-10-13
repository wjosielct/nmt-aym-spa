# ==============================
# ./tests/test_paper_tables.py
# ==============================

import csv
import tempfile
import unittest
from pathlib import Path

from src.analysis.paper_tables import (
    PaperTableError,
    _fmt_delta_ci,
    _load_inference_summary,
    _render_table_4,
    _sample_mean_sd,
    _split_stats,
)


class PaperTablesUnitTests(unittest.TestCase):

    def test_sample_standard_deviation_is_ddof_one(self):
        mean, sd = _sample_mean_sd([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(sd, 1.0)

    def test_delta_ci_formatter_preserves_sign(self):
        positive = _fmt_delta_ci(1.234, 0.5, 1.9)
        negative = _fmt_delta_ci(-1.234, -1.9, -0.5)
        self.assertIn("+1.23", positive)
        self.assertIn("-1.23", negative)

    def test_split_stats_counts_pairs_and_unique_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.tsv"
            path.write_text(
                "dialogue_id\taym\tspa\n"
                "d1\ta\tx\n"
                "d1\tb\ty\n"
                "d2\tc\tz\n",
                encoding="utf-8",
            )
            n_pairs, groups = _split_stats(path, "dialogue_id")
            self.assertEqual(n_pairs, 3)
            self.assertEqual(groups, {"d1", "d2"})

    def test_inference_summary_requires_exact_final_shape(self):
        header = [
            "stage",
            "domain",
            "direction",
            "architecture",
            "encoder_layers",
            "decoder_layers",
            "n_parameters",
            "n_sentences",
            "gpu_name",
            "n_seeds",
            "seeds",
            "latency_ms_per_sentence_mean",
            "latency_ms_per_sentence_sd",
            "generated_tokens_per_second_mean",
            "generated_tokens_per_second_sd",
            "peak_gpu_memory_mib_mean",
            "peak_gpu_memory_mib_sd",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerow(
                    {
                        "stage": "base",
                        "domain": "biblical",
                        "direction": "aym->spa",
                        "architecture": "4E-4D",
                        "encoder_layers": "4",
                        "decoder_layers": "4",
                        "n_parameters": "9421824",
                        "n_sentences": "6033",
                        "gpu_name": "GPU",
                        "n_seeds": "3",
                        "seeds": "111;222;333",
                        "latency_ms_per_sentence_mean": "1.0",
                        "latency_ms_per_sentence_sd": "0.1",
                        "generated_tokens_per_second_mean": "2.0",
                        "generated_tokens_per_second_sd": "0.2",
                        "peak_gpu_memory_mib_mean": "3.0",
                        "peak_gpu_memory_mib_sd": "0.3",
                    }
                )
            with self.assertRaises(PaperTableError):
                _load_inference_summary(path)

    def test_bootstrap_table_caption_marks_selection_and_points_to_full_results(self):
        rows = [
            {
                "family": "adaptation_vs_base",
                "comparison": "FT - Base (4E-4D)",
                "direction": "aym_to_spa",
                "condition": "Biblical",
                "bleu": (1.0, 0.5, 1.5),
                "chrf++": (2.0, 1.0, 3.0),
            }
        ]
        latex = _render_table_4(rows)
        self.assertIn("Selected prespecified paired-bootstrap contrasts", latex)
        self.assertIn("complete set of prespecified bootstrap contrasts", latex)
        self.assertNotIn("Primary paired-bootstrap contrasts.", latex)

    def test_split_stats_requires_grouping_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.tsv"
            path.write_text("other\nvalue\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _split_stats(path, "dialogue_id")


if __name__ == "__main__":
    unittest.main()
