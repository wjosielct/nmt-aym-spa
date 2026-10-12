import math
import unittest

from src.benchmarking.inference_benchmark import (
    count_generated_tokens,
    model_tag,
    read_benchmark_protocol,
    summarize_runs,
)


class InferenceBenchmarkUnitTests(unittest.TestCase):
    def _config(self):
        return {
            "base_test_max_len": 150,
            "ft_test_max_len": 150,
            "inference_benchmark": {
                "conditions": [
                    {"stage": "base", "domain": "biblical"},
                    {"stage": "finetuned", "domain": "conversational"},
                ],
                "batch_size": 1,
                "warmup_examples": 10,
                "repetitions": 1,
                "max_len": 150,
                "precision": "fp32",
                "cuda_device_index": 0,
                "require_cuda": True,
                "output_subdir": "inference_benchmark",
                "output_prefix": "inference_benchmark",
            },
        }

    def test_model_tag_matches_pipeline_convention(self):
        self.assertEqual(
            model_tag("aym", "spa", 4, 4, 111),
            "aym_to_spa_Enc4_Dec4_Seed111",
        )

    def test_generated_token_count_excludes_bos_and_pad_and_includes_eos(self):
        # BOS=1, EOS=2, PAD=0. The EOS token is counted; trailing PAD is not.
        self.assertEqual(
            count_generated_tokens([1, 10, 11, 0, 12, 2, 0], bos_id=1, eos_id=2, pad_id=0),
            4,
        )

    def test_protocol_is_fixed_to_two_prespecified_conditions(self):
        protocol = read_benchmark_protocol(self._config())
        self.assertEqual(protocol["batch_size"], 1)
        self.assertEqual(protocol["warmup_examples"], 10)
        self.assertEqual(protocol["max_len"], 150)
        self.assertEqual(protocol["precision"], "fp32")
        self.assertEqual(len(protocol["conditions"]), 2)

    def test_protocol_rejects_non_fp32(self):
        config = self._config()
        config["inference_benchmark"]["precision"] = "fp16"
        with self.assertRaises(ValueError):
            read_benchmark_protocol(config)

    def test_summary_uses_sample_standard_deviation_across_seeds(self):
        rows = []
        for seed, latency, throughput, memory in [
            (111, 10.0, 100.0, 500.0),
            (222, 12.0, 110.0, 502.0),
            (333, 14.0, 120.0, 504.0),
        ]:
            rows.append(
                {
                    "stage": "base",
                    "domain": "biblical",
                    "direction": "aym->spa",
                    "architecture": "4E-4D",
                    "encoder_layers": 4,
                    "decoder_layers": 4,
                    "n_parameters": 9421824,
                    "n_sentences": 6033,
                    "gpu_name": "Test GPU",
                    "seed": seed,
                    "latency_ms_per_sentence": latency,
                    "generated_tokens_per_second": throughput,
                    "peak_gpu_memory_mib": memory,
                }
            )

        summary = summarize_runs(rows, expected_seeds=[111, 222, 333])
        self.assertEqual(len(summary), 1)
        result = summary[0]
        self.assertTrue(math.isclose(result["latency_ms_per_sentence_mean"], 12.0))
        self.assertTrue(math.isclose(result["latency_ms_per_sentence_sd"], 2.0))
        self.assertTrue(math.isclose(result["generated_tokens_per_second_mean"], 110.0))
        self.assertTrue(math.isclose(result["generated_tokens_per_second_sd"], 10.0))
        self.assertTrue(math.isclose(result["peak_gpu_memory_mib_mean"], 502.0))
        self.assertTrue(math.isclose(result["peak_gpu_memory_mib_sd"], 2.0))


if __name__ == "__main__":
    unittest.main()
