#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import benchmark  # noqa: E402


class BenchmarkParserTest(unittest.TestCase):
    def make_output(self) -> str:
        lines = ["Benchmark:"]
        for index, key in enumerate(benchmark.BENCHMARK_KEYS, start=1):
            lines.append(f"  {key}: {float(index):.3f}")
        lines.append("Benchmark generated tokens: 12 34 56")
        return "\n".join(lines) + "\n"

    def test_parse_complete_output(self) -> None:
        metrics, tokens = benchmark.parse_benchmark(self.make_output())
        self.assertEqual(set(metrics), set(benchmark.BENCHMARK_KEYS))
        self.assertEqual(tokens, (12, 34, 56))
        self.assertEqual(metrics["num_threads"], 1.0)

    def test_missing_metric_is_rejected(self) -> None:
        output = self.make_output().replace("  vision_load_ms: 5.000\n", "")
        with self.assertRaisesRegex(ValueError, "vision_load_ms"):
            benchmark.parse_benchmark(output)

    def test_thread_list_is_deduplicated(self) -> None:
        self.assertEqual(benchmark.resolve_threads("1,2,2,8"), [1, 2, 8])

    def test_negative_thread_count_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            benchmark.resolve_threads("-1")

    def test_output_directory_files(self) -> None:
        metrics = {key: float(index) for index, key in enumerate(benchmark.BENCHMARK_KEYS, start=1)}
        metrics["benchmark_process_wall_ms"] = 123.0
        row = benchmark.BenchmarkRow(
            case="case",
            prompt_mode="spotting",
            image="image.png",
            threads=4,
            metrics=metrics,
            generated_tokens=(12, 34),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            benchmark.write_output_dir(output, [row], "benchmark command")
            self.assertIn("benchmark_process_wall_ms", (output / "cold_start.csv").read_text(encoding="utf-8"))
            self.assertIn("warm_inference_total_ms", (output / "warm_inference.csv").read_text(encoding="utf-8"))
            self.assertTrue((output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
