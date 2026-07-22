#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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
        output = "\n".join(
            line for line in self.make_output().splitlines() if not line.startswith("  vision_load_ms:")
        )
        with self.assertRaisesRegex(ValueError, "vision_load_ms"):
            benchmark.parse_benchmark(output)

    def test_thread_list_is_deduplicated(self) -> None:
        self.assertEqual(benchmark.resolve_threads("1,2,2,8"), [1, 2, 8])

    def test_negative_thread_count_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            benchmark.resolve_threads("-1")

    def test_build_command_adds_selected_vulkan_backends(self) -> None:
        command = benchmark.build_benchmark_command(
            binary=Path("/tmp/hunyuan_ocr_cli"),
            model=Path("/tmp/model"),
            image=Path("/tmp/image.png"),
            prompt_mode="document",
            max_tokens=512,
            threads=16,
            warmup=1,
            repeat=3,
            mmap_weights=True,
            vision_vulkan=True,
            vision_vulkan_device=1,
            text_vulkan=True,
            text_vulkan_device=0,
        )
        self.assertIn("--vision-vulkan", command)
        self.assertEqual(command[command.index("--vision-vulkan-device") + 1], "1")
        self.assertIn("--text-vulkan", command)
        self.assertEqual(command[command.index("--text-vulkan-device") + 1], "0")
        self.assertIn("--mmap-weights", command)

    def test_process_memory_peak_parses_linux_status(self) -> None:
        peak = benchmark.ProcessMemoryPeak()
        peak.update_from_linux_status(
            "VmHWM:\t2048 kB\n"
            "VmRSS:\t1536 kB\n"
            "RssAnon:\t1024 kB\n"
            "RssFile:\t512 kB\n"
        )
        peak.update_from_linux_status(
            "VmHWM:\t3072 kB\n"
            "VmRSS:\t2048 kB\n"
            "RssAnon:\t768 kB\n"
            "RssFile:\t1280 kB\n"
        )
        self.assertTrue(peak.supported)
        self.assertEqual(peak.peak_rss_bytes, 3072 * 1024)
        self.assertEqual(peak.max_sampled_rss_anon_bytes, 1024 * 1024)
        self.assertEqual(peak.max_sampled_rss_file_bytes, 1280 * 1024)

    def test_process_memory_peak_rejects_incomplete_status(self) -> None:
        peak = benchmark.ProcessMemoryPeak()
        peak.update_from_linux_status("VmRSS:\t1536 kB\n")
        self.assertFalse(peak.supported)

    def test_device_memory_peak_reports_delta_from_baseline(self) -> None:
        peak = benchmark.DeviceMemoryPeak(device=0)
        peak.update_from_nvidia_smi("512\n")
        peak.update_from_nvidia_smi("2048\n")
        self.assertTrue(peak.supported)
        self.assertEqual(peak.baseline_bytes, 512 * 1024 * 1024)
        self.assertEqual(peak.peak_bytes, 2048 * 1024 * 1024)
        self.assertEqual(peak.peak_delta_bytes, 1536 * 1024 * 1024)

    def test_device_memory_peak_rejects_invalid_output(self) -> None:
        peak = benchmark.DeviceMemoryPeak(device=0)
        peak.update_from_nvidia_smi("N/A\n")
        self.assertFalse(peak.supported)

    def test_device_memory_sampler_timeout_is_ignored(self) -> None:
        peak = benchmark.DeviceMemoryPeak(device=0)
        with (
            mock.patch.object(benchmark.shutil, "which", return_value="/tmp/nvidia-smi"),
            mock.patch.object(
                benchmark.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
            ),
        ):
            peak.sample_nvidia_smi()
        self.assertFalse(peak.supported)

    @unittest.skipUnless(os.name == "posix", "requires POSIX executable scripts")
    def test_process_wall_excludes_device_sampler_shutdown(self) -> None:
        output = self.make_output()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "fake_cli"
            cli.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "time.sleep(0.05)\n"
                f"print({output!r}, end='')\n",
                encoding="utf-8",
            )
            cli.chmod(0o755)

            nvidia_smi = root / "nvidia-smi"
            nvidia_smi.write_text(
                "#!/bin/sh\n"
                "sleep 0.5\n"
                "printf '1\\n'\n",
                encoding="utf-8",
            )
            nvidia_smi.chmod(0o755)

            path = str(root) + os.pathsep + os.environ.get("PATH", "")
            with mock.patch.dict(os.environ, {"PATH": path}):
                metrics, tokens = benchmark.run_benchmark(
                    binary=cli,
                    model=root,
                    image=root / "image.png",
                    prompt_mode="document",
                    max_tokens=1,
                    threads=1,
                    warmup=0,
                    repeat=1,
                    mmap_weights=False,
                    vision_vulkan=False,
                    vision_vulkan_device=0,
                    text_vulkan=False,
                    text_vulkan_device=0,
                    nvidia_smi_device=0,
                )

        self.assertEqual(tokens, (12, 34, 56))
        self.assertLess(metrics["benchmark_process_wall_ms"], 300.0)

    def test_output_directory_files(self) -> None:
        metrics = {key: float(index) for index, key in enumerate(benchmark.BENCHMARK_KEYS, start=1)}
        metrics["benchmark_process_wall_ms"] = 123.0
        metrics["benchmark_process_memory_supported"] = 1.0
        metrics["benchmark_process_peak_rss_bytes"] = 3 * 1024 * 1024
        metrics["benchmark_process_max_sampled_rss_anon_bytes"] = 2 * 1024 * 1024
        metrics["benchmark_process_max_sampled_rss_file_bytes"] = 1 * 1024 * 1024
        metrics["benchmark_device_memory_supported"] = 1.0
        metrics["benchmark_device_memory_device"] = 0.0
        metrics["benchmark_device_memory_baseline_bytes"] = 512 * 1024 * 1024
        metrics["benchmark_device_memory_peak_bytes"] = 2 * 1024 * 1024 * 1024
        metrics["benchmark_device_memory_peak_delta_bytes"] = 1536 * 1024 * 1024
        row = benchmark.BenchmarkRow(
            config="text_vulkan_p28",
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
            cold_start = (output / "cold_start.csv").read_text(encoding="utf-8")
            self.assertIn("config", cold_start)
            self.assertIn("benchmark_process_peak_rss_bytes", cold_start)
            self.assertIn("benchmark_device_memory_peak_delta_bytes", cold_start)
            self.assertIn("warm_inference_total_ms", (output / "warm_inference.csv").read_text(encoding="utf-8"))
            summary = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("text_vulkan_p28", summary)
            self.assertIn("peak RSS MiB", summary)
            self.assertIn("peak VRAM delta MiB", summary)


if __name__ == "__main__":
    unittest.main()
