#!/usr/bin/env python3

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))

from patch_innerproduct_m1 import (
    PatchError,
    find_matching_gemm_layers,
    patch_param_text,
)


PARAM = """7767517
3 3
Input in0 0 1 in0
Gemm gemm_6 1 1 in0 hidden 10=-1 2=0 3=1 4=0 5=1 6=1 7=0 8=1024 9=3584
Gemm gemm_7 1 1 hidden out0 10=-1 2=0 3=1 4=0 5=1 6=1 7=0 8=2048 9=1024
"""


class PatchInnerProductM1Test(unittest.TestCase):
    def test_converts_only_requested_compatible_gemm(self) -> None:
        patched = patch_param_text(PARAM, {"gemm_6"})

        self.assertIn(
            "InnerProduct gemm_6 1 1 in0 hidden 0=1024 1=0 2=3670016 8=0",
            patched,
        )
        self.assertIn(
            "Gemm gemm_7 1 1 hidden out0 10=-1 2=0 3=1 4=0 5=1 6=1",
            patched,
        )
        self.assertEqual(patched.splitlines()[:2], ["7767517", "3 3"])

    def test_rejects_missing_requested_layer(self) -> None:
        with self.assertRaisesRegex(PatchError, "requested Gemm layer not found"):
            patch_param_text(PARAM, {"gemm_99"})

    def test_rejects_incompatible_gemm_contract(self) -> None:
        incompatible = PARAM.replace("3=1", "3=0", 1)
        with self.assertRaisesRegex(PatchError, "transB=1"):
            patch_param_text(incompatible, {"gemm_6"})

    def test_rejects_non_unit_alpha(self) -> None:
        incompatible = PARAM.replace("10=-1", "0=0.5 10=-1", 1)
        with self.assertRaisesRegex(PatchError, "alpha=1"):
            patch_param_text(incompatible, {"gemm_6"})

    def test_finds_only_gemms_with_the_requested_shape(self) -> None:
        layers = find_matching_gemm_layers(
            PARAM,
            input_size=3584,
            output_size=1024,
            expected_count=1,
        )

        self.assertEqual(layers, {"gemm_6"})

    def test_rejects_unexpected_matching_layer_count(self) -> None:
        with self.assertRaisesRegex(PatchError, "expected 24 matching Gemm layers, found 1"):
            find_matching_gemm_layers(
                PARAM,
                input_size=3584,
                output_size=1024,
                expected_count=24,
            )

    def test_cli_converts_all_layers_matching_a_guarded_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.param"
            output = Path(temp_dir) / "output.param"
            source.write_text(PARAM, encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "export" / "patch_innerproduct_m1.py"),
                    "--param",
                    str(source),
                    "--output",
                    str(output),
                    "--matching-shape",
                    "3584",
                    "1024",
                    "--expected-count",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = output.read_text(encoding="utf-8")
            self.assertIn("InnerProduct gemm_6", patched)
            self.assertIn("Gemm gemm_7", patched)


if __name__ == "__main__":
    unittest.main()
