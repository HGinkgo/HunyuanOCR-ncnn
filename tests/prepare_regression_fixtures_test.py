#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/prepare_regression_fixtures.py"
FIXED_REVISION = "9e01f897bf8956f77a80c350dc0491d6bbbd43e6"


def load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("prepare_regression_fixtures", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, **overrides: str) -> None:
    values = {
        "model_revision": FIXED_REVISION,
        "transformers": "5.13.0",
        "device": "cpu",
        "attn_implementation": '"eager"',
        "dtype": "torch.float32",
        "repetition_penalty": "1.08",
        "processor.min_pixels": "262144",
        "processor.max_pixels": "524288",
        "do_sample": "False",
    }
    values.update(overrides)
    path.write_text(
        "HunyuanOCR Transformers fp32 deterministic baseline\n"
        + "\n".join(f"{key}: {value}" for key, value in values.items())
        + "\n\n[case]\n",
        encoding="utf-8",
    )


class PrepareRegressionFixturesTest(unittest.TestCase):
    def test_imports_without_numpy_and_reports_actionable_error_when_conversion_starts(self) -> None:
        module = load_module()
        with mock.patch.object(module, "np", None), self.assertRaisesRegex(
            SystemExit, "NumPy is required.*python -m pip install numpy"
        ):
            module.require_numpy()

    def test_accepts_fixed_cpu_fp32_eager_provenance(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            baseline = Path(tmp_text)
            write_summary(baseline / "summary.txt")
            module.validate_baseline_provenance(baseline, FIXED_REVISION, "5.13.0")

    def test_rejects_wrong_revision_before_fixture_conversion(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            baseline = Path(tmp_text)
            write_summary(baseline / "summary.txt", model_revision="wrong")
            with self.assertRaisesRegex(SystemExit, "model_revision mismatch"):
                module.validate_baseline_provenance(baseline, FIXED_REVISION, "5.13.0")

    def test_rejects_non_cpu_gold(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            baseline = Path(tmp_text)
            write_summary(baseline / "summary.txt", device="cuda")
            with self.assertRaisesRegex(SystemExit, "device mismatch"):
                module.validate_baseline_provenance(baseline, FIXED_REVISION, "5.13.0")

    def test_rejects_missing_summary(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            with self.assertRaisesRegex(SystemExit, "baseline summary not found"):
                module.validate_baseline_provenance(Path(tmp_text), FIXED_REVISION, "5.13.0")

    def test_rejects_missing_revision_even_if_expected_matches_old_sentinel(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            baseline = Path(tmp_text)
            write_summary(baseline / "summary.txt")
            lines = [
                line
                for line in (baseline / "summary.txt").read_text(encoding="utf-8").splitlines()
                if not line.startswith("model_revision:")
            ]
            (baseline / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "model_revision missing"):
                module.validate_baseline_provenance(baseline, "<missing>", "5.13.0")

    def test_rejects_empty_or_sentinel_revision_values(self) -> None:
        module = load_module()
        for value in ("", "unknown", "<missing>"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp_text:
                baseline = Path(tmp_text)
                write_summary(baseline / "summary.txt", model_revision=value)
                with self.assertRaisesRegex(SystemExit, "model_revision missing"):
                    module.validate_baseline_provenance(baseline, value, "5.13.0")

    def test_main_preserves_existing_output_when_case_input_is_invalid(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            baseline = root / "baseline"
            output = root / "fixtures"
            baseline.mkdir()
            output.mkdir()
            write_summary(baseline / "summary.txt")
            (output / "sentinel.txt").write_text("keep", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                '[{"name":"missing_case","image":"image.png","prompt_mode":"document"}]',
                encoding="utf-8",
            )
            argv = [
                "prepare_regression_fixtures.py",
                "--baseline-dir",
                str(baseline),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--force",
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                module.main()
            self.assertEqual((output / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_load_cases_rejects_path_traversal_and_duplicate_names(self) -> None:
        module = load_module()
        for names, message in [(["../escape"], "unsafe case name"), (["same", "same"], "duplicate case name")]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp_text:
                manifest = Path(tmp_text) / "manifest.json"
                manifest.write_text(
                    __import__("json").dumps(
                        [{"name": name, "image": "image.png", "prompt_mode": "document"} for name in names]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SystemExit, message):
                    module.load_cases(manifest)


if __name__ == "__main__":
    unittest.main()
