#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODEL = REPO_ROOT / "tools" / "package_model.py"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_workspace(root: Path, *, include_stock: bool = True) -> Path:
    workspace = root / "workspace"
    files: dict[str, str] = {
        "models/export/dflash/ncnn/dflash.ncnn.param": "dflash-param\n",
        "models/export/dflash/ncnn/dflash.ncnn.bin": "dflash-bin\n",
        "models/export/text_decoder_dflash_aux/text_decoder_kv.ncnn.param": (
            "aux-decoder-param out1 out2 out3 out4\n"
        ),
        "models/export/text_decoder_dflash_aux/text_decoder_kv.ncnn.bin": "aux-decoder-bin\n",
        "models/export/vision_dynamic/ncnn/vision.ncnn.param": "vision-param\n",
        "models/export/vision_dynamic/ncnn/vision.ncnn.bin": "vision-bin\n",
        "models/export/vision_dynamic/ncnn/pos_embed.bin": "pos-embed\n",
    }
    if include_stock:
        files.update(
            {
                "models/tokenizer/vocab.txt": "vocab\n",
                "models/tokenizer/merges.txt": "merges\n",
                "models/tokenizer/special_tokens.json": "{}\n",
                "models/tokenizer/eos_ids.json": "[1]\n",
                "models/export/text_embed/text_embed.ncnn.param": "base-embed-param\n",
                "models/export/text_embed/text_embed.ncnn.bin": "base-embed-bin\n",
                "models/export/text_decoder/text_decoder_kv.ncnn.param": "base-decoder-param\n",
                "models/export/text_decoder/text_decoder_kv.ncnn.bin": "base-decoder-bin\n",
                "models/export/lm_head/lm_head.ncnn.param": "base-lm-param\n",
                "models/export/lm_head/lm_head.ncnn.bin": "base-lm-bin\n",
            }
        )
    for rel, content in files.items():
        write_file(workspace / rel, content)
    return workspace


def make_base_runtime(root: Path) -> Path:
    runtime = root / "base_runtime"
    files = {
        "tokenizer/vocab.txt": "runtime-vocab\n",
        "tokenizer/merges.txt": "runtime-merges\n",
        "tokenizer/special_tokens.json": "{\"runtime\": true}\n",
        "tokenizer/eos_ids.json": "[42]\n",
        "text_embed/text_embed.ncnn.param": "runtime-embed-param\n",
        "text_embed/text_embed.ncnn.bin": "runtime-embed-bin\n",
        "text_decoder/text_decoder_kv.ncnn.param": "runtime-decoder-param\n",
        "text_decoder/text_decoder_kv.ncnn.bin": "runtime-decoder-bin\n",
        "lm_head/lm_head.ncnn.param": "runtime-lm-param\n",
        "lm_head/lm_head.ncnn.bin": "runtime-lm-bin\n",
        # Present in real runtimes but must not be auto-copied by --base-runtime-dir.
        "vision/vision.ncnn.param": "runtime-vision-param\n",
        "vision/vision.ncnn.bin": "runtime-vision-bin\n",
        "vision/pos_embed.bin": "runtime-pos-embed\n",
    }
    for rel, content in files.items():
        write_file(runtime / rel, content)
    return runtime


def run_package(workspace: Path, output: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PACKAGE_MODEL),
        "--workspace",
        str(workspace),
        "--output",
        str(output),
        "--copy",
        *extra,
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


class PackageModelTest(unittest.TestCase):
    def test_only_canonical_dynamic_vision_is_exposed(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PACKAGE_MODEL), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--vision-backend", result.stdout)
        self.assertNotIn("--vision-summary", result.stdout)

    def test_default_uses_base_decoder_without_dflash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_default_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            output = root / "out_default"
            result = run_package(workspace, output, [])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dflash: disabled", result.stdout)
            self.assertIn("base_runtime_dir: stock-export", result.stdout)
            decoder_param = output / "text_decoder" / "text_decoder_kv.ncnn.param"
            self.assertEqual(decoder_param.read_text(encoding="utf-8"), "base-decoder-param\n")
            self.assertFalse((output / "dflash").exists())

    def test_dflash_uses_auxiliary_decoder_and_packages_dflash_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_dflash_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            output = root / "out_dflash"
            result = run_package(workspace, output, ["--dflash"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dflash: enabled", result.stdout)

            decoder_param = output / "text_decoder" / "text_decoder_kv.ncnn.param"
            decoder_bin = output / "text_decoder" / "text_decoder_kv.ncnn.bin"
            dflash_param = output / "dflash" / "dflash.ncnn.param"
            dflash_bin = output / "dflash" / "dflash.ncnn.bin"

            self.assertEqual(
                decoder_param.read_text(encoding="utf-8"),
                "aux-decoder-param out1 out2 out3 out4\n",
            )
            self.assertEqual(decoder_bin.read_text(encoding="utf-8"), "aux-decoder-bin\n")
            self.assertNotEqual(
                decoder_param.read_text(encoding="utf-8"),
                "base-decoder-param\n",
            )
            self.assertEqual(dflash_param.read_text(encoding="utf-8"), "dflash-param\n")
            self.assertEqual(dflash_bin.read_text(encoding="utf-8"), "dflash-bin\n")

            for path in (decoder_param, decoder_bin, dflash_param, dflash_bin):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())

    def test_copy_outputs_are_regular_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_copy_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            output = root / "out_copy"
            result = run_package(workspace, output, ["--dflash"])
            self.assertEqual(result.returncode, 0, result.stderr)
            for rel in (
                "dflash/dflash.ncnn.param",
                "dflash/dflash.ncnn.bin",
                "text_decoder/text_decoder_kv.ncnn.param",
                "text_decoder/text_decoder_kv.ncnn.bin",
                "tokenizer/vocab.txt",
            ):
                path = output / rel
                self.assertTrue(path.is_file(), rel)
                self.assertFalse(path.is_symlink(), rel)

    @unittest.skipIf(os.name == "nt", "creating source symlinks requires elevated privileges on Windows")
    def test_copy_materializes_source_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_symlink_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            # Point dflash sources through a symlink so copy must materialize real files.
            real_dflash = root / "real_dflash"
            write_file(real_dflash / "dflash.ncnn.param", "linked-dflash-param\n")
            write_file(real_dflash / "dflash.ncnn.bin", "linked-dflash-bin\n")
            dflash_dir = workspace / "models" / "export" / "dflash" / "ncnn"
            for name in ("dflash.ncnn.param", "dflash.ncnn.bin"):
                target = dflash_dir / name
                target.unlink()
                target.symlink_to(real_dflash / name)

            output = root / "out_symlink"
            result = run_package(workspace, output, ["--dflash"])
            self.assertEqual(result.returncode, 0, result.stderr)
            packaged = output / "dflash" / "dflash.ncnn.param"
            self.assertTrue(packaged.is_file())
            self.assertFalse(packaged.is_symlink())
            self.assertEqual(packaged.read_text(encoding="utf-8"), "linked-dflash-param\n")

    def test_missing_dflash_files_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_missing_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            (workspace / "models/export/dflash/ncnn/dflash.ncnn.bin").unlink()
            output = root / "out_missing"
            result = run_package(workspace, output, ["--dflash"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required file not found", result.stderr)

    def test_dflash_path_flags_require_dflash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_flags_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root)
            output = root / "out_flags"
            result = run_package(
                workspace,
                output,
                ["--dflash-dir", str(workspace / "models/export/dflash/ncnn")],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--dflash", result.stderr)

            result = run_package(
                workspace,
                output,
                [
                    "--dflash-decoder-dir",
                    str(workspace / "models/export/text_decoder_dflash_aux"),
                ],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--dflash", result.stderr)

    def test_base_runtime_dir_packages_without_stock_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_runtime_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root, include_stock=False)
            runtime = make_base_runtime(root)
            output = root / "out_runtime"
            result = run_package(
                workspace,
                output,
                ["--base-runtime-dir", str(runtime)],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dflash: disabled", result.stdout)
            self.assertIn(f"base_runtime_dir: {runtime.resolve()}", result.stdout)
            self.assertEqual(
                (output / "tokenizer/vocab.txt").read_text(encoding="utf-8"),
                "runtime-vocab\n",
            )
            self.assertEqual(
                (output / "text_embed/text_embed.ncnn.param").read_text(encoding="utf-8"),
                "runtime-embed-param\n",
            )
            self.assertEqual(
                (output / "lm_head/lm_head.ncnn.param").read_text(encoding="utf-8"),
                "runtime-lm-param\n",
            )
            self.assertEqual(
                (output / "text_decoder/text_decoder_kv.ncnn.param").read_text(encoding="utf-8"),
                "runtime-decoder-param\n",
            )
            self.assertFalse((output / "dflash").exists())
            # Vision still comes from the canonical dynamic export, not base runtime vision/.
            self.assertEqual(
                (output / "vision/vision.ncnn.param").read_text(encoding="utf-8"),
                "vision-param\n",
            )

    def test_base_runtime_dir_with_dflash_replaces_decoder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_runtime_dflash_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root, include_stock=False)
            runtime = make_base_runtime(root)
            output = root / "out_runtime_dflash"
            result = run_package(
                workspace,
                output,
                ["--base-runtime-dir", str(runtime), "--dflash"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dflash: enabled", result.stdout)
            self.assertEqual(
                (output / "tokenizer/vocab.txt").read_text(encoding="utf-8"),
                "runtime-vocab\n",
            )
            self.assertEqual(
                (output / "text_embed/text_embed.ncnn.bin").read_text(encoding="utf-8"),
                "runtime-embed-bin\n",
            )
            self.assertEqual(
                (output / "lm_head/lm_head.ncnn.bin").read_text(encoding="utf-8"),
                "runtime-lm-bin\n",
            )
            self.assertEqual(
                (output / "text_decoder/text_decoder_kv.ncnn.param").read_text(encoding="utf-8"),
                "aux-decoder-param out1 out2 out3 out4\n",
            )
            self.assertNotEqual(
                (output / "text_decoder/text_decoder_kv.ncnn.param").read_text(encoding="utf-8"),
                "runtime-decoder-param\n",
            )
            self.assertEqual(
                (output / "dflash/dflash.ncnn.param").read_text(encoding="utf-8"),
                "dflash-param\n",
            )

    def test_missing_base_runtime_file_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="package_model_runtime_missing_") as tmp:
            root = Path(tmp)
            workspace = make_workspace(root, include_stock=False)
            runtime = make_base_runtime(root)
            (runtime / "text_embed/text_embed.ncnn.bin").unlink()
            output = root / "out_runtime_missing"
            result = run_package(
                workspace,
                output,
                ["--base-runtime-dir", str(runtime)],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required file not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
