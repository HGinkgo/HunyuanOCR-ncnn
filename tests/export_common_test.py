#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "export"))

import _common


class ExportCommonTest(unittest.TestCase):
    def test_selects_transformers_5_13_components(self) -> None:
        embedding = object()
        text_backbone = SimpleNamespace(embed_tokens=embedding)
        vision_tower = object()
        model = SimpleNamespace(model=SimpleNamespace(language_model=text_backbone, vision_tower=vision_tower))

        self.assertIs(_common.text_backbone(model), text_backbone)
        self.assertIs(_common.input_embedding(model), embedding)
        self.assertIs(_common.vision_tower(model), vision_tower)

    def test_keeps_legacy_component_fallbacks(self) -> None:
        embedding = object()
        text_backbone = SimpleNamespace(embed_tokens=embedding)
        vision_tower = object()
        model = SimpleNamespace(model=text_backbone, vit=vision_tower)

        self.assertIs(_common.text_backbone(model), text_backbone)
        self.assertIs(_common.input_embedding(model), embedding)
        self.assertIs(_common.vision_tower(model), vision_tower)

    def test_run_pnnx_resolves_paths_before_changing_working_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hunyuanocr_pnnx_paths_") as tmp_text:
            tmp = Path(tmp_text)
            (tmp / "bin").mkdir()
            (tmp / "out").mkdir()
            (tmp / "bin" / "pnnx").touch()
            (tmp / "out" / "model.pt").touch()
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                with mock.patch.object(_common.subprocess, "run") as run:
                    _common.run_pnnx(
                        Path("bin/pnnx"),
                        Path("out/model.pt"),
                        Path("out/model.ncnn.param"),
                        Path("out/model.ncnn.bin"),
                        "[1,4]f32",
                    )
            finally:
                os.chdir(old_cwd)

            command = run.call_args.args[0]
            self.assertTrue(Path(command[0]).is_absolute())
            self.assertTrue(Path(command[1]).is_absolute())
            self.assertIn(f"ncnnparam={(tmp / 'out/model.ncnn.param').resolve()}", command)
            self.assertIn(f"ncnnbin={(tmp / 'out/model.ncnn.bin').resolve()}", command)
            self.assertEqual(run.call_args.kwargs["cwd"], (tmp / "out").resolve())


if __name__ == "__main__":
    unittest.main()
