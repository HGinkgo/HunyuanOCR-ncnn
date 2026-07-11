#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


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


if __name__ == "__main__":
    unittest.main()
