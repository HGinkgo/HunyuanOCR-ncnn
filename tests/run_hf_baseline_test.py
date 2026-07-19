#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/run_hf_baseline.py"


def load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("run_hf_baseline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunHfBaselineTest(unittest.TestCase):
    def test_parse_args_defaults_to_hunyuan_ocr_1_5_penalty(self) -> None:
        module = load_module()
        argv = ["run_hf_baseline.py", "--model-dir", "/tmp/hunyuanocr-1.5"]
        with mock.patch.object(sys, "argv", argv):
            args = module.parse_args()
        self.assertEqual(args.repetition_penalty, 1.08)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.expected_revision, "9e01f897bf8956f77a80c350dc0491d6bbbd43e6")
        self.assertEqual(args.expected_transformers_version, "5.13.0")

    def test_read_model_revision_from_snapshot_metadata(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("abc123\netag\n123.0\n", encoding="utf-8")

            self.assertEqual(module.read_model_revision(model_dir), "abc123")

    def test_read_model_revision_reports_unknown_without_metadata(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(module.read_model_revision(Path(tmp)), "unknown")

    def test_validate_model_revision_rejects_mismatch(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(SystemExit, "checkpoint revision mismatch"):
            module.validate_model_revision("wrong", "expected")

    def test_validate_model_revision_rejects_missing_even_if_expected_is_unknown(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(SystemExit, "checkpoint revision metadata missing"):
            module.validate_model_revision("unknown", "unknown")

    def test_validate_transformers_version_rejects_mismatch(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(SystemExit, "Transformers version mismatch"):
            module.validate_transformers_version("5.12.0", "5.13.0")

    def test_validate_transformers_version_rejects_missing_sentinels(self) -> None:
        module = load_module()
        for value in ("", "unknown", "<missing>"):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, "Transformers version missing"):
                module.validate_transformers_version(value, value)

    def test_generation_options_include_repetition_penalty(self) -> None:
        module = load_module()

        self.assertEqual(
            module.generation_options(max_new_tokens=512, repetition_penalty=1.08),
            {
                "max_new_tokens": 512,
                "do_sample": False,
                "repetition_penalty": 1.08,
            },
        )

    def test_load_cases_reads_512_token_limit(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "manifest.json"
            manifest.write_text(
                '[{"name":"page","image":"page.png","prompt_mode":"document","max_tokens":512}]',
                encoding="utf-8",
            )

            cases = module.load_cases(manifest)

        self.assertEqual(cases[0].max_tokens, 512)

    def test_resolve_position_ids_supports_transformers_5_13_contract(self) -> None:
        module = load_module()

        class FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name

            def permute(self, *dims: int):
                self.permuted_dims = dims
                return self

        computed = FakeTensor("computed")

        class FakeHunyuanModel:
            def compute_3d_position_ids(self, **kwargs):
                self.kwargs = kwargs
                return computed

        class FakeModel:
            def __init__(self) -> None:
                self.model = FakeHunyuanModel()

        inputs = {
            "input_ids": FakeTensor("input_ids"),
            "attention_mask": FakeTensor("attention_mask"),
            "image_grid_thw": FakeTensor("image_grid_thw"),
            "mm_token_type_ids": FakeTensor("mm_token_type_ids"),
        }
        model = FakeModel()

        model_position_ids, canonical_position_ids = module.resolve_position_ids(inputs, model)

        self.assertIs(model_position_ids, computed)
        self.assertIs(canonical_position_ids, computed)
        self.assertEqual(computed.permuted_dims, (1, 0, 2))
        self.assertEqual(
            model.model.kwargs,
            {
                "input_ids": inputs["input_ids"],
                "image_grid_thw": inputs["image_grid_thw"],
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs["mm_token_type_ids"],
            },
        )

    def test_build_text_embeddings_uses_public_model_interface(self) -> None:
        module = load_module()

        class FakeModel:
            def get_input_embeddings(self):
                return lambda input_ids: ("embeddings", input_ids)

        self.assertEqual(module.build_text_embeddings(FakeModel(), "input_ids"), ("embeddings", "input_ids"))

    def test_encode_image_features_uses_transformers_5_13_pooler_output(self) -> None:
        module = load_module()

        class FakeOutput:
            pooler_output = "vision_features"

        class FakeModel:
            def get_image_features(self, pixel_values, image_grid_thw, return_dict):
                self.args = (pixel_values, image_grid_thw, return_dict)
                return FakeOutput()

        model = FakeModel()

        self.assertEqual(module.encode_image_features(model, "pixels", "grid"), "vision_features")
        self.assertEqual(model.args, ("pixels", "grid", True))

    def test_encode_image_features_keeps_legacy_vit_fallback(self) -> None:
        module = load_module()

        class FakeModel:
            def vit(self, pixel_values, image_grid_thw):
                return ("legacy_vision_features", pixel_values, image_grid_thw)

        self.assertEqual(
            module.encode_image_features(FakeModel(), "pixels", "grid"),
            ("legacy_vision_features", "pixels", "grid"),
        )

    def test_get_placeholder_mask_uses_transformers_5_13_model(self) -> None:
        module = load_module()

        class FakeInnerModel:
            def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
                return ("mask", input_ids, inputs_embeds, image_features)

        class FakeModel:
            model = FakeInnerModel()

        self.assertEqual(
            module.get_placeholder_mask(FakeModel(), "ids", "embeddings", "features"),
            ("mask", "ids", "embeddings", "features"),
        )

    def test_get_placeholder_mask_keeps_legacy_top_level_fallback(self) -> None:
        module = load_module()

        class FakeModel:
            model = object()

            def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
                return ("legacy_mask", input_ids, inputs_embeds, image_features)

        self.assertEqual(
            module.get_placeholder_mask(FakeModel(), "ids", "embeddings", "features"),
            ("legacy_mask", "ids", "embeddings", "features"),
        )

    def test_normalize_chat_template_tokens_flattens_single_5_13_batch(self) -> None:
        module = load_module()

        self.assertEqual(module.normalize_chat_template_tokens([[10, 20, 30]]), [10, 20, 30])

    def test_normalize_chat_template_tokens_keeps_legacy_sequence(self) -> None:
        module = load_module()

        self.assertEqual(module.normalize_chat_template_tokens([10, 20, 30]), [10, 20, 30])

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
