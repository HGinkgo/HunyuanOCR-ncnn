#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from atomic_output import build_directory_transactionally, paths_overlap


class AtomicOutputTest(unittest.TestCase):
    def test_failed_build_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            output = root / "output"
            output.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")

            def fail_build(staging: Path) -> None:
                (staging / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("stop")

            with self.assertRaisesRegex(RuntimeError, "stop"):
                build_directory_transactionally(output, fail_build, replace_existing=True)

            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((output / "partial.txt").exists())

    def test_successful_build_replaces_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            output = root / "output"
            output.mkdir()
            (output / "old.txt").write_text("old", encoding="utf-8")

            def build(staging: Path) -> None:
                (staging / "new.txt").write_text("new", encoding="utf-8")

            build_directory_transactionally(output, build, replace_existing=True)

            self.assertFalse((output / "old.txt").exists())
            self.assertEqual((output / "new.txt").read_text(encoding="utf-8"), "new")

    def test_paths_overlap_for_equal_ancestor_or_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text).resolve()
            child = root / "child"
            peer = root.parent / f"{root.name}-peer"
            self.assertTrue(paths_overlap(root, root))
            self.assertTrue(paths_overlap(root, child))
            self.assertTrue(paths_overlap(child, root))
            self.assertFalse(paths_overlap(root, peer))

    def test_existing_regular_file_is_rejected_before_builder_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            output = Path(tmp_text) / "output"
            output.write_text("keep", encoding="utf-8")
            called = False

            def build(staging: Path) -> None:
                nonlocal called
                called = True

            with self.assertRaises(NotADirectoryError):
                build_directory_transactionally(output, build, replace_existing=True)
            self.assertFalse(called)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_existing_directory_symlink_is_rejected_before_builder_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            target = root / "target"
            target.mkdir()
            output = root / "output"
            output.symlink_to(target, target_is_directory=True)
            called = False

            def build(staging: Path) -> None:
                nonlocal called
                called = True

            with self.assertRaises(NotADirectoryError):
                build_directory_transactionally(output, build, replace_existing=True)
            self.assertFalse(called)
            self.assertTrue(output.is_symlink())

    def test_broken_symlink_is_rejected_before_builder_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            output = root / "output"
            output.symlink_to(root / "missing", target_is_directory=True)
            called = False

            def build(staging: Path) -> None:
                nonlocal called
                called = True

            with self.assertRaises(NotADirectoryError):
                build_directory_transactionally(output, build, replace_existing=True)
            self.assertFalse(called)
            self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
