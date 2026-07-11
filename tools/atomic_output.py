"""Transactional directory publishing for generated baseline artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def lexical_absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def has_symlink_component(path: Path) -> bool:
    path = lexical_absolute_path(path)
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def validate_case_names(names: list[object]) -> None:
    seen: set[str] = set()
    for name in names:
        unsafe = (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or any(character in name for character in '<>:"/\\|?*\0')
            or name.endswith((" ", "."))
            or (isinstance(name, str) and name.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES)
        )
        if unsafe:
            raise ValueError(f"unsafe case name: {name!r}")
        normalized = name.casefold()
        if normalized in seen:
            raise ValueError(f"duplicate case name: {name}")
        seen.add(normalized)


def build_directory_transactionally(
    output_dir: Path,
    builder: Callable[[Path], None],
    *,
    replace_existing: bool,
) -> None:
    output_dir = lexical_absolute_path(output_dir)
    if has_symlink_component(output_dir) or (output_dir.exists() and not output_dir.is_dir()):
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and not replace_existing:
        raise FileExistsError(f"output directory exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        builder(staging)
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
            backup.rmdir()
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except BaseException:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
