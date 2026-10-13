"""Group raw images by capacitor type."""

from __future__ import annotations

import shutil
from pathlib import Path

from capacitor_detection.constants import CAPACITOR_TYPES


def sort_by_type(src_dir: Path, dst_dir: Path, types: tuple[str, ...] = CAPACITOR_TYPES) -> int:
    """Copy files from ``src_dir/<batch>/<TYPE ...>/`` into ``dst_dir/<TYPE>/``.

    Sub-folders are matched on their first three characters (e.g. ``"C05 VS-W"`` -> ``C05``).
    Returns the number of files copied.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    count = 0
    for batch in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        for folder in sorted(p for p in batch.iterdir() if p.is_dir()):
            kind = folder.name[:3]
            if kind not in types:
                continue
            target = dst_dir / kind
            target.mkdir(parents=True, exist_ok=True)
            for file in folder.iterdir():
                if file.is_file():
                    shutil.copy(file, target / file.name)
                    count += 1
    return count
