"""The NCCL patch series that ships inside the package (``sparknet/nccl/patches``).

The series file lists the patches in order; ``export`` copies the series and
its patches to a directory so an image build can apply them with ``git am``
after ``pip install``, without a checkout of this repository.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

SERIES_FILE = "series"


def patch_directory() -> Path:
    """The packaged ``patches`` directory (a real directory in both a checkout and a wheel)."""
    return Path(str(resources.files("sparknet.nccl") / "patches"))


def series(directory: Path | None = None) -> list[str]:
    """Patch file names in application order, comments and blank lines removed."""
    text = ((directory or patch_directory()) / SERIES_FILE).read_text()
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def series_text() -> str:
    return (patch_directory() / SERIES_FILE).read_text()


def missing(directory: Path | None = None) -> list[str]:
    """Series entries that have no patch file."""
    directory = directory or patch_directory()
    return [name for name in series(directory) if not (directory / name).is_file()]


def export(destination: str | Path) -> list[Path]:
    """Copy the series file and every patch into ``destination``; returns the written paths."""
    source = patch_directory()
    if missing(source):
        raise FileNotFoundError(f"patch files missing for {', '.join(missing(source))}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name in [SERIES_FILE, *series(source)]:
        target = destination / name
        shutil.copyfile(source / name, target)
        written.append(target)
    return written


__all__ = ["SERIES_FILE", "export", "missing", "patch_directory", "series", "series_text"]
