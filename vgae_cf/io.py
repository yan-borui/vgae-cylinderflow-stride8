"""Atomic local artifacts and run locks, without content hashing."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess


def read_json(file_name: str | Path) -> dict:
    return json.loads(Path(file_name).read_text(encoding="utf-8"))


def write_json(file_name: str | Path, value: object) -> None:
    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_name.with_name(file_name.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file_name)


def append_json(file_name: str | Path, value: object) -> None:
    with Path(file_name).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


@contextmanager
def file_lock(directory: Path):
    """Use a process-scoped lock; stale files never prevent recovery."""
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def source_identity() -> dict:
    """Snapshot executable source text; Git identity is supplementary."""
    root = Path(__file__).resolve().parent.parent
    sources = {}
    for package in ("vgae_cf", "dgn4cfd"):
        for item in sorted((root / package).rglob("*")):
            if item.is_file() and item.suffix in (".py", ".json"):
                sources[item.relative_to(root).as_posix()] = item.read_text(
                    encoding="utf-8"
                )
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        revision = commit.stdout.strip() if commit.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    return {"commit": revision, "files": sources}
