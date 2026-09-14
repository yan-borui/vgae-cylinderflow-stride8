"""Parse repository source/configuration only; never import or run the model."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    python_files = [
        item
        for package in ("vgae_cf", "dgn4cfd", "scripts")
        for item in (root / package).rglob("*.py")
    ]
    for item in python_files:
        source = item.read_text(encoding="utf-8")
        ast.parse(source, filename=str(item))
        compile(
            source, str(item), "exec"
        )  # Compile only; the code object is never executed.
    configs = list((root / "vgae_cf" / "configs").glob("*.json"))
    for item in configs:
        json.loads(item.read_text(encoding="utf-8"))
    design = json.loads(
        (root / "vgae_cf" / "configs" / "scale_axes.json").read_text(encoding="utf-8")
    )
    points = set()
    for dimension, values in design["axes"].items():
        for value in values:
            point = {**design["center"], dimension: value}
            points.add((point["width"], tuple(point["depths"]), point["latent"]))
    if (
        len(points) != 8
        or len(points) + 1 + 3 * len(design["confirmation_seeds"]) != 15
    ):
        raise ValueError(
            "static campaign configuration differs from eight axes + baseline + six repeats"
        )
    try:
        import tomllib
    except ModuleNotFoundError:
        print("TOML parsing skipped: use Python 3.11+ for this optional static check")
    else:
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    print(
        f"Parsed {len(python_files)} Python files, {len(configs)} JSON configuration; design: 9 + 6 = 15"
    )
    print(
        "Application imports, GPU execution and software runtime tests: not performed"
    )


if __name__ == "__main__":
    main()
