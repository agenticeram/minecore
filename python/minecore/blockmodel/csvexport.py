"""Block model to CSV (centroids, indices, attributes)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from minecore.blockmodel import BlockModel


def export_block_model_csv(path: Path, model: BlockModel) -> int:
    """Write block centroids, indices and all attributes; returns the row count."""
    grid = model.grid
    xyz = grid.centroids()
    nx, ny, _ = grid.count
    index = np.arange(grid.blocks)
    i, j, k = index % nx, (index // nx) % ny, index // (nx * ny)
    table = model.table()
    names = list(model.attributes)
    columns = [np.asarray(table.column(n).to_numpy(zero_copy_only=False), float) for n in names]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["x", "y", "z", "i", "j", "k", *names])
        for b in range(grid.blocks):
            vals = ["" if math.isnan(c[b]) else f"{c[b]:.6g}" for c in columns]
            w.writerow(
                [
                    f"{xyz[b, 0]:.3f}",
                    f"{xyz[b, 1]:.3f}",
                    f"{xyz[b, 2]:.3f}",
                    i[b],
                    j[b],
                    k[b],
                    *vals,
                ]
            )
    return grid.blocks
