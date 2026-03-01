#!/usr/bin/env python3
from pathlib import Path
import csv

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    data_path = repo / "scripts" / "data" / "dados_serpa.csv"
    out1 = repo / "figuras" / "map_basemap_osm_21S_clean.png"
    out2 = repo / "artigo" / "figuras" / "map_basemap_osm_21S_clean.png"

    xs, ys = [], []
    with data_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            xs.append(float(row["x"].replace(",", ".")))
            ys.append(float(row["y"].replace(",", ".")))
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ctx.add_basemap(ax, crs="EPSG:32721", source=ctx.providers.OpenStreetMap.Mapnik, alpha=1.0)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=500, bbox_inches="tight", pad_inches=0)
    fig.savefig(out2, dpi=500, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"saved {out1}")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
