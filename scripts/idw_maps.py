import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import contextily as ctx
except Exception:  # pragma: no cover - optional basemap
    ctx = None

CRS_UTM21S = "EPSG:32721"


def _add_osm_basemap(ax):
    if ctx is None:
        return
    try:
        ctx.add_basemap(
            ax,
            crs=CRS_UTM21S,
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6,
        )
    except Exception:
        pass


def idw(x, y, z, X, Y, power=2):
    dist2 = (X[..., None] - x[None, None, :]) ** 2 + (Y[..., None] - y[None, None, :]) ** 2
    dist2[dist2 == 0] = 1e-12
    w = 1.0 / (dist2 ** (power / 2))
    num = np.sum(w * z[None, None, :], axis=2)
    den = np.sum(w, axis=2)
    return num / den


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_path = script_dir / "data" / "dados_serpa.csv"

    xs, ys, phs, ods, turbs = [], [], [], [], []
    with data_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            xs.append(float(row["x"].replace(",", ".")))
            ys.append(float(row["y"].replace(",", ".")))
            phs.append(float(row["pH"].replace(",", ".")))
            ods.append(float(row["OD"].replace(",", ".")))
            turbs.append(float(row["TURB"].replace(",", ".")))

    xs = np.array(xs)
    ys = np.array(ys)
    phs = np.array(phs)
    ods = np.array(ods)
    turbs = np.array(turbs)

    nx, ny = 200, 200
    xi = np.linspace(xs.min(), xs.max(), nx)
    yi = np.linspace(ys.min(), ys.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    Z_ph = idw(xs, ys, phs, X, Y, power=2)
    Z_od = idw(xs, ys, ods, X, Y, power=2)
    Z_turb = idw(xs, ys, turbs, X, Y, power=2)

    out_dir = repo_root / "figuras"
    out_dir.mkdir(exist_ok=True)

    def save_map(Z, title, fname):
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        _add_osm_basemap(ax)
        im = plt.imshow(
            Z,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            cmap="viridis",
            aspect="auto",
            alpha=0.75,
        )
        plt.scatter(xs, ys, s=10, c="k", alpha=0.6, zorder=3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=600)
        plt.close()

    save_map(Z_ph, "pH (IDW)", "idw_ph.png")
    save_map(Z_od, "OD (IDW)", "idw_od.png")
    save_map(Z_turb, "Turbidez (IDW)", "idw_turb.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, Z, title in zip(
        axes, [Z_ph, Z_od, Z_turb], ["pH (IDW)", "OD (IDW)", "Turbidez (IDW)"]
    ):
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        _add_osm_basemap(ax)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            cmap="viridis",
            aspect="auto",
            alpha=0.75,
        )
        ax.scatter(xs, ys, s=8, c="k", alpha=0.6, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "idw_combined.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
