import os
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
from pathlib import Path
import numpy as np

try:
    import contextily as ctx
except Exception:  # pragma: no cover - optional basemap
    ctx = None

CRS_UTM21S = "EPSG:32721"


def _add_osm_basemap(ax, xmin, xmax, ymin, ymax):
    # Prefer offline local basemap first, so figures are deterministic without internet.
    candidates = [
        Path(__file__).resolve().parents[1] / "artigo" / "figuras" / "map_basemap_osm_21S_clean.png",
        Path(__file__).resolve().parents[1] / "figuras" / "map_basemap_osm_21S_clean.png",
        Path(__file__).resolve().parents[1] / "figuras" / "map_points_osm_21S.png",
        Path(__file__).resolve().parents[1] / "artigo" / "figuras" / "map_points_osm_21S.png",
    ]
    for path in candidates:
        if path.exists():
            try:
                img = plt.imread(path)
                # Trim white margins from exported map figures before reuse as basemap.
                if img.ndim == 3 and img.shape[2] >= 3:
                    rgb = img[..., :3]
                    # Keep pixels that are not almost white.
                    non_white = (rgb < 0.98).any(axis=2)
                    if non_white.any():
                        ys, xs = non_white.nonzero()
                        img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                    # Soften previously overlaid point markers from static map.
                    try:
                        from scipy.ndimage import gaussian_filter

                        rgb = np.stack(
                            [gaussian_filter(img[..., c], sigma=4.0) for c in range(3)], axis=-1
                        )
                        # Strong desaturation to suppress any old marker colors.
                        gray = (
                            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                        )[..., None]
                        img[..., :3] = 0.10 * rgb + 0.90 * gray
                    except Exception:
                        pass
                ax.imshow(
                    img,
                    extent=[xmin, xmax, ymin, ymax],
                    origin="upper",
                    aspect="auto",
                    alpha=0.40,
                    zorder=0,
                )
                return
            except Exception:
                continue

    # Online fallback when local static basemap is unavailable.
    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                crs=CRS_UTM21S,
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=0.6,
            )
            return
        except Exception:
            pass


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_loss_curve(train_losses: list, val_losses: list, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve_multi.png"), dpi=dpi)
    plt.close()


def save_boxplots(x, feature_cols, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    data = [x[:, i] for i in range(x.shape[1])]
    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=feature_cols, vert=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplots_features_multi.png"), dpi=dpi)
    plt.close()


def save_corr_heatmap(x, feature_cols, out_dir: str, dpi: int):
    import numpy as np

    _ensure_dir(out_dir)
    x_np = np.asarray(x)
    x_np = (x_np - x_np.mean(axis=0, keepdims=True)) / x_np.std(
        axis=0, keepdims=True
    )
    corr = (x_np.T @ x_np) / (x_np.shape[0] - 1)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=30, ha="right")
    plt.yticks(range(len(feature_cols)), feature_cols)
    # no title
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap_multi.png"), dpi=dpi)
    plt.close()


def save_pred_scatter(y_true: torch.Tensor, y_pred: torch.Tensor, names, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    for i, name in enumerate(names):
        yt = y_true[:, i].cpu().numpy()
        yp = y_pred[:, i].cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.scatter(yt, yp, s=18, alpha=0.7)
        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())
        plt.plot([mn, mx], [mn, mx], "--", color="#444")
        label = _label_with_unit(name)
        plt.xlabel(f"{label} (true)")
        plt.ylabel(f"{label} (predicted)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_vs_real_{name}_multi.png"), dpi=dpi)
        plt.close()


def save_residuals(y_true: torch.Tensor, y_pred: torch.Tensor, names, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    for i, name in enumerate(names):
        yt = y_true[:, i].cpu().numpy()
        yp = y_pred[:, i].cpu().numpy()
        res = yp - yt
        plt.figure(figsize=(5, 4))
        plt.scatter(yt, res, s=18, alpha=0.7)
        plt.axhline(0, linestyle="--", color="#444")
        label = _label_with_unit(name)
        plt.xlabel(f"{label} (true)")
        plt.ylabel("Residual (pred - true)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"residuals_{name}_multi.png"), dpi=dpi)
        plt.close()


def save_pollution_scatter(x, y, score, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    _add_osm_basemap(ax, min(x), max(x), min(y), max(y))
    sc = plt.scatter(x, y, c=score, s=24, cmap="inferno", zorder=3)
    plt.colorbar(sc, fraction=0.046, pad=0.04, label="Pollution index (0–1)")
    _format_utm_axes_km(plt.gca())
    # no title
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pollution_scatter.png"), dpi=dpi)
    plt.close()


def save_pollution_idw(x, y, score, out_dir: str, dpi: int, power: int = 2):
    _ensure_dir(out_dir)
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(score)
    nx, ny = 200, 200
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)
    dist2 = (X[..., None] - x[None, None, :]) ** 2 + (Y[..., None] - y[None, None, :]) ** 2
    dist2[dist2 == 0] = 1e-12
    w = 1.0 / (dist2 ** (power / 2))
    Z = (w * z[None, None, :]).sum(axis=2) / w.sum(axis=2)

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    _add_osm_basemap(ax, x.min(), x.max(), y.min(), y.max())
    im = plt.imshow(
        Z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="inferno",
        aspect="auto",
        alpha=0.75,
    )
    plt.scatter(x, y, s=10, c="k", alpha=0.6, zorder=3)
    _format_utm_axes_km(ax)
    # no title
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pollution index (0–1)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pollution_idw.png"), dpi=dpi)
    plt.close()


def save_idw_map(x, y, z, title, fname, out_dir: str, dpi: int, power: int = 2):
    import numpy as np

    _ensure_dir(out_dir)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    nx, ny = 200, 200
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)
    dist2 = (X[..., None] - x[None, None, :]) ** 2 + (Y[..., None] - y[None, None, :]) ** 2
    dist2[dist2 == 0] = 1e-12
    w = 1.0 / (dist2 ** (power / 2))
    Z = (w * z[None, None, :]).sum(axis=2) / w.sum(axis=2)

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    _add_osm_basemap(ax, x.min(), x.max(), y.min(), y.max())
    im = plt.imshow(
        Z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis",
        aspect="auto",
        alpha=0.75,
    )
    plt.scatter(x, y, s=10, c="k", alpha=0.6, zorder=3)
    _format_utm_axes_km(ax)
    label = _label_with_unit(title)
    # no title
    plt.colorbar(im, fraction=0.046, pad=0.04, label=label)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=dpi)
    plt.close()


def save_idw_combined(x, y, zs, titles, fname, out_dir: str, dpi: int, power: int = 2):
    import numpy as np

    _ensure_dir(out_dir)
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = 200, 200
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, z, title in zip(axes, zs, titles):
        z = np.asarray(z)
        dist2 = (X[..., None] - x[None, None, :]) ** 2 + (Y[..., None] - y[None, None, :]) ** 2
        dist2[dist2 == 0] = 1e-12
        w = 1.0 / (dist2 ** (power / 2))
        Z = (w * z[None, None, :]).sum(axis=2) / w.sum(axis=2)
        label = _label_with_unit(title)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        _add_osm_basemap(ax, x.min(), x.max(), y.min(), y.max())
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="viridis",
            aspect="auto",
            alpha=0.75,
        )
        ax.scatter(x, y, s=8, c="k", alpha=0.6, zorder=3)
        # no title
        _format_utm_axes_km(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=dpi)
    plt.close(fig)


def _label_with_unit(name: str) -> str:
    key = name.strip()
    key_up = key.upper()
    if "PH" == key_up or key_up.startswith("PH "):
        return "pH"
    if key_up.startswith("OD"):
        return "Dissolved oxygen (mg/L)"
    if "TURB" in key_up:
        return "Turbidity (NTU)"
    if "POLLUTION" in key_up:
        return "Pollution index (0–1)"
    return key


def _format_utm_axes_km(ax: plt.Axes):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}"))
    ax.set_xlabel("x (UTM, km)")
    ax.set_ylabel("y (UTM, km)")
