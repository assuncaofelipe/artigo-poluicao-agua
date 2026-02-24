import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import ticker
import torch

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


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_boxplots(
    x: torch.Tensor, feature_cols: Tuple[str, ...], out_dir: str, dpi: int
):
    _ensure_dir(out_dir)
    data = [x[:, i].cpu().numpy() for i in range(x.shape[1])]
    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=feature_cols, vert=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplots_features.png"), dpi=dpi)
    plt.close()


def save_corr_heatmap(
    x: torch.Tensor, feature_cols: Tuple[str, ...], out_dir: str, dpi: int
):
    _ensure_dir(out_dir)
    x_np = x.cpu().numpy()
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
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=dpi)
    plt.close()


def save_target_hist(y: torch.Tensor, target_col: str, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.hist(y.cpu().numpy(), bins=12, color="#2f6f7e")
    plt.xlabel(_label_with_unit(target_col))
    plt.ylabel("Frequency (samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{target_col}.png"), dpi=dpi)
    plt.close()


def save_spatial_map(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    name: str,
    out_dir: str,
    dpi: int,
):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    xs = x.cpu().numpy()
    ys = y.cpu().numpy()
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    _add_osm_basemap(ax)
    sc = plt.scatter(xs, ys, c=z.cpu().numpy(), s=26, zorder=3)
    label = _label_with_unit(name)
    plt.colorbar(sc, fraction=0.046, pad=0.04, label=label)
    _format_utm_axes_km(ax)
    # no title
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"map_{name}.png"), dpi=dpi)
    plt.close()


def save_pred_scatter(
    y_true: torch.Tensor, y_pred: torch.Tensor, target_col: str, out_dir: str, dpi: int
):
    _ensure_dir(out_dir)
    yt = y_true.cpu().numpy().flatten()
    yp = y_pred.cpu().numpy().flatten()
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=18, alpha=0.7)
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    plt.plot([mn, mx], [mn, mx], "--", color="#444")
    label = _label_with_unit(target_col)
    plt.xlabel(f"{label} (true)")
    plt.ylabel(f"{label} (predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pred_vs_real_{target_col}.png"), dpi=dpi)
    plt.close()


def save_loss_curve(
    train_losses: list, val_losses: list, out_dir: str, dpi: int
):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=dpi)
    plt.close()


def _label_with_unit(name: str) -> str:
    key = name.strip()
    key_up = key.upper()
    if key_up == "PH":
        return "pH"
    if key_up in {"OD", "DO", "DISSOLVED OXYGEN"}:
        return "Dissolved oxygen (mg/L)"
    if key_up in {"TURB", "TURBIDEZ", "TURBIDITY"}:
        return "Turbidity (NTU)"
    if key_up == "TEMP":
        return "Temperature (°C)"
    if key_up in {"COR_A", "COR_V"}:
        return key  # unknown unit
    return key


def _format_utm_axes_km(ax: plt.Axes):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}"))
    ax.set_xlabel("x (UTM, km)")
    ax.set_ylabel("y (UTM, km)")
