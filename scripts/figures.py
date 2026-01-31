import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch


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
    plt.ylabel("Valor")
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
    plt.title("Correlação (features normalizadas)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=dpi)
    plt.close()


def save_target_hist(y: torch.Tensor, target_col: str, out_dir: str, dpi: int):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.hist(y.cpu().numpy(), bins=12, color="#2f6f7e")
    plt.xlabel(target_col)
    plt.ylabel("Frequência")
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
    sc = plt.scatter(x.cpu().numpy(), y.cpu().numpy(), c=z.cpu().numpy(), s=26)
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name)
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
    plt.xlabel(f"{target_col} real")
    plt.ylabel(f"{target_col} predito")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pred_vs_real_{target_col}.png"), dpi=dpi)
    plt.close()


def save_loss_curve(
    train_losses: list, val_losses: list, out_dir: str, dpi: int
):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="treino")
    plt.plot(val_losses, label="validação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=dpi)
    plt.close()
