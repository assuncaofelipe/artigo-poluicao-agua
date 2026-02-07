import argparse
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data import _read_csv, augment_repeat


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _label_with_unit(name: str) -> str:
    key = name.strip()
    key_up = key.upper()
    if key_up == "PH":
        return "pH"
    if key_up in {"OD", "DO"}:
        return "Dissolved oxygen (mg/L)"
    if "TURB" in key_up:
        return "Turbidity (NTU)"
    if key_up == "TEMP":
        return "Temperature (°C)"
    if key_up == "X":
        return "x (UTM, m)"
    if key_up == "Y":
        return "y (UTM, m)"
    return key


def _plot_hist_grid(cols: List[str], orig: np.ndarray, aug: np.ndarray, out_path: str, dpi: int):
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(9, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(cols):
        r = i // 2
        c = i % 2
        ax = axes[r][c]
        ax.hist(orig[:, i], bins=12, alpha=0.6, label="Original", color="#2f6f7e")
        ax.hist(aug[:, i], bins=12, alpha=0.4, label="Augmented (x10)", color="#e36f5a")
        ax.set_xlabel(_label_with_unit(col))
        ax.set_ylabel("Frequency (samples)")
        ax.legend()

    # hide extra axes
    for j in range(n, rows * 2):
        r = j // 2
        c = j % 2
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_box_grid(cols: List[str], orig: np.ndarray, aug: np.ndarray, out_path: str, dpi: int):
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(9, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(cols):
        r = i // 2
        c = i % 2
        ax = axes[r][c]
        ax.boxplot([orig[:, i], aug[:, i]], labels=["Original", "Augmented (x10)"], vert=True)
        ax.set_ylabel(_label_with_unit(col))

    for j in range(n, rows * 2):
        r = j // 2
        c = j % 2
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _split_cols(cols: List[str], size: int = 4):
    return [cols[i : i + size] for i in range(0, len(cols), size)]


def main():
    ap = argparse.ArgumentParser(description="Compare original vs augmented data distributions.")
    ap.add_argument("--factor", type=int, default=None, help="Override augmentation factor.")
    ap.add_argument("--noise-std", type=float, default=None, help="Override augmentation std.")
    args = ap.parse_args()

    cfg = Config()
    factor = args.factor if args.factor is not None else cfg.augment_factor
    noise_std = args.noise_std if args.noise_std is not None else cfg.augment_noise_std

    x, y = _read_csv(cfg.data_path, cfg.feature_cols, cfg.target_cols)
    x_aug, y_aug = augment_repeat(x, y, factor=factor, noise_std=noise_std, seed=cfg.seed)

    _ensure_dir(cfg.fig_dir)
    feature_cols = list(cfg.feature_cols)
    target_cols = list(cfg.target_cols)

    all_cols = feature_cols + target_cols
    orig_all = np.concatenate([x.numpy(), y.numpy()], axis=1)
    aug_all = np.concatenate([x_aug.numpy(), y_aug.numpy()], axis=1)

    for idx, chunk in enumerate(_split_cols(all_cols, size=4), start=1):
        sel = [all_cols.index(c) for c in chunk]
        _plot_hist_grid(
            chunk,
            orig_all[:, sel],
            aug_all[:, sel],
            os.path.join(cfg.fig_dir, f"aug_compare_hist_{idx}.png"),
            cfg.dpi,
        )
        _plot_box_grid(
            chunk,
            orig_all[:, sel],
            aug_all[:, sel],
            os.path.join(cfg.fig_dir, f"aug_compare_box_{idx}.png"),
            cfg.dpi,
        )


if __name__ == "__main__":
    main()
