import torch

from config import Config
from data import load_data, load_raw_columns
from figures import (
    save_boxplots,
    save_corr_heatmap,
    save_spatial_map,
    save_target_hist,
)


def main():
    cfg = Config()
    (x_train, y_train), (x_val, y_val), (x_test, y_test), _, _ = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_col,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    raw_feat = load_raw_columns(cfg.data_path, cfg.feature_cols)
    x_raw = torch.stack([raw_feat[c] for c in cfg.feature_cols], dim=1)
    raw_target = load_raw_columns(cfg.data_path, (cfg.target_col,))
    y_raw = raw_target[cfg.target_col]

    save_boxplots(x_raw, cfg.feature_cols, cfg.fig_dir, cfg.dpi)
    save_corr_heatmap(x_raw, cfg.feature_cols, cfg.fig_dir, cfg.dpi)
    save_target_hist(y_raw, cfg.target_col, cfg.fig_dir, cfg.dpi)

    raw_cols = ("x", "y", "pH", "OD", "TURB")
    raw = load_raw_columns(cfg.data_path, raw_cols)
    x_raw = raw["x"]
    y_raw = raw["y"]
    for name in ("pH", "OD", "TURB"):
        if name in raw:
            save_spatial_map(x_raw, y_raw, raw[name], name, cfg.fig_dir, cfg.dpi)


if __name__ == "__main__":
    main()
