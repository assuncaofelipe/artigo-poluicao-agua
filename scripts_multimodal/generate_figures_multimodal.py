import json
import os
from pathlib import Path

import torch

from config import Config
from data import load_data
from encoding import rate_encode
from figures import (
    save_boxplots,
    save_corr_heatmap,
    save_idw_map,
    save_idw_combined,
    save_loss_curve,
    save_pollution_idw,
    save_pollution_scatter,
    save_pred_scatter,
    save_residuals,
)
from model import SNNRegressor


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test), _, _, y_mean, y_std = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=len(cfg.target_cols),
        beta=cfg.beta,
        thresh=cfg.thresh,
    )

    ckpt = os.path.join(cfg.out_dir, "best_multi.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        spikes = rate_encode(x_test, cfg.time_steps)
        pred_test = model(spikes)

    if cfg.normalize_targets:
        pred_test = pred_test * y_std + y_mean
        y_true = y_test
    else:
        y_true = y_test

    # scatter + residuals per target (test)
    save_pred_scatter(y_true, pred_test, cfg.target_cols, cfg.fig_dir, cfg.dpi)
    save_residuals(y_true, pred_test, cfg.target_cols, cfg.fig_dir, cfg.dpi)

    # loss curve if log exists
    loss_path = Path(cfg.out_dir) / "loss_multi.json"
    if loss_path.exists():
        data = json.loads(loss_path.read_text(encoding="utf-8"))
        save_loss_curve(data["train"], data["val"], cfg.fig_dir, cfg.dpi)

    # correlation and boxplots (raw features)
    import csv

    raw_path = Path(cfg.data_path)
    raw_feats = {k: [] for k in cfg.feature_cols}
    raw_targets = {k: [] for k in cfg.target_cols}
    xs, ys = [], []
    with raw_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            for k in cfg.feature_cols:
                raw_feats[k].append(float(row[k].replace(",", ".")))
            for k in cfg.target_cols:
                raw_targets[k].append(float(row[k].replace(",", ".")))
            xs.append(float(row["x"].replace(",", ".")))
            ys.append(float(row["y"].replace(",", ".")))

    x_raw = torch.tensor([raw_feats[k] for k in cfg.feature_cols]).t().numpy()
    save_boxplots(x_raw, cfg.feature_cols, cfg.fig_dir, cfg.dpi)
    save_corr_heatmap(x_raw, cfg.feature_cols, cfg.fig_dir, cfg.dpi)

    # IDW maps for pH/OD/TURB
    save_idw_map(xs, ys, raw_targets["pH"], "pH (IDW)", "idw_ph_multi.png", cfg.fig_dir, cfg.dpi)
    save_idw_map(xs, ys, raw_targets["OD"], "OD (IDW)", "idw_od_multi.png", cfg.fig_dir, cfg.dpi)
    save_idw_map(xs, ys, raw_targets["TURB"], "Turbidez (IDW)", "idw_turb_multi.png", cfg.fig_dir, cfg.dpi)
    save_idw_combined(
        xs,
        ys,
        [raw_targets["pH"], raw_targets["OD"], raw_targets["TURB"]],
        ["pH (IDW)", "OD (IDW)", "Turbidez (IDW)"],
        "idw_combined_multi.png",
        cfg.fig_dir,
        cfg.dpi,
    )

    # pollution maps based on predictions over all points
    x_all = torch.cat([x_train, x_val, x_test], dim=0)
    with torch.no_grad():
        spikes = rate_encode(x_all, cfg.time_steps)
        pred_all = model(spikes).cpu()
    if cfg.normalize_targets:
        pred_all = pred_all * y_std + y_mean

    mins = {k: min(v) for k, v in raw_targets.items()}
    maxs = {k: max(v) for k, v in raw_targets.items()}

    ph = pred_all[:, 0]
    od = pred_all[:, 1]
    turb = pred_all[:, 2]

    ph_norm = (ph - mins["pH"]) / (maxs["pH"] - mins["pH"] + 1e-8)
    od_norm = (od - mins["OD"]) / (maxs["OD"] - mins["OD"] + 1e-8)
    turb_norm = (turb - mins["TURB"]) / (maxs["TURB"] - mins["TURB"] + 1e-8)

    ph_dev = torch.abs(ph - cfg.ph_target)
    ph_dev_norm = (ph_dev - ph_dev.min()) / (ph_dev.max() - ph_dev.min() + 1e-8)
    od_inv = 1.0 - od_norm

    w_ph, w_od, w_turb = cfg.pollution_weights
    pollution = w_ph * ph_dev_norm + w_od * od_inv + w_turb * turb_norm

    save_pollution_scatter(xs, ys, pollution.numpy(), cfg.fig_dir, cfg.dpi)
    save_pollution_idw(xs, ys, pollution.numpy(), cfg.fig_dir, cfg.dpi)


if __name__ == "__main__":
    main()
