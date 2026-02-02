import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, load_data
from encoding import rate_encode
from figures import save_pollution_idw, save_pollution_scatter, save_pred_scatter
from model import SNNRegressor


def choose_device():
    choice = input("Usar GPU (cuda) ou CPU? [gpu/cpu] ").strip().lower()
    if choice.startswith("g") and torch.cuda.is_available():
        return torch.device("cuda")
    if choice.startswith("g") and not torch.cuda.is_available():
        print("CUDA indisponivel; usando CPU.")
    return torch.device("cpu")


def evaluate():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = choose_device()
    print(f"device={device.type}")

    (x_train, y_train), (x_val, y_val), (x_test, y_test), mean, std, y_mean, y_std = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    if cfg.normalize_targets:
        y_test = (y_test - y_mean) / y_std

    test_ds = SerpaDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=len(cfg.target_cols),
        beta=cfg.beta,
        thresh=cfg.thresh,
    ).to(device)

    ckpt = os.path.join(cfg.out_dir, "best_multi.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loss_fn = nn.MSELoss(reduction="none")
    loss_w = torch.tensor(cfg.target_loss_weights, dtype=torch.float32, device=device)
    test_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            spikes = rate_encode(xb, cfg.time_steps)
            pred = model(spikes)
            loss = loss_fn(pred, yb)
            loss = (loss * loss_w).mean()
            test_loss += loss.item() * xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())

    test_loss /= len(test_loader.dataset)
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    if cfg.normalize_targets:
        preds_den = preds * y_std + y_mean
        trues_den = trues * y_std + y_mean
    else:
        preds_den = preds
        trues_den = trues

    mae = torch.mean(torch.abs(preds_den - trues_den), dim=0).tolist()
    print(f"test_mse={test_loss:.4f} per_target_mae={mae}")
    save_pred_scatter(trues_den, preds_den, cfg.target_cols, cfg.fig_dir, cfg.dpi)

    # generate pollution index map using predictions over all points
    # compute predictions for full dataset
    x_all = torch.cat([x_train, x_val, x_test], dim=0)
    with torch.no_grad():
        spikes = rate_encode(x_all.to(device), cfg.time_steps)
        pred_all = model(spikes).cpu()

    if cfg.normalize_targets:
        pred_all = pred_all * y_std + y_mean

    # load raw x,y for plotting (same order as CSV)
    import csv
    from pathlib import Path
    raw_path = Path(cfg.data_path)
    xs, ys = [], []
    with raw_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            xs.append(float(row["x"].replace(",", ".")))
            ys.append(float(row["y"].replace(",", ".")))

    # compute pollution index (0..1)
    # normalization by min/max from raw data
    raw_targets = {k: [] for k in cfg.target_cols}
    with raw_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            for k in cfg.target_cols:
                raw_targets[k].append(float(row[k].replace(",", ".")))

    mins = {k: min(v) for k, v in raw_targets.items()}
    maxs = {k: max(v) for k, v in raw_targets.items()}

    # pred_all order: [pH, OD, TURB]
    ph = pred_all[:, 0]
    od = pred_all[:, 1]
    turb = pred_all[:, 2]

    # normalize
    ph_norm = (ph - mins["pH"]) / (maxs["pH"] - mins["pH"] + 1e-8)
    od_norm = (od - mins["OD"]) / (maxs["OD"] - mins["OD"] + 1e-8)
    turb_norm = (turb - mins["TURB"]) / (maxs["TURB"] - mins["TURB"] + 1e-8)

    # higher is worse: turb high, OD low, pH away from neutral
    ph_dev = torch.abs(ph - cfg.ph_target)
    ph_dev_norm = (ph_dev - ph_dev.min()) / (ph_dev.max() - ph_dev.min() + 1e-8)
    od_inv = 1.0 - od_norm

    w_ph, w_od, w_turb = cfg.pollution_weights
    pollution = w_ph * ph_dev_norm + w_od * od_inv + w_turb * turb_norm

    save_pollution_scatter(xs, ys, pollution.numpy(), cfg.fig_dir, cfg.dpi)
    save_pollution_idw(xs, ys, pollution.numpy(), cfg.fig_dir, cfg.dpi)


if __name__ == "__main__":
    evaluate()
