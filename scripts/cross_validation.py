import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, augment_repeat, kfold_indices, load_xy
from encoding import rate_encode
from model import SNNRegressor


def choose_device():
    choice = input("Usar GPU (cuda) ou CPU? [gpu/cpu] ").strip().lower()
    if choice.startswith("g") and torch.cuda.is_available():
        return torch.device("cuda")
    if choice.startswith("g") and not torch.cuda.is_available():
        print("CUDA indisponivel; usando CPU.")
    return torch.device("cpu")


def normalize(train_x, val_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std


def train_fold(cfg, device, train_x, train_y, val_x, val_y):
    train_x, val_x = normalize(train_x, val_x)
    if cfg.augment_factor > 1:
        train_x, train_y = augment_repeat(
            train_x, train_y, cfg.augment_factor, cfg.augment_noise_std, cfg.seed
        )
    train_ds = SerpaDataset(train_x, train_y)
    val_ds = SerpaDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=1,
        beta=cfg.beta,
        thresh=cfg.thresh,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if cfg.augment_noise_std > 0:
                xb = xb + torch.randn_like(xb) * cfg.augment_noise_std
            spikes = rate_encode(xb, cfg.time_steps)
            pred, _ = model(spikes)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                spikes = rate_encode(xb, cfg.time_steps)
                pred, _ = model(spikes)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1

        if cfg.early_stopping and epochs_no_improve >= cfg.patience:
            break

    return best_val, best_epoch


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = choose_device()
    print(f"device={device.type}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    x, y = load_xy(cfg.data_path, cfg.feature_cols, cfg.target_col)
    splits = kfold_indices(x.shape[0], cfg.k_folds, cfg.seed)

    results = []
    for i, (train_idx, val_idx) in enumerate(splits, 1):
        train_x, train_y = x[train_idx], y[train_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        val_loss, best_epoch = train_fold(cfg, device, train_x, train_y, val_x, val_y)
        results.append({"fold": i, "val_loss": val_loss, "best_epoch": best_epoch})
        print(f"fold={i} val_loss={val_loss:.4f} best_epoch={best_epoch}")

    avg = sum(r["val_loss"] for r in results) / len(results)
    with open(os.path.join(cfg.out_dir, "cross_validation.json"), "w", encoding="utf-8") as f:
        json.dump({"avg_val_loss": avg, "results": results}, f, indent=2)
    print(f"avg_val_loss={avg:.4f}")


if __name__ == "__main__":
    main()
