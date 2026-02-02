import itertools
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, augment_repeat, load_data
from encoding import rate_encode
from model import SNNRegressor


def choose_device():
    choice = input("Usar GPU (cuda) ou CPU? [gpu/cpu] ").strip().lower()
    if choice.startswith("g") and torch.cuda.is_available():
        return torch.device("cuda")
    if choice.startswith("g") and not torch.cuda.is_available():
        print("CUDA indisponivel; usando CPU.")
    return torch.device("cpu")


def train_one(cfg, device, hp, data_cache):
    (x_train, y_train), (x_val, y_val), _, mean, std, y_mean, y_std = data_cache
    if cfg.normalize_targets:
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std

    if cfg.augment_factor > 1:
        x_train, y_train = augment_repeat(
            x_train, y_train, cfg.augment_factor, 0.0, cfg.seed
        )

    train_ds = SerpaDataset(x_train, y_train)
    val_ds = SerpaDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=hp["hidden_size"],
        output_size=len(cfg.target_cols),
        beta=hp["beta"],
        thresh=hp["thresh"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if hp["aug_std"] > 0:
                xb = xb + torch.randn_like(xb) * hp["aug_std"]
            spikes = rate_encode(xb, hp["time_steps"])
            pred = model(spikes)
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
                spikes = rate_encode(xb, hp["time_steps"])
                pred = model(spikes)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_no_improve += 1

        if cfg.early_stopping and epochs_no_improve >= cfg.patience:
            break

    return best_val, best_epoch, best_state, mean, std, y_mean, y_std


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = choose_device()
    print(f"device={device.type}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    data_cache = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    grid = list(
        itertools.product(
            cfg.grid_hidden_sizes,
            cfg.grid_time_steps,
            cfg.grid_lr,
            cfg.grid_beta,
            cfg.grid_thresh,
            cfg.grid_aug_std,
        )
    )

    results = []
    best = {"val": float("inf")}
    for i, (hidden_size, time_steps, lr, beta, thresh, aug_std) in enumerate(grid, 1):
        hp = {
            "hidden_size": hidden_size,
            "time_steps": time_steps,
            "lr": lr,
            "beta": beta,
            "thresh": thresh,
            "aug_std": aug_std,
        }
        val, epoch, state, mean, std, y_mean, y_std = train_one(cfg, device, hp, data_cache)
        row = {"idx": i, "val_loss": val, "best_epoch": epoch, **hp}
        results.append(row)
        print(f"[{i}/{len(grid)}] val={val:.4f} hp={hp}")

        if val < best["val"]:
            best = {"val": val, "best_epoch": epoch, **hp}
            torch.save(state, os.path.join(cfg.out_dir, "grid_best_multi.pt"))
            with open(os.path.join(cfg.out_dir, "grid_norm_multi.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mean": mean.squeeze(0).tolist(),
                        "std": std.squeeze(0).tolist(),
                        "feature_cols": list(cfg.feature_cols),
                        "target_cols": list(cfg.target_cols),
                        "y_mean": y_mean.squeeze(0).tolist(),
                        "y_std": y_std.squeeze(0).tolist(),
                    },
                    f,
                )

    with open(os.path.join(cfg.out_dir, "grid_search_multi.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
