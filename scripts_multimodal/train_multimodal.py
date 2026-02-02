import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, augment_repeat, load_data
from encoding import rate_encode
from figures import save_loss_curve
from model import SNNRegressor


def choose_device():
    choice = input("Usar GPU (cuda) ou CPU? [gpu/cpu] ").strip().lower()
    if choice.startswith("g") and torch.cuda.is_available():
        return torch.device("cuda")
    if choice.startswith("g") and not torch.cuda.is_available():
        print("CUDA indisponivel; usando CPU.")
    return torch.device("cpu")


def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = choose_device()
    print(f"device={device.type}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    (x_train, y_train), (x_val, y_val), _, mean, std, y_mean, y_std = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    if cfg.normalize_targets:
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std

    if cfg.augment_factor > 1:
        x_train, y_train = augment_repeat(
            x_train, y_train, cfg.augment_factor, cfg.augment_noise_std, cfg.seed
        )

    train_ds = SerpaDataset(x_train, y_train)
    val_ds = SerpaDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=len(cfg.target_cols),
        beta=cfg.beta,
        thresh=cfg.thresh,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction="none")
    loss_w = torch.tensor(cfg.target_loss_weights, dtype=torch.float32, device=device)

    best_val = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if cfg.augment_noise_std > 0:
                xb = xb + torch.randn_like(xb) * cfg.augment_noise_std
            spikes = rate_encode(xb, cfg.time_steps)
            pred = model(spikes)
            loss = loss_fn(pred, yb)
            loss = (loss * loss_w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                spikes = rate_encode(xb, cfg.time_steps)
                pred = model(spikes)
                loss = loss_fn(pred, yb)
                loss = (loss * loss_w).mean()
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_multi.pt"))
            with open(os.path.join(cfg.out_dir, "norm_multi.json"), "w", encoding="utf-8") as f:
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
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if cfg.early_stopping and epochs_no_improve >= cfg.patience:
            print(
                f"early_stop epoch={epoch} best_val={best_val:.4f} "
                f"patience={cfg.patience}"
            )
            break

    with open(os.path.join(cfg.out_dir, "loss_multi.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)
    save_loss_curve(train_losses, val_losses, cfg.fig_dir, cfg.dpi)


if __name__ == "__main__":
    train()
