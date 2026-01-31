import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, load_data
from encoding import rate_encode
from model import SNNRegressor
from figures import save_pred_scatter


def evaluate():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    (x_train, y_train), (x_val, y_val), (x_test, y_test), _, _ = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_col,
        cfg.test_split,
        cfg.val_split,
        cfg.seed,
    )

    test_ds = SerpaDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=1,
        beta=cfg.beta,
        thresh=cfg.thresh,
    ).to(device)

    ckpt = os.path.join(cfg.out_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loss_fn = nn.MSELoss()
    test_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            spikes = rate_encode(xb, cfg.time_steps)
            pred, _ = model(spikes)
            loss = loss_fn(pred, yb)
            test_loss += loss.item() * xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())

    test_loss /= len(test_loader.dataset)
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mae = torch.mean(torch.abs(preds - trues)).item()
    print(f"test_mse={test_loss:.4f} test_mae={mae:.4f}")
    save_pred_scatter(trues, preds, cfg.target_col, cfg.fig_dir, cfg.dpi)


if __name__ == "__main__":
    evaluate()
