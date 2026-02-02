import os
import torch

from config import Config
from data import load_data
from model import SNNRegressor
from encoding import rate_encode


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
        pred = model(spikes)

    if cfg.normalize_targets:
        pred = pred * y_std + y_mean
        y_true = y_test
    else:
        y_true = y_test

    mae = torch.mean(torch.abs(pred - y_true), dim=0)
    mse = torch.mean((pred - y_true) ** 2, dim=0)

    r2 = []
    for i in range(y_true.shape[1]):
        ss_res = torch.sum((y_true[:, i] - pred[:, i]) ** 2).item()
        ss_tot = torch.sum((y_true[:, i] - y_true[:, i].mean()) ** 2).item()
        r2.append(1 - ss_res / ss_tot if ss_tot != 0 else 0.0)

    # accuracy within tolerance
    # tolerances: pH=0.2, OD=0.2, TURB=0.5 (adjust as needed)
    errs = torch.abs(pred - y_true)
    acc_ph = (errs[:, 0] <= 0.2).float().mean().item()
    acc_od = (errs[:, 1] <= 0.2).float().mean().item()
    acc_tu = (errs[:, 2] <= 0.5).float().mean().item()

    print("MAE", mae.tolist())
    print("MSE", mse.tolist())
    print("R2", r2)
    print("Acc_tol", [acc_ph, acc_od, acc_tu])


if __name__ == "__main__":
    main()
