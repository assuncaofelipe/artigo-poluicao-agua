import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

from benchmark_multiseed import _load_best_hparams, _metrics, resolve_device, run_snn
from config import Config


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: SNN with vs without data augmentation.")
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 84], help="Seeds for repeated evaluation.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="auto")
    p.add_argument("--epochs", type=int, default=250, help="Override SNN epochs for faster ablation.")
    p.add_argument("--factor-noaug", type=int, default=1)
    p.add_argument("--factor-aug", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument("--fig-dir", default="./figuras")
    return p.parse_args()


def summarize(rows):
    keys = [
        "mae_macro", "rmse_macro", "r2_macro",
        "mae_pH", "mae_OD", "mae_TURB",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[f"{k}_mean"] = float(mean(vals))
        out[f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def run_condition(cfg: Config, hp: dict, device, seeds, augment_factor: int, noise_std: float):
    rows = []
    cfg.augment_factor = augment_factor
    cfg.augment_noise_std = noise_std
    for seed in seeds:
        y_true, y_pred = run_snn(cfg, seed, device, hp)
        rows.append(_metrics(y_true, y_pred, cfg.target_cols))
    return rows, summarize(rows)


def save_plot(sum_no, sum_aug, out_path: Path, dpi: int = 600):
    labels = ["MAE macro", "RMSE macro", r"$R^2$ macro"]
    no_vals = [sum_no["mae_macro_mean"], sum_no["rmse_macro_mean"], sum_no["r2_macro_mean"]]
    aug_vals = [sum_aug["mae_macro_mean"], sum_aug["rmse_macro_mean"], sum_aug["r2_macro_mean"]]

    x = range(len(labels))
    w = 0.36
    plt.figure(figsize=(7.0, 4.2))
    plt.bar([i - w / 2 for i in x], no_vals, width=w, label="No augmentation", color="#b0b7c3", edgecolor="#333")
    plt.bar([i + w / 2 for i in x], aug_vals, width=w, label="Augmentation x10", color="#1f77b4", edgecolor="#333")
    plt.xticks(list(x), labels)
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    args = parse_args()
    cfg = Config()
    cfg.out_dir = args.out_dir
    cfg.fig_dir = args.fig_dir
    cfg.epochs = args.epochs
    device = resolve_device(args.device)
    hp = _load_best_hparams(cfg)
    hp["aug_std"] = args.noise_std

    rows_no, sum_no = run_condition(cfg, hp, device, args.seeds, args.factor_noaug, 0.0)
    rows_aug, sum_aug = run_condition(cfg, hp, device, args.seeds, args.factor_aug, args.noise_std)

    mae_impr_pct = ((sum_no["mae_macro_mean"] - sum_aug["mae_macro_mean"]) / sum_no["mae_macro_mean"]) * 100.0
    rmse_impr_pct = ((sum_no["rmse_macro_mean"] - sum_aug["rmse_macro_mean"]) / sum_no["rmse_macro_mean"]) * 100.0
    r2_gain = sum_aug["r2_macro_mean"] - sum_no["r2_macro_mean"]

    out_dir = Path(cfg.out_dir)
    fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "augmentation_ablation.json"
    csv_path = out_dir / "augmentation_ablation_summary.csv"
    fig_path = fig_dir / "augmentation_ablation_bar.png"

    payload = {
        "seeds": args.seeds,
        "device": str(device),
        "epochs": cfg.epochs,
        "factor_noaug": args.factor_noaug,
        "factor_aug": args.factor_aug,
        "noise_std": args.noise_std,
        "no_augmentation": sum_no,
        "augmentation": sum_aug,
        "improvement": {
            "mae_macro_pct_reduction": mae_impr_pct,
            "rmse_macro_pct_reduction": rmse_impr_pct,
            "r2_macro_gain": r2_gain,
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "mae_macro_mean", "mae_macro_std", "rmse_macro_mean", "rmse_macro_std", "r2_macro_mean", "r2_macro_std"])
        writer.writerow(["no_augmentation", sum_no["mae_macro_mean"], sum_no["mae_macro_std"], sum_no["rmse_macro_mean"], sum_no["rmse_macro_std"], sum_no["r2_macro_mean"], sum_no["r2_macro_std"]])
        writer.writerow(["augmentation_x10", sum_aug["mae_macro_mean"], sum_aug["mae_macro_std"], sum_aug["rmse_macro_mean"], sum_aug["rmse_macro_std"], sum_aug["r2_macro_mean"], sum_aug["r2_macro_std"]])

    save_plot(sum_no, sum_aug, fig_path)

    print("Ablation complete")
    print(f"MAE macro reduction: {mae_impr_pct:.2f}%")
    print(f"RMSE macro reduction: {rmse_impr_pct:.2f}%")
    print(f"R2 macro gain: {r2_gain:.4f}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"FIG: {fig_path}")


if __name__ == "__main__":
    main()
