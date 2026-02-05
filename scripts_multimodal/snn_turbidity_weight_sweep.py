import argparse
import json
import os
from statistics import mean, pstdev

from benchmark_multiseed import _load_best_hparams, _metrics, resolve_device, run_snn
from config import Config


def parse_args():
    p = argparse.ArgumentParser(
        description="Varre pesos de perda da SNN para melhorar turbidez."
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 13, 21, 42, 84],
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "gpu"],
        default="auto",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    return p.parse_args()


def eval_weights(cfg, seeds, device, hp, loss_weights):
    cfg.target_loss_weights = loss_weights
    rows = []
    for s in seeds:
        y_true, y_pred = run_snn(cfg, s, device, hp)
        rows.append(_metrics(y_true, y_pred, cfg.target_cols))

    rec = {
        "loss_weights": list(loss_weights),
        "mae_macro_mean": mean([r["mae_macro"] for r in rows]),
        "mae_macro_std": pstdev([r["mae_macro"] for r in rows]) if len(rows) > 1 else 0.0,
        "rmse_macro_mean": mean([r["rmse_macro"] for r in rows]),
        "rmse_macro_std": pstdev([r["rmse_macro"] for r in rows]) if len(rows) > 1 else 0.0,
        "r2_macro_mean": mean([r["r2_macro"] for r in rows]),
        "r2_macro_std": pstdev([r["r2_macro"] for r in rows]) if len(rows) > 1 else 0.0,
        "mae_pH_mean": mean([r["mae_pH"] for r in rows]),
        "mae_OD_mean": mean([r["mae_OD"] for r in rows]),
        "mae_TURB_mean": mean([r["mae_TURB"] for r in rows]),
    }
    return rec


def main():
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    device = resolve_device(args.device)
    hp = _load_best_hparams(cfg)

    candidates = [
        (0.2, 0.2, 0.6),  # atual
        (0.15, 0.15, 0.7),
        (0.1, 0.1, 0.8),
        (0.05, 0.15, 0.8),
    ]

    results = []
    for w in candidates:
        rec = eval_weights(cfg, args.seeds, device, hp, w)
        results.append(rec)
        print(
            f"weights={w} mae_macro={rec['mae_macro_mean']:.4f} "
            f"mae_turb={rec['mae_TURB_mean']:.4f} r2_macro={rec['r2_macro_mean']:.4f}"
        )

    best_turb = min(results, key=lambda x: x["mae_TURB_mean"])
    best_macro = min(results, key=lambda x: x["mae_macro_mean"])

    out = {
        "results": results,
        "best_by_turbidity_mae": best_turb,
        "best_by_macro_mae": best_macro,
    }
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, "snn_weight_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
