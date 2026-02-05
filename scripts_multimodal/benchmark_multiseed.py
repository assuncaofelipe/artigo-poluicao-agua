import argparse
import csv
import json
import os
import random
import time
from statistics import mean, pstdev

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import SerpaDataset, augment_repeat, load_data
from encoding import rate_encode
from model import SNNRegressor


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark multiseed: SNN vs baselines classicos (multimodal)."
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 13, 21, 42, 84],
        help="Lista de seeds para repetir os experimentos.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "gpu"],
        default="auto",
        help="Dispositivo para treinar a SNN.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override de epocas para SNN (padrao: config.py).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["snn", "rf", "mlp"],
        choices=["snn", "rf", "mlp"],
        help="Modelos a comparar.",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str):
    if choice == "gpu":
        choice = "cuda"
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitada, mas nao esta disponivel.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_best_hparams(cfg: Config):
    path = os.path.join(cfg.out_dir, "grid_search_multi.json")
    hp = {
        "hidden_size": cfg.hidden_size,
        "time_steps": cfg.time_steps,
        "lr": cfg.lr,
        "beta": cfg.beta,
        "thresh": cfg.thresh,
        "aug_std": cfg.augment_noise_std,
    }
    if not os.path.exists(path):
        return hp
    try:
        with open(path, encoding="utf-8") as f:
            best = json.load(f).get("best", {})
        hp.update(
            {
                "hidden_size": int(best.get("hidden_size", hp["hidden_size"])),
                "time_steps": int(best.get("time_steps", hp["time_steps"])),
                "lr": float(best.get("lr", hp["lr"])),
                "beta": float(best.get("beta", hp["beta"])),
                "thresh": float(best.get("thresh", hp["thresh"])),
                "aug_std": float(best.get("aug_std", hp["aug_std"])),
            }
        )
    except Exception:
        pass
    return hp


def _metrics(y_true, y_pred, target_cols):
    out = {}
    for i, t in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[f"mae_{t}"] = float(mean_absolute_error(yt, yp))
        out[f"rmse_{t}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        out[f"r2_{t}"] = float(r2_score(yt, yp))
    out["mae_macro"] = float(np.mean([out[f"mae_{t}"] for t in target_cols]))
    out["rmse_macro"] = float(np.mean([out[f"rmse_{t}"] for t in target_cols]))
    out["r2_macro"] = float(np.mean([out[f"r2_{t}"] for t in target_cols]))
    return out


def run_snn(cfg, seed, device, hp):
    set_seed(seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), _, _, y_mean, y_std = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        seed,
    )

    if cfg.normalize_targets:
        y_train_n = (y_train - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std
    else:
        y_train_n = y_train.clone()
        y_val_n = y_val.clone()

    if cfg.augment_factor > 1:
        x_train, y_train_n = augment_repeat(
            x_train, y_train_n, cfg.augment_factor, hp["aug_std"], seed
        )

    train_ds = SerpaDataset(x_train, y_train_n)
    val_ds = SerpaDataset(x_val, y_val_n)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=hp["hidden_size"],
        output_size=len(cfg.target_cols),
        beta=hp["beta"],
        thresh=hp["thresh"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss(reduction="none")
    loss_w = torch.tensor(cfg.target_loss_weights, dtype=torch.float32, device=device)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    epochs = cfg.epochs
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            spikes = rate_encode(xb, hp["time_steps"])
            pred = model(spikes)
            loss = loss_fn(pred, yb)
            loss = (loss * loss_w).mean()
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
                loss = (loss * loss_w).mean()
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if cfg.early_stopping and no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        spikes = rate_encode(x_test.to(device), hp["time_steps"])
        pred = model(spikes).cpu()

    if cfg.normalize_targets:
        pred = pred * y_std + y_mean
    return y_test.numpy(), pred.numpy()


def run_rf(cfg, seed):
    set_seed(seed)
    (x_train, y_train), _, (x_test, y_test), _, _, _, _ = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        seed,
    )
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train.numpy(), y_train.numpy())
    pred = model.predict(x_test.numpy())
    return y_test.numpy(), pred


def run_mlp(cfg, seed):
    set_seed(seed)
    (x_train, y_train), _, (x_test, y_test), _, _, _, _ = load_data(
        cfg.data_path,
        cfg.feature_cols,
        cfg.target_cols,
        cfg.test_split,
        cfg.val_split,
        seed,
    )
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=4000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
    )
    model.fit(x_train.numpy(), y_train.numpy())
    pred = model.predict(x_test.numpy())
    return y_test.numpy(), pred


def summarize(rows, target_cols):
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summary = []
    keys = (
        [f"mae_{t}" for t in target_cols]
        + [f"rmse_{t}" for t in target_cols]
        + [f"r2_{t}" for t in target_cols]
        + ["mae_macro", "rmse_macro", "r2_macro"]
    )
    for model_name, model_rows in by_model.items():
        rec = {"model": model_name, "n_runs": len(model_rows)}
        for k in keys:
            vals = [r[k] for r in model_rows]
            rec[f"{k}_mean"] = mean(vals)
            rec[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        summary.append(rec)
    return summary


def write_outputs(cfg, rows, summary, target_cols):
    os.makedirs(cfg.out_dir, exist_ok=True)
    raw_path = os.path.join(cfg.out_dir, "benchmark_multiseed_raw.csv")
    summary_path = os.path.join(cfg.out_dir, "benchmark_multiseed_summary.csv")
    md_path = os.path.join(cfg.out_dir, "benchmark_multiseed_table.md")
    json_path = os.path.join(cfg.out_dir, "benchmark_multiseed.json")

    raw_keys = ["seed", "model"] + sorted([k for k in rows[0].keys() if k not in {"seed", "model"}])
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_keys)
        w.writeheader()
        w.writerows(rows)

    summary_keys = ["model", "n_runs"]
    for metric_prefix in (
        [f"mae_{t}" for t in target_cols]
        + [f"rmse_{t}" for t in target_cols]
        + [f"r2_{t}" for t in target_cols]
        + ["mae_macro", "rmse_macro", "r2_macro"]
    ):
        summary_keys.extend([f"{metric_prefix}_mean", f"{metric_prefix}_std"])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_keys)
        w.writeheader()
        w.writerows(summary)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"raw": rows, "summary": summary}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Modelo | MAE macro (mean±std) | RMSE macro (mean±std) | R2 macro (mean±std) |\n")
        f.write("|---|---:|---:|---:|\n")
        for rec in sorted(summary, key=lambda x: x["mae_macro_mean"]):
            f.write(
                f"| {rec['model']} | "
                f"{rec['mae_macro_mean']:.4f} ± {rec['mae_macro_std']:.4f} | "
                f"{rec['rmse_macro_mean']:.4f} ± {rec['rmse_macro_std']:.4f} | "
                f"{rec['r2_macro_mean']:.4f} ± {rec['r2_macro_std']:.4f} |\n"
            )

    return raw_path, summary_path, md_path, json_path


def main():
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    device = resolve_device(args.device)
    hp = _load_best_hparams(cfg)

    rows = []
    task_models = []
    if "snn" in args.models:
        task_models.append("snn")
    if "rf" in args.models:
        task_models.append("random_forest")
    if "mlp" in args.models:
        task_models.append("mlp")

    tasks = [(s, m) for s in args.seeds for m in task_models]
    total_tasks = len(tasks)
    if total_tasks == 0:
        raise RuntimeError("Nenhum modelo foi selecionado.")

    t0 = time.time()
    completed = 0

    for seed, model_name in tasks:
        completed += 1
        elapsed = time.time() - t0
        avg = elapsed / max(completed - 1, 1)
        eta = avg * (total_tasks - completed) if completed > 1 else 0.0

        print(
            f"\n[{completed}/{total_tasks}] seed={seed} model={model_name} | "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )
        step_t0 = time.time()

        if model_name == "snn":
            y_true, y_pred = run_snn(cfg, seed, device, hp)
            rec = {"seed": seed, "model": "snn", **_metrics(y_true, y_pred, cfg.target_cols)}
        elif model_name == "random_forest":
            y_true, y_pred = run_rf(cfg, seed)
            rec = {"seed": seed, "model": "random_forest", **_metrics(y_true, y_pred, cfg.target_cols)}
        else:
            y_true, y_pred = run_mlp(cfg, seed)
            rec = {"seed": seed, "model": "mlp", **_metrics(y_true, y_pred, cfg.target_cols)}

        rows.append(rec)
        step_dt = time.time() - step_t0
        print(
            f"{model_name}: mae_macro={rec['mae_macro']:.4f} "
            f"rmse_macro={rec['rmse_macro']:.4f} r2_macro={rec['r2_macro']:.4f} "
            f"| step={step_dt:.1f}s"
        )

    summary = summarize(rows, cfg.target_cols)
    raw_path, summary_path, md_path, json_path = write_outputs(
        cfg, rows, summary, cfg.target_cols
    )
    print("\nArquivos gerados:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {md_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
