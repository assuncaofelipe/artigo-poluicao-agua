import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figuras"
BENCH_SUMMARY = ROOT / "scripts_multimodal" / "outputs" / "benchmark_multiseed_summary.csv"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_benchmark_summary(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _model_label(name: str) -> str:
    mapping = {
        "snn": "SNN",
        "random_forest": "RF",
        "mlp": "MLP",
        "lstm": "LSTM",
    }
    return mapping.get(name, name)


def save_mae_bar(rows, out_path: Path, dpi: int = 600):
    rows = sorted(rows, key=lambda r: float(r["mae_macro_mean"]))
    labels = [_model_label(r["model"]) for r in rows]
    maes = [float(r["mae_macro_mean"]) for r in rows]
    stds = [float(r["mae_macro_std"]) for r in rows]

    colors = ["#1f77b4" if l == "SNN" else "#9aa0a6" for l in labels]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, maes, yerr=stds, capsize=4, color=colors, edgecolor="#333")
    plt.ylabel("MAE macro")
    plt.xlabel("Model")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    for b, v in zip(bars, maes):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_energy_platform_bar(out_path: Path, dpi: int = 600):
    # Values used in artigo/results.tex (standardized Raspberry throughput)
    rows = [
        ("ESP32 (MicroPython)", 11.204),
        ("Raspberry Pi 4B", 0.0512),
        ("Android 23021RAAEG", 0.0104),
        ("Android SM-A566E", 0.0151),
        ("Android 2201117SG", 0.00285),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#d55e00", "#0072b2", "#009e73", "#56b4e9", "#66a61e"]

    plt.figure(figsize=(8.2, 4.3))
    bars = plt.bar(range(len(labels)), vals, color=colors, edgecolor="#333")
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=18, ha="right")
    plt.ylabel("Energy per inference (mJ, log scale)")
    plt.grid(axis="y", which="both", alpha=0.25, linestyle="--")
    for b, v in zip(bars, vals):
        fmt = f"{v:.3f}" if v >= 1 else f"{v:.4f}"
        plt.text(b.get_x() + b.get_width() / 2, v * 1.12, fmt, ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _radar_angles(n):
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    return angles + angles[:1]


def save_snn_platform_radar(out_path: Path, dpi: int = 600):
    # SNN-only deployment comparison (energy/latency available only for SNN platforms)
    data = {
        "ESP32 (uPy)": {"lat_ms": 199.298, "thr": 5.0, "energy_mj": 11.204},
        "RPi4B": {"lat_ms": 0.214, "thr": 4670.50, "energy_mj": 0.0512},
        "A23021": {"lat_ms": 0.093, "thr": 10780.37, "energy_mj": 0.0104},
        "A566E": {"lat_ms": 0.030, "thr": 33425.81, "energy_mj": 0.0151},
        "2201117SG": {"lat_ms": 0.610, "thr": 16313.16, "energy_mj": 0.00285},
    }
    axes_names = ["Accuracy (SNN)", "Latency (inv.)", "Throughput", "Energy (inv.)"]
    # Accuracy shared for SNN across deployment platforms (same model family) using benchmark macro R2 from paper
    acc_value = 0.291

    # Build normalized metrics in [0,1]
    lat_vals = np.array([v["lat_ms"] for v in data.values()], dtype=float)
    thr_vals = np.array([v["thr"] for v in data.values()], dtype=float)
    en_vals = np.array([v["energy_mj"] for v in data.values()], dtype=float)
    lat_inv = 1.0 / lat_vals
    en_inv = 1.0 / en_vals

    def norm(a):
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-12:
            return np.ones_like(a)
        return (a - mn) / (mx - mn)

    lat_n = norm(lat_inv)
    thr_n = norm(thr_vals)
    en_n = norm(en_inv)
    acc_n = np.full_like(lat_n, acc_value)  # constant SNN accuracy reference

    angles = _radar_angles(len(axes_names))
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=8)
    ax.set_ylim(0, 1.0)

    palette = ["#d55e00", "#0072b2", "#009e73", "#56b4e9", "#66a61e"]
    for (label, _vals), c, a, t, e in zip(data.items(), palette, acc_n, lat_n, en_n):
        # placeholder zip mismatch if thr missing handled below
        pass
    for idx, (label, vals) in enumerate(data.items()):
        series = [float(acc_n[idx]), float(lat_n[idx]), float(thr_n[idx]), float(en_n[idx])]
        series = series + series[:1]
        ax.plot(angles, series, linewidth=1.6, label=label, color=palette[idx % len(palette)])
        ax.fill(angles, series, alpha=0.07, color=palette[idx % len(palette)])

    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    _ensure_dir(OUT_DIR)
    rows = _load_benchmark_summary(BENCH_SUMMARY)
    save_mae_bar(rows, OUT_DIR / "mae_vs_model_bar.png")
    save_energy_platform_bar(OUT_DIR / "energy_platforms_bar.png")
    save_snn_platform_radar(OUT_DIR / "snn_platform_radar.png")
    print("Generated:")
    print(OUT_DIR / "mae_vs_model_bar.png")
    print(OUT_DIR / "energy_platforms_bar.png")
    print(OUT_DIR / "snn_platform_radar.png")


if __name__ == "__main__":
    main()
