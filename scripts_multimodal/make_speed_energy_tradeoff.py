#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    # Values aligned with Table "Comparative summary of energy per inference across platforms".
    points = [
        ("ESP32", 5.0, 11.204),
        ("Raspberry Pi 4B", 4670.50, 0.0512),
        ("Device A", 12337.40, 0.0104),
        ("Device B", 30905.08, 0.0151),
        ("Device C", 16121.71, 0.00285),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for name, throughput, energy in points:
        ax.scatter(throughput, energy, s=130, alpha=0.9)
        ax.annotate(
            name,
            (throughput, energy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )

    # Better region: high throughput + low energy.
    ax.text(
        0.97,
        0.08,
        "Better operating zone\n(faster + lower energy)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Throughput (samples/s, log scale)")
    ax.set_ylabel("Energy per inference (mJ/inf, log scale)")
    ax.set_title("Speed-Energy Trade-off Across Deployment Platforms")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    out_paths = [
        Path("figuras/speed_energy_tradeoff.png"),
        Path("artigo/figuras/speed_energy_tradeoff.png"),
    ]
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("Saved:", ", ".join(str(p) for p in out_paths))


if __name__ == "__main__":
    main()
