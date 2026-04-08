#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_MATRICES: Dict[str, dict] = {
    "xgb_clean.png": {
        "title": "XGBoost",
        "matrix": [[201, 62], [8, 20]],
    },
    "lgbm_clean.png": {
        "title": "LightGBM",
        "matrix": [[198, 65], [5, 23]],
    },
    "rf_clean.png": {
        "title": "Random Forest",
        "matrix": [[152, 111], [6, 22]],
    },
}


def load_config(path: Path | None) -> Dict[str, dict]:
    if path is None:
        return DEFAULT_MATRICES
    return json.loads(path.read_text(encoding="utf-8"))


def apply_article_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    sns.set_style("white")


def render_confusion_matrix(output_path: Path, title: str, matrix: List[List[int]]) -> None:
    cm = np.array(matrix)

    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    cmap = sns.light_palette("#1f4e79", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        square=True,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title(title, pad=8)
    ax.set_xlabel("Predito", labelpad=8)
    ax.set_ylabel("Valor real", labelpad=8)
    ax.set_xticklabels(["Não surto", "Surto"], rotation=0)
    ax.set_yticklabels(["Não surto", "Surto"], rotation=0)
    ax.tick_params(axis="both", length=0)

    # Improve contrast of annotations cell-by-cell.
    threshold = cm.max() / 2
    for text, value in zip(ax.texts, cm.flatten()):
        text.set_color("white" if value > threshold else "black")

    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready confusion matrices in Portuguese."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file mapping output filenames to {title, matrix}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/confusion_clean"),
        help="Directory where images will be saved.",
    )
    args = parser.parse_args()

    apply_article_style()
    config = load_config(args.config)
    for filename, spec in config.items():
        render_confusion_matrix(args.output_dir / filename, spec["title"], spec["matrix"])
        print(f"saved {args.output_dir / filename}")


if __name__ == "__main__":
    main()
