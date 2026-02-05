import argparse
import csv
import json
import math
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(
        description="Calcula significancia estatistica pareada a partir do benchmark multiseed."
    )
    p.add_argument(
        "--input",
        default="./outputs/benchmark_multiseed_raw.csv",
        help="CSV bruto gerado pelo benchmark_multiseed.py",
    )
    p.add_argument(
        "--output",
        default="./outputs/benchmark_significance.json",
        help="Arquivo JSON de saida com resultados estatisticos.",
    )
    p.add_argument(
        "--reference",
        default="snn",
        help="Modelo de referencia para comparar com os demais.",
    )
    return p.parse_args()


def _comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def sign_test_two_sided(values_a, values_b):
    diffs = [a - b for a, b in zip(values_a, values_b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "p_value": 1.0, "pos": pos, "neg": neg}
    k = min(pos, neg)
    p = 2.0 * sum(_comb(n, i) for i in range(k + 1)) / (2**n)
    p = min(p, 1.0)
    return {"n": n, "p_value": p, "pos": pos, "neg": neg}


def try_wilcoxon(values_a, values_b):
    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat, p = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
        return {"available": True, "stat": float(stat), "p_value": float(p)}
    except Exception:
        return {"available": False, "stat": None, "p_value": None}


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        out = {"seed": int(r["seed"]), "model": r["model"]}
        for k, v in r.items():
            if k in {"seed", "model"}:
                continue
            out[k] = float(v)
        parsed.append(out)
    return parsed


def main():
    args = parse_args()
    rows = load_rows(args.input)
    by_model_seed = defaultdict(dict)
    for r in rows:
        by_model_seed[r["model"]][r["seed"]] = r

    if args.reference not in by_model_seed:
        raise ValueError(f"Modelo de referencia nao encontrado: {args.reference}")

    ref = by_model_seed[args.reference]
    metrics = ["mae_macro", "rmse_macro", "r2_macro"]
    out = {"reference": args.reference, "comparisons": []}

    for model, model_data in by_model_seed.items():
        if model == args.reference:
            continue
        common_seeds = sorted(set(ref.keys()) & set(model_data.keys()))
        comp = {"model_a": args.reference, "model_b": model, "seeds": common_seeds, "metrics": {}}
        for m in metrics:
            a_vals = [ref[s][m] for s in common_seeds]
            b_vals = [model_data[s][m] for s in common_seeds]
            sign = sign_test_two_sided(a_vals, b_vals)
            wilc = try_wilcoxon(a_vals, b_vals)
            mean_a = sum(a_vals) / len(a_vals)
            mean_b = sum(b_vals) / len(b_vals)
            if m == "r2_macro":
                better = args.reference if mean_a > mean_b else model
            else:
                better = args.reference if mean_a < mean_b else model
            comp["metrics"][m] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "better": better,
                "sign_test": sign,
                "wilcoxon": wilc,
            }
        out["comparisons"].append(comp)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"saved: {args.output}")
    for comp in out["comparisons"]:
        print(f"\n{comp['model_a']} vs {comp['model_b']}")
        for m in metrics:
            mm = comp["metrics"][m]
            print(
                f"- {m}: mean_a={mm['mean_a']:.4f} mean_b={mm['mean_b']:.4f} "
                f"sign_p={mm['sign_test']['p_value']:.4f} "
                f"wilcoxon_p={mm['wilcoxon']['p_value']}"
            )


if __name__ == "__main__":
    main()
