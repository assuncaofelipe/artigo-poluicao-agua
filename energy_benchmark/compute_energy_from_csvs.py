#!/usr/bin/env python3
import argparse
import csv
from statistics import mean, pstdev


def load_csv(path: str):
    currents = []
    voltages = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            currents.append(float(row["current_mA"]) / 1000.0)
            voltages.append(float(row["voltage_V"]))
    if not currents:
        raise SystemExit(f"Sem amostras em {path}")
    return currents, voltages


def main():
    ap = argparse.ArgumentParser(description="Calcula energia por inferência a partir de CSVs idle/load.")
    ap.add_argument("--idle", required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--voltage", type=float, default=5.0, help="Tensão fixa usada (V)")
    ap.add_argument("--throughput", type=float, required=True, help="amostras/segundo")
    ap.add_argument("--out", default="results_energy_manual.csv")
    args = ap.parse_args()

    idle_i, _ = load_csv(args.idle)
    load_i, _ = load_csv(args.load)

    idle_i_mean = mean(idle_i)
    load_i_mean = mean(load_i)
    idle_i_std = pstdev(idle_i)
    load_i_std = pstdev(load_i)

    delta_i = max(load_i_mean - idle_i_mean, 0.0)
    idle_p = idle_i_mean * args.voltage
    load_p = load_i_mean * args.voltage
    delta_p = delta_i * args.voltage

    energy_per_inf_j = delta_p / args.throughput
    energy_per_inf_mj = energy_per_inf_j * 1000.0

    header = [
        "idle_i_A",
        "load_i_A",
        "idle_i_std_A",
        "load_i_std_A",
        "idle_p_W",
        "load_p_W",
        "delta_p_W",
        "throughput_samples_s",
        "energy_per_inf_mJ",
    ]
    row = [
        f"{idle_i_mean:.6f}",
        f"{load_i_mean:.6f}",
        f"{idle_i_std:.6f}",
        f"{load_i_std:.6f}",
        f"{idle_p:.6f}",
        f"{load_p:.6f}",
        f"{delta_p:.6f}",
        f"{args.throughput:.2f}",
        f"{energy_per_inf_mj:.6f}",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    print("Energia por inferência (mJ):", f"{energy_per_inf_mj:.6f}")
    print("CSV:", args.out)


if __name__ == "__main__":
    main()
