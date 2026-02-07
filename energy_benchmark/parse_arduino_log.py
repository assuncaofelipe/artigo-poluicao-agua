#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from datetime import datetime


LINE_RE = re.compile(
    r"(?:^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s*->\s*)?"
    r"Vout:\s*(?P<vout>[0-9.]+)\s*V\s*\|\s*I:\s*(?P<i>[-0-9.]+)\s*A"
)


def main():
    ap = argparse.ArgumentParser(
        description="Converte log textual do Arduino (ACS712) para CSV e gera resumo."
    )
    ap.add_argument("--in", dest="input_path", required=True, help="Arquivo .txt com o log bruto")
    ap.add_argument("--out", default="power_samples_parsed.csv", help="CSV de saída")
    ap.add_argument("--voltage", type=float, default=None, help="Força tensão fixa (V)")
    ap.add_argument("--summary", default="power_summary.csv", help="CSV com resumo")
    args = ap.parse_args()

    rows = []
    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            vout = float(m.group("vout"))
            current_a = float(m.group("i"))
            current_a = abs(current_a)
            timestamp = m.group("time")
            rows.append((timestamp, vout if args.voltage is None else args.voltage, current_a))

    if not rows:
        raise SystemExit("Nenhuma linha válida encontrada no log.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "voltage_V", "current_A", "power_W"])
        for ts, v, i in rows:
            w.writerow([ts or "", f"{v:.6f}", f"{i:.6f}", f"{(v*i):.6f}"])

    currents = [r[2] for r in rows]
    volts = [r[1] for r in rows]
    powers = [v * i for v, i in zip(volts, currents)]

    summary = {
        "n_samples": len(rows),
        "current_mean_A": statistics.fmean(currents),
        "current_std_A": statistics.pstdev(currents),
        "voltage_mean_V": statistics.fmean(volts),
        "power_mean_W": statistics.fmean(powers),
        "power_std_W": statistics.pstdev(powers),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(summary.keys()))
        w.writerow([f"{summary[k]:.6f}" if isinstance(summary[k], float) else summary[k] for k in summary])

    print(f"CSV gerado: {args.out}")
    print(f"Resumo: {args.summary}")
    print(
        f"N={summary['n_samples']} | "
        f"I_mean={summary['current_mean_A']:.3f} A | "
        f"P_mean={summary['power_mean_W']:.3f} W"
    )


if __name__ == "__main__":
    main()
