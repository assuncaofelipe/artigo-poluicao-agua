#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time


def adb_shell(cmd: str) -> str:
    out = subprocess.check_output(["adb", "shell", cmd], text=True)
    return out.strip()


def read_power():
    # tenta caminhos comuns
    paths = [
        "/sys/class/power_supply/battery/current_now",
        "/sys/class/power_supply/battery/voltage_now",
    ]
    current = adb_shell(f"cat {paths[0]}")
    voltage = adb_shell(f"cat {paths[1]}")
    return current, voltage


def main():
    ap = argparse.ArgumentParser(description="Loga corrente/tensao do Android via adb.")
    ap.add_argument("--out", default="android_power_samples.csv")
    ap.add_argument("--interval", type=float, default=0.5, help="Intervalo (s)")
    ap.add_argument("--duration", type=float, default=30.0, help="Duracao (s)")
    args = ap.parse_args()

    header = ["timestamp", "current_uA", "voltage_uV"]
    start = time.time()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        while True:
            now = time.time()
            if now - start > args.duration:
                break
            try:
                current, voltage = read_power()
                w.writerow([f"{now:.6f}", current, voltage])
            except subprocess.CalledProcessError:
                pass
            time.sleep(args.interval)

    print(f"CSV gerado: {args.out}")


if __name__ == "__main__":
    main()
