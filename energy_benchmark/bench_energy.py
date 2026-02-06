#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except Exception:  # pragma: no cover - fallback for full TF if needed
    import tensorflow.lite as tflite  # type: ignore


@dataclass
class PowerSample:
    ts: float
    current_mA: float
    voltage_V: float


def _read_number(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def _convert_units(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "a":
        return value * 1000.0
    if unit == "ma":
        return value
    if unit == "ua":
        return value / 1000.0
    raise ValueError(f"Unsupported unit: {unit}")


def _find_sysfs_paths() -> Tuple[Optional[str], Optional[str]]:
    base = "/sys/class/power_supply"
    if not os.path.isdir(base):
        return None, None
    for name in os.listdir(base):
        candidate = os.path.join(base, name)
        cur = os.path.join(candidate, "current_now")
        volt = os.path.join(candidate, "voltage_now")
        if os.path.isfile(cur) and os.path.isfile(volt):
            return cur, volt
    return None, None


def _load_norm(path: str) -> Tuple[List[str], List[float], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["feature_cols"], data["mean"], data["std"]


def _build_input(feature_cols: List[str]) -> np.ndarray:
    # Input mínimo e determinístico: zeros normalizados.
    x = np.zeros((1, len(feature_cols)), dtype=np.float32)
    return x


def _run_inference(
    model_path: str,
    norm_path: str,
    runs: int,
    warmup: int,
) -> Tuple[float, int]:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    feature_cols, mean, std = _load_norm(norm_path)
    x = _build_input(feature_cols)
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = x.astype(np.float32)

    for _ in range(warmup):
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_detail["index"])

    t0 = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_detail["index"])
    t1 = time.perf_counter()
    return (t1 - t0), runs


def _sample_power_sysfs(
    samples: List[PowerSample],
    stop_event: threading.Event,
    current_path: str,
    voltage_path: str,
    current_unit: str,
    voltage_unit: str,
    interval_s: float,
) -> None:
    while not stop_event.is_set():
        try:
            cur_raw = _read_number(current_path)
            volt_raw = _read_number(voltage_path)
            cur_mA = _convert_units(cur_raw, current_unit)
            # voltage conversion to volts
            if voltage_unit.lower() == "v":
                volt_V = volt_raw
            elif voltage_unit.lower() == "mv":
                volt_V = volt_raw / 1000.0
            elif voltage_unit.lower() == "uv":
                volt_V = volt_raw / 1_000_000.0
            else:
                raise ValueError(f"Unsupported voltage unit: {voltage_unit}")
            samples.append(PowerSample(time.time(), cur_mA, volt_V))
        except Exception:
            pass
        time.sleep(interval_s)


def _sample_power_csv(
    samples: List[PowerSample],
    stop_event: threading.Event,
    csv_path: str,
    interval_s: float,
) -> None:
    # CSV esperado: timestamp,current_mA,voltage_V
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    idx = 0
    n = len(rows)
    while not stop_event.is_set() and n > 0:
        row = rows[idx % n]
        try:
            samples.append(
                PowerSample(
                    time.time(),
                    float(row["current_mA"]),
                    float(row["voltage_V"]),
                )
            )
        except Exception:
            pass
        idx += 1
        time.sleep(interval_s)


def _sample_power_manual(
    samples: List[PowerSample],
    stop_event: threading.Event,
    current_mA: float,
    voltage_V: float,
    interval_s: float,
) -> None:
    while not stop_event.is_set():
        samples.append(PowerSample(time.time(), current_mA, voltage_V))
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de energia no Raspberry Pi (TFLite).")
    parser.add_argument("--model", default="model.tflite")
    parser.add_argument("--norm", default="norm_multi.json")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument("--provider", choices=["sysfs", "csv", "manual"], default="sysfs")
    parser.add_argument("--current-path", default="")
    parser.add_argument("--voltage-path", default="")
    parser.add_argument("--current-unit", default="uA", help="A, mA ou uA (padrão uA)")
    parser.add_argument("--voltage-unit", default="uV", help="V, mV ou uV (padrão uV)")
    parser.add_argument("--csv", default="", help="CSV com colunas timestamp,current_mA,voltage_V")
    parser.add_argument("--current-mA", type=float, default=0.0)
    parser.add_argument("--voltage-V", type=float, default=5.0)
    parser.add_argument("--out", default="results_energy.csv")
    args = parser.parse_args()

    interval_s = max(args.interval_ms / 1000.0, 0.01)

    if args.provider == "sysfs":
        cur_path, volt_path = _find_sysfs_paths()
        if args.current_path:
            cur_path = args.current_path
        if args.voltage_path:
            volt_path = args.voltage_path
        if not cur_path or not volt_path:
            raise SystemExit("Não foi possível localizar current_now/voltage_now em /sys/class/power_supply.")
        sampler = ("sysfs", cur_path, volt_path)
    elif args.provider == "csv":
        if not args.csv:
            raise SystemExit("Informe --csv com o arquivo de medições.")
        sampler = ("csv", args.csv)
    else:
        if args.current_mA <= 0.0:
            raise SystemExit("Informe --current-mA para o modo manual.")
        sampler = ("manual", args.current_mA, args.voltage_V)

    samples: List[PowerSample] = []
    stop_event = threading.Event()

    if sampler[0] == "sysfs":
        _, cur_path, volt_path = sampler
        t = threading.Thread(
            target=_sample_power_sysfs,
            args=(samples, stop_event, cur_path, volt_path, args.current_unit, args.voltage_unit, interval_s),
            daemon=True,
        )
    elif sampler[0] == "csv":
        _, csv_path = sampler
        t = threading.Thread(
            target=_sample_power_csv,
            args=(samples, stop_event, csv_path, interval_s),
            daemon=True,
        )
    else:
        _, cur_mA, volt_V = sampler
        t = threading.Thread(
            target=_sample_power_manual,
            args=(samples, stop_event, cur_mA, volt_V, interval_s),
            daemon=True,
        )

    t.start()
    total_time_s, runs = _run_inference(args.model, args.norm, args.runs, args.warmup)
    stop_event.set()
    t.join(timeout=1.0)

    if not samples:
        raise SystemExit("Sem amostras de corrente/tensão. Verifique o provider.")

    avg_current_mA = float(np.mean([s.current_mA for s in samples]))
    avg_voltage_V = float(np.mean([s.voltage_V for s in samples]))
    power_mW = avg_current_mA * avg_voltage_V
    energy_mJ = power_mW * total_time_s
    energy_per_inf_mJ = energy_mJ / runs

    device = "Raspberry Pi 4B"
    timestamp = datetime.now().isoformat(timespec="seconds")

    print("=== Benchmark Energia (Raspberry Pi) ===")
    print(f"Device: {device}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Samples: {len(samples)}")
    print(f"Current avg: {avg_current_mA:.3f} mA")
    print(f"Voltage avg: {avg_voltage_V:.3f} V")
    print(f"Power avg: {power_mW:.3f} mW")
    print(f"Inference time: {total_time_s:.3f} s ({runs} runs)")
    print(f"Energy total: {energy_mJ:.3f} mJ")
    print(f"Energy per inference: {energy_per_inf_mJ:.6f} mJ/inf")
    print(f"Timestamp: {timestamp}")

    header = [
        "device",
        "os",
        "python",
        "samples",
        "current_mA",
        "voltage_V",
        "power_mW",
        "total_time_s",
        "runs",
        "energy_mJ",
        "energy_per_inf_mJ",
        "timestamp",
        "notes",
    ]
    row = [
        device,
        platform.platform(),
        platform.python_version(),
        len(samples),
        f"{avg_current_mA:.6f}",
        f"{avg_voltage_V:.6f}",
        f"{power_mW:.6f}",
        f"{total_time_s:.6f}",
        runs,
        f"{energy_mJ:.6f}",
        f"{energy_per_inf_mJ:.9f}",
        timestamp,
        "",
    ]

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"CSV atualizado: {args.out}")


if __name__ == "__main__":
    main()
