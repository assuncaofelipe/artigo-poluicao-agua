import json
import time
import platform
from datetime import datetime
import csv
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except Exception as exc:
    raise SystemExit(
        "tflite-runtime nao encontrado. Instale com: pip install -r requirements.txt"
    ) from exc


def main():
    with open("norm_multi.json", "r", encoding="utf-8") as f:
        norm = json.load(f)

    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    n_features = len(norm["feature_cols"])
    x = np.random.randn(1, n_features).astype(np.float32)

    # warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])

    n_runs = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    t1 = time.perf_counter()

    lat_ms = (t1 - t0) * 1000 / n_runs
    throughput = n_runs / (t1 - t0)

    # sample latencies (optional)
    sample_runs = 200
    samples = []
    for _ in range(sample_runs):
        s0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        s1 = time.perf_counter()
        samples.append((s1 - s0) * 1000)

    device = "Raspberry Pi 4B"
    os_name = platform.platform()
    cpu = platform.processor()
    pyver = platform.python_version()
    lat_std = float(np.std(samples))
    ts = datetime.now().isoformat(timespec="seconds")

    print("=== Benchmark TFLite (Raspberry Pi) ===")
    print(f"Device: {device}")
    print(f"OS: {os_name}")
    print(f"CPU: {cpu}")
    print(f"Python: {pyver}")
    print(f"Features: {n_features}")
    print(f"Latency mean: {lat_ms:.3f} ms/amostra")
    print(f"Latency std: {lat_std:.3f} ms")
    print(f"Throughput: {throughput:.2f} amostras/s")
    print(f"Timestamp: {ts}")

    row = {
        "device": device,
        "os": os_name,
        "cpu": cpu,
        "python": pyver,
        "features": n_features,
        "latency_mean_ms": f"{lat_ms:.3f}",
        "latency_std_ms": f"{lat_std:.3f}",
        "throughput_samples_s": f"{throughput:.2f}",
        "date": ts,
        "notes": "",
    }

    csv_path = "results.csv"
    write_header = False
    try:
        with open(csv_path, "r", encoding="utf-8"):
            pass
    except FileNotFoundError:
        write_header = True

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "os",
                "cpu",
                "python",
                "features",
                "latency_mean_ms",
                "latency_std_ms",
                "throughput_samples_s",
                "date",
                "notes",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"CSV atualizado: {csv_path}")


if __name__ == "__main__":
    main()
