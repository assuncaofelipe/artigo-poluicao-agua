"""
Benchmark ESP32 SNN - Captura métricas de latência via serial

Similar ao bench_tflite.py mas para ESP32.
Captura métricas de inferência da SNN rodando no microcontrolador.

Uso:
    python bench_esp32.py           # Usa COM4 por padrão
    python bench_esp32.py COM3      # Especifica porta
    python bench_esp32.py --reset   # Reseta ESP32 antes de capturar
"""

import json
import sys
import csv
import re
import time
from datetime import datetime

try:
    import serial
except ImportError:
    raise SystemExit("pyserial não encontrado. Instale com: pip install pyserial")


def find_json_in_output(lines: list[str]) -> dict | None:
    """Extrai JSON de benchmark do output do ESP32."""
    in_json_block = False
    json_line = None
    
    for line in lines:
        if "<<<BENCHMARK_JSON>>>" in line:
            in_json_block = True
            continue
        if "<<<END_BENCHMARK_JSON>>>" in line:
            in_json_block = False
            continue
        if in_json_block and line.strip().startswith("{"):
            json_line = line.strip()
            break
    
    if json_line:
        try:
            return json.loads(json_line)
        except json.JSONDecodeError as e:
            print(f"Erro ao parsear JSON: {e}")
            print(f"Linha: {json_line}")
    
    return None


def extract_mae_metrics(lines: list[str]) -> dict:
    """Extrai métricas MAE do output."""
    mae = {}
    for line in lines:
        if "pH MAE:" in line:
            match = re.search(r"pH MAE:\s+([\d.]+)", line)
            if match:
                mae["ph_mae"] = float(match.group(1))
        elif "OD MAE:" in line:
            match = re.search(r"OD MAE:\s+([\d.]+)", line)
            if match:
                mae["od_mae"] = float(match.group(1))
        elif "Turbidity MAE:" in line:
            match = re.search(r"Turbidity MAE:\s+([\d.]+)", line)
            if match:
                mae["turb_mae"] = float(match.group(1))
        elif "Total spikes:" in line:
            match = re.search(r"Total spikes:\s+(\d+)", line)
            if match:
                mae["total_spikes"] = int(match.group(1))
            match = re.search(r"([\d.]+)% firing rate", line)
            if match:
                mae["firing_rate"] = float(match.group(1))
    return mae


def capture_esp32_benchmark(port: str = "COM4", baudrate: int = 115200, 
                            timeout: float = 120.0, reset: bool = False) -> dict | None:
    """
    Captura benchmark do ESP32 via serial.
    
    Args:
        port: Porta serial (ex: COM4, /dev/ttyUSB0)
        baudrate: Taxa de transmissão
        timeout: Tempo máximo de espera em segundos
        reset: Se True, reseta ESP32 via DTR/RTS
        
    Returns:
        Dict com métricas ou None se falhar
    """
    print(f"Conectando a {port} @ {baudrate}...")
    
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1.0,
            rtscts=False,
            dsrdtr=False
        )
    except serial.SerialException as e:
        print(f"Erro ao abrir porta {port}: {e}")
        return None
    
    if reset:
        print("Resetando ESP32...")
        ser.dtr = False
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.5)
    
    print(f"Aguardando benchmark (timeout: {timeout}s)...")
    print("-" * 50)
    
    lines = []
    start_time = time.time()
    found_benchmark = False
    
    try:
        while (time.time() - start_time) < timeout:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print(line)
                        lines.append(line)
                        
                        if "<<<END_BENCHMARK_JSON>>>" in line:
                            found_benchmark = True
                            # Continua capturando para pegar MAE
                        
                        # Termina após capturar stats de spikes
                        if found_benchmark and "firing rate" in line:
                            print("-" * 50)
                            print("Benchmark completo!")
                            break
                            
                except UnicodeDecodeError:
                    pass
            else:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        ser.close()
    
    if not found_benchmark:
        print("Benchmark não encontrado no output")
        return None
    
    # Extrai métricas
    benchmark_data = find_json_in_output(lines)
    if not benchmark_data:
        print("Não foi possível extrair JSON do benchmark")
        return None
    
    # Adiciona métricas MAE
    mae_metrics = extract_mae_metrics(lines)
    benchmark_data.update(mae_metrics)
    
    # Adiciona timestamp
    benchmark_data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    
    return benchmark_data


def save_to_csv(data: dict, csv_path: str = "results_esp32.csv"):
    """Salva resultados em CSV."""
    fieldnames = [
        "device", "cores", "cpu_mhz", "features", "hidden", "outputs",
        "time_steps", "latency_mean_ms", "latency_std_ms", "throughput",
        "ph_mae", "od_mae", "turb_mae", "total_spikes", "firing_rate",
        "warmup_runs", "benchmark_runs", "timestamp", "notes"
    ]
    
    # Garante que todos os campos existem
    row = {k: data.get(k, "") for k in fieldnames}
    row["notes"] = ""
    
    write_header = False
    try:
        with open(csv_path, "r", encoding="utf-8"):
            pass
    except FileNotFoundError:
        write_header = True
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"CSV atualizado: {csv_path}")


def main():
    # Parse argumentos simples
    port = "COM4"
    reset = False
    
    for arg in sys.argv[1:]:
        if arg == "--reset":
            reset = True
        elif arg.startswith("COM") or arg.startswith("/dev/"):
            port = arg
        elif arg in ["-h", "--help"]:
            print(__doc__)
            return
    
    print("=== Benchmark ESP32 SNN ===")
    print(f"Porta: {port}")
    print(f"Reset: {reset}")
    print()
    
    # Captura benchmark
    data = capture_esp32_benchmark(port=port, reset=reset)
    
    if data:
        print("\n=== RESULTADOS ===")
        print(f"Device: {data.get('device', 'N/A')}")
        print(f"Cores: {data.get('cores', 'N/A')}")
        print(f"CPU: {data.get('cpu_mhz', 'N/A')} MHz")
        print(f"Model: {data.get('features', 'N/A')} -> {data.get('hidden', 'N/A')} -> {data.get('outputs', 'N/A')}")
        print(f"Time steps: {data.get('time_steps', 'N/A')}")
        print(f"Latency mean: {data.get('latency_mean_ms', 'N/A'):.4f} ms/sample")
        print(f"Latency std: {data.get('latency_std_ms', 'N/A'):.4f} ms")
        print(f"Throughput: {data.get('throughput', 'N/A'):.2f} samples/s")
        if 'ph_mae' in data:
            print(f"pH MAE: {data['ph_mae']:.4f}")
            print(f"OD MAE: {data['od_mae']:.4f} mg/L")
            print(f"Turb MAE: {data['turb_mae']:.4f} NTU")
        if 'firing_rate' in data:
            print(f"Firing rate: {data['firing_rate']:.2f}%")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        
        # Salva CSV
        save_to_csv(data)
        
        # Salva JSON completo
        json_path = "results_esp32.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"JSON salvo: {json_path}")
        
    else:
        print("Falha ao capturar benchmark")
        sys.exit(1)


if __name__ == "__main__":
    main()
