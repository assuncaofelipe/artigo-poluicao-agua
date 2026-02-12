#!/usr/bin/env python3
"""
Benchmark comparativo: PyTorch (.pt) vs Python Puro

Este script compara a performance de inferência entre:
1. PyTorch: carrega o .pt e usa o modelo treinado diretamente
2. Python Puro: usa a implementação snn_pure.py (sem dependências)

Execute em um PC ou Raspberry Pi que tenha PyTorch instalado.
Na ESP32, rode apenas o snn_pure.py ou esp32_micropython/main.py.

Uso:
    python benchmark_pytorch_vs_pure.py
"""

import sys
import os
import time
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..", "..", "scripts_multimodal"))

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

WARMUP = 100
RUNS = 5000
SAMPLE_RUNS = 500

# Entrada de teste típica do dataset
TEST_INPUT = [335575.0, 9657300.0, 32.0, 90.0, 30.0]

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def measure_time_us():
    """Retorna tempo em microsegundos"""
    return int(time.perf_counter() * 1_000_000)


def compute_stats(samples):
    """Calcula estatísticas de uma lista de amostras"""
    n = len(samples)
    mean = sum(samples) / n
    variance = sum((s - mean) ** 2 for s in samples) / n
    std = variance ** 0.5
    return {
        "mean": mean,
        "std": std,
        "min": min(samples),
        "max": max(samples)
    }


# ==============================================================================
# BENCHMARK PYTORCH
# ==============================================================================

def benchmark_pytorch():
    """Benchmark usando PyTorch diretamente"""
    try:
        import torch
        from config import Config
        from model import SNNRegressor
        from encoding import rate_encode
    except ImportError as e:
        print(f"PyTorch não disponível: {e}")
        return None
    
    cfg = Config()
    checkpoint_path = os.path.join(script_dir, "..", "scripts_multimodal", "outputs", "grid_best_multi.pt")
    norm_path = os.path.join(script_dir, "..", "scripts_multimodal", "outputs", "grid_norm_multi.json")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint não encontrado: {checkpoint_path}")
        return None
    
    # Carrega normalização
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    
    # Carrega modelo
    model = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=len(cfg.target_cols),
        beta=cfg.beta,
        thresh=cfg.thresh,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    
    # Prepara entrada
    x_mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    x_std = torch.tensor(norm_stats["std"], dtype=torch.float32)
    y_mean = torch.tensor(norm_stats["y_mean"], dtype=torch.float32)
    y_std = torch.tensor(norm_stats["y_std"], dtype=torch.float32)
    
    raw_input = torch.tensor([TEST_INPUT], dtype=torch.float32)
    norm_input = (raw_input - x_mean) / x_std
    
    # Encoding
    spikes = rate_encode(norm_input, cfg.time_steps)  # [T, B, F]
    
    print("\n=== BENCHMARK PYTORCH ===")
    print(f"Modelo: {checkpoint_path}")
    print(f"Device: CPU")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}, Samples: {SAMPLE_RUNS}")
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(spikes)
    
    # Benchmark throughput
    t0 = measure_time_us()
    with torch.no_grad():
        for _ in range(RUNS):
            _ = model(spikes)
    t1 = measure_time_us()
    
    total_us = t1 - t0
    total_ms = total_us / 1000.0
    latency_mean = total_ms / RUNS
    throughput = RUNS / (total_ms / 1000.0)
    
    # Coleta amostras
    samples = []
    with torch.no_grad():
        for _ in range(SAMPLE_RUNS):
            s0 = measure_time_us()
            _ = model(spikes)
            s1 = measure_time_us()
            samples.append((s1 - s0) / 1000.0)
    
    stats = compute_stats(samples)
    
    # Predição final
    with torch.no_grad():
        norm_out = model(spikes)
        pred = (norm_out * y_std + y_mean).squeeze().tolist()
    
    results = {
        "backend": "PyTorch",
        "latency_mean_ms": latency_mean,
        "latency_std_ms": stats["std"],
        "latency_min_ms": stats["min"],
        "latency_max_ms": stats["max"],
        "throughput_inf_s": throughput,
        "predictions": {
            "pH": pred[0],
            "OD": pred[1],
            "Turbidez": pred[2]
        }
    }
    
    return results


# ==============================================================================
# BENCHMARK PYTHON PURO
# ==============================================================================

def benchmark_pure():
    """Benchmark usando Python puro"""
    # Importa o módulo snn_pure
    sys.path.insert(0, script_dir)
    import snn_pure
    
    # Carrega modelo do JSON
    json_path = os.path.join(script_dir, "..", "esp32_micropython", "model_weights.json")
    if not os.path.exists(json_path):
        print(f"JSON não encontrado: {json_path}")
        print("Execute primeiro: python export_pt_to_json.py")
        return None
    
    snn_pure.load_model_from_json(json_path)
    norm_input = snn_pure.normalize_input(TEST_INPUT)
    
    print("\n=== BENCHMARK PYTHON PURO ===")
    print(f"Modelo: {json_path}")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}, Samples: {SAMPLE_RUNS}")
    
    # Warmup
    for _ in range(WARMUP):
        snn_pure.snn_inference_pure(norm_input)
    
    # Benchmark throughput
    t0 = measure_time_us()
    for _ in range(RUNS):
        snn_pure.snn_inference_pure(norm_input)
    t1 = measure_time_us()
    
    total_us = t1 - t0
    total_ms = total_us / 1000.0
    latency_mean = total_ms / RUNS
    throughput = RUNS / (total_ms / 1000.0)
    
    # Coleta amostras
    samples = []
    for _ in range(SAMPLE_RUNS):
        s0 = measure_time_us()
        snn_pure.snn_inference_pure(norm_input)
        s1 = measure_time_us()
        samples.append((s1 - s0) / 1000.0)
    
    stats = compute_stats(samples)
    
    # Predição final
    pred, spikes = snn_pure.predict(TEST_INPUT)
    
    results = {
        "backend": "Python Puro",
        "latency_mean_ms": latency_mean,
        "latency_std_ms": stats["std"],
        "latency_min_ms": stats["min"],
        "latency_max_ms": stats["max"],
        "throughput_inf_s": throughput,
        "total_spikes": spikes,
        "predictions": {
            "pH": pred[0],
            "OD": pred[1],
            "Turbidez": pred[2]
        }
    }
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("BENCHMARK: PyTorch (.pt) vs Python Puro")
    print("=" * 60)
    print(f"\nEntrada de teste: {TEST_INPUT}")
    print(f"Features: [x, y, temp, cor_aparente, cor_verdadeira]")
    
    results = {}
    
    # Benchmark PyTorch
    pytorch_results = benchmark_pytorch()
    if pytorch_results:
        results["pytorch"] = pytorch_results
        print(f"\n--- PyTorch ---")
        print(f"Latência: {pytorch_results['latency_mean_ms']:.4f} ± {pytorch_results['latency_std_ms']:.4f} ms")
        print(f"Throughput: {pytorch_results['throughput_inf_s']:.1f} inf/s")
        print(f"Predições: pH={pytorch_results['predictions']['pH']:.4f}, OD={pytorch_results['predictions']['OD']:.4f}, Turb={pytorch_results['predictions']['Turbidez']:.4f}")
    
    # Benchmark Python Puro
    pure_results = benchmark_pure()
    if pure_results:
        results["pure"] = pure_results
        print(f"\n--- Python Puro ---")
        print(f"Latência: {pure_results['latency_mean_ms']:.4f} ± {pure_results['latency_std_ms']:.4f} ms")
        print(f"Throughput: {pure_results['throughput_inf_s']:.1f} inf/s")
        print(f"Predições: pH={pure_results['predictions']['pH']:.4f}, OD={pure_results['predictions']['OD']:.4f}, Turb={pure_results['predictions']['Turbidez']:.4f}")
    
    # Comparação
    if pytorch_results and pure_results:
        print("\n" + "=" * 60)
        print("COMPARAÇÃO")
        print("=" * 60)
        
        pt_lat = pytorch_results['latency_mean_ms']
        pure_lat = pure_results['latency_mean_ms']
        
        ratio = pure_lat / pt_lat
        if ratio > 1:
            print(f"Python Puro é {ratio:.2f}x MAIS LENTO que PyTorch")
        else:
            print(f"Python Puro é {1/ratio:.2f}x MAIS RÁPIDO que PyTorch")
        
        print(f"\nLatência PyTorch:     {pt_lat:.4f} ms")
        print(f"Latência Python Puro: {pure_lat:.4f} ms")
        print(f"Diferença:            {pure_lat - pt_lat:.4f} ms ({(ratio-1)*100:+.1f}%)")
        
        # Verifica consistência de predições
        print("\n--- Consistência das Predições ---")
        for key in ["pH", "OD", "Turbidez"]:
            pt_val = pytorch_results['predictions'][key]
            pure_val = pure_results['predictions'][key]
            diff = abs(pt_val - pure_val)
            print(f"{key}: PyTorch={pt_val:.4f}, Pure={pure_val:.4f}, Δ={diff:.6f}")
    
    # Salva resultados
    output_path = os.path.join(script_dir, "benchmark_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✔ Resultados salvos em: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
