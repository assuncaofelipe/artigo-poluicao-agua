"""
Inferência SNN em Python puro - roda em MicroPython (ESP32) ou CPython.

Este módulo NÃO requer PyTorch ou NumPy. Carrega pesos de um arquivo JSON
e faz toda a matemática usando apenas Python puro.

Uso em MicroPython (ESP32):
    import snn_pure
    snn_pure.run_benchmark()

Uso em CPython (PC/Raspberry):
    python snn_pure.py
"""

import math
import time

try:
    import ujson as json
except ImportError:
    import json

# ==============================================================================
# CONFIGURAÇÃO - pode ser sobrescrita ao carregar modelo
# ==============================================================================

INPUT_SIZE = 5
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3
TIME_STEPS = 15
BETA = 0.95
THRESH = 0.8

NORM_MEAN = [335575.28125, 9657301.0, 31.790476, 89.880951, 29.214285]
NORM_STD = [1009.637268, 1739.921753, 0.390003, 25.901505, 13.846785]
OUT_MEAN = [5.622857, 1.725238, 2.527381]
OUT_STD = [0.316531, 0.227631, 1.624802]

# Pesos (serão carregados de arquivo)
W1 = None
B1 = None
W2 = None
B2 = None

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def ticks_us():
    """Retorna tempo em microsegundos (compatível MicroPython/CPython)"""
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.time() * 1_000_000)


def ticks_diff(new, old):
    """Diferença de tempo (compatível MicroPython/CPython)"""
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(new, old)
    return new - old


def sigmoid(x):
    """Função sigmoide para encoding de spikes"""
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ==============================================================================
# CARREGAMENTO DO MODELO
# ==============================================================================

def load_model_from_json(filepath):
    """
    Carrega pesos e configuração de um arquivo JSON.
    
    Args:
        filepath: Caminho para model_weights.json
    """
    global INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    global TIME_STEPS, BETA, THRESH
    global NORM_MEAN, NORM_STD, OUT_MEAN, OUT_STD
    global W1, B1, W2, B2
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    INPUT_SIZE = data["input_size"]
    HIDDEN_SIZE = data["hidden_size"]
    OUTPUT_SIZE = data["output_size"]
    TIME_STEPS = data["time_steps"]
    BETA = data["beta"]
    THRESH = data["thresh"]
    
    NORM_MEAN = data["norm_mean"]
    NORM_STD = data["norm_std"]
    OUT_MEAN = data["out_mean"]
    OUT_STD = data["out_std"]
    
    W1 = data["W1"]
    B1 = data["B1"]
    W2 = data["W2"]
    B2 = data["B2"]
    
    return data


def load_model_from_module():
    """
    Carrega pesos do módulo model_data.py (para MicroPython direto).
    """
    global INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    global TIME_STEPS, BETA, THRESH
    global NORM_MEAN, NORM_STD, OUT_MEAN, OUT_STD
    global W1, B1, W2, B2
    
    from model_data import (
        INPUT_SIZE as IS, HIDDEN_SIZE as HS, OUTPUT_SIZE as OS,
        SNN_TIME_STEPS, SNN_BETA, SNN_THRESHOLD,
        NORM_MEAN as NM, NORM_STD as NS, OUT_MEAN as OM, OUT_STD as OST,
        W1 as W1_mod, B1 as B1_mod, W2 as W2_mod, B2 as B2_mod,
    )
    
    INPUT_SIZE = IS
    HIDDEN_SIZE = HS
    OUTPUT_SIZE = OS
    TIME_STEPS = SNN_TIME_STEPS
    BETA = SNN_BETA
    THRESH = SNN_THRESHOLD
    NORM_MEAN = list(NM)
    NORM_STD = list(NS)
    OUT_MEAN = list(OM)
    OUT_STD = list(OST)
    W1 = [list(row) for row in W1_mod]
    B1 = list(B1_mod)
    W2 = [list(row) for row in W2_mod]
    B2 = list(B2_mod)


# ==============================================================================
# NORMALIZAÇÃO
# ==============================================================================

def normalize_input(raw):
    """Normaliza entrada usando z-score"""
    return [(raw[i] - NORM_MEAN[i]) / NORM_STD[i] for i in range(INPUT_SIZE)]


def denormalize_output(norm_out):
    """Desnormaliza saída"""
    return [norm_out[i] * OUT_STD[i] + OUT_MEAN[i] for i in range(OUTPUT_SIZE)]


# ==============================================================================
# INFERÊNCIA SNN (PYTHON PURO)
# ==============================================================================

def snn_inference_pure(norm_input):
    """
    Executa inferência SNN completa em Python puro.
    Replica exatamente o comportamento do modelo PyTorch.
    
    Args:
        norm_input: Lista com entrada normalizada [5 valores]
        
    Returns:
        (output, total_spikes): Saída normalizada e contagem de spikes
    """
    # Estados
    membrane = [0.0] * HIDDEN_SIZE
    output_accum = [0.0] * OUTPUT_SIZE
    total_spikes = 0
    
    # Buffers
    spike_input = [0.0] * INPUT_SIZE
    hidden_spikes = [0.0] * HIDDEN_SIZE
    
    for t in range(TIME_STEPS):
        # 1. Encoding: gera spikes de entrada usando sigmoide
        for i in range(INPUT_SIZE):
            spike_input[i] = sigmoid(norm_input[i])
        
        # 2. Camada hidden: fc1 + LIF
        for h in range(HIDDEN_SIZE):
            # Corrente sináptica: I = W1 * spike_input + B1
            i_syn = B1[h]
            for i in range(INPUT_SIZE):
                i_syn += W1[h][i] * spike_input[i]
            
            # Neurônio LIF: v = beta * v + I
            v = BETA * membrane[h] + i_syn
            
            # Disparo: se v > thresh, gera spike
            if v > THRESH:
                spike = 1.0
                total_spikes += 1
                v -= THRESH  # Reset por subtração
            else:
                spike = 0.0
            
            membrane[h] = v
            hidden_spikes[h] = spike
        
        # 3. Camada de saída: fc2 (acumula ao longo do tempo)
        for o in range(OUTPUT_SIZE):
            acc = B2[o]
            for h in range(HIDDEN_SIZE):
                acc += W2[o][h] * hidden_spikes[h]
            output_accum[o] += acc
    
    # Média temporal
    output = [output_accum[o] / TIME_STEPS for o in range(OUTPUT_SIZE)]
    
    return output, total_spikes


def predict(raw_input):
    """
    Faz predição completa: normaliza -> inferência -> desnormaliza.
    
    Args:
        raw_input: Lista com 5 valores [x, y, temp, cor_a, cor_v]
        
    Returns:
        (predictions, spikes): [pH, OD, Turbidez] e contagem de spikes
    """
    norm_in = normalize_input(raw_input)
    norm_out, spikes = snn_inference_pure(norm_in)
    predictions = denormalize_output(norm_out)
    return predictions, spikes


# ==============================================================================
# BENCHMARK
# ==============================================================================

def run_benchmark(warmup=10, runs=1000, sample_runs=200, test_input=None):
    """
    Executa benchmark de latência/throughput.
    
    Args:
        warmup: Iterações de aquecimento
        runs: Iterações para medir throughput
        sample_runs: Amostras para estatísticas de latência
        test_input: Entrada de teste (usa padrão se None)
        
    Returns:
        dict com métricas
    """
    if W1 is None:
        raise RuntimeError("Modelo não carregado! Use load_model_from_json() ou load_model_from_module()")
    
    if test_input is None:
        # Valores típicos do dataset
        test_input = [335575.0, 9657300.0, 32.0, 90.0, 30.0]
    
    norm_input = normalize_input(test_input)
    
    print("\n=== BENCHMARK SNN PYTHON PURO ===")
    print(f"Arquitetura: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print(f"Time steps: {TIME_STEPS}")
    print(f"Beta: {BETA}, Thresh: {THRESH}")
    print(f"Warmup: {warmup}, Runs: {runs}, Samples: {sample_runs}")
    
    # Warmup
    for _ in range(warmup):
        snn_inference_pure(norm_input)
    
    # Benchmark throughput
    t0 = ticks_us()
    for _ in range(runs):
        snn_inference_pure(norm_input)
    t1 = ticks_us()
    
    total_us = ticks_diff(t1, t0)
    total_ms = total_us / 1000.0
    latency_mean_ms = total_ms / runs
    throughput = runs / (total_ms / 1000.0)
    
    # Coleta amostras para std
    samples = []
    for _ in range(sample_runs):
        s0 = ticks_us()
        snn_inference_pure(norm_input)
        s1 = ticks_us()
        samples.append(ticks_diff(s1, s0) / 1000.0)
    
    sample_mean = sum(samples) / len(samples)
    variance = sum((s - sample_mean) ** 2 for s in samples) / len(samples)
    latency_std = math.sqrt(variance)
    latency_min = min(samples)
    latency_max = max(samples)
    
    # Teste de predição
    pred, spikes = predict(test_input)
    firing_rate = spikes / (TIME_STEPS * HIDDEN_SIZE)
    
    results = {
        "latency_mean_ms": latency_mean_ms,
        "latency_std_ms": latency_std,
        "latency_min_ms": latency_min,
        "latency_max_ms": latency_max,
        "throughput_inf_s": throughput,
        "total_spikes": spikes,
        "firing_rate": firing_rate,
        "predictions": {
            "pH": pred[0],
            "OD": pred[1],
            "Turbidez": pred[2]
        }
    }
    
    print(f"\n--- Resultados ---")
    print(f"Latência média: {latency_mean_ms:.3f} ms")
    print(f"Latência std:   {latency_std:.3f} ms")
    print(f"Latência min:   {latency_min:.3f} ms")
    print(f"Latência max:   {latency_max:.3f} ms")
    print(f"Throughput:     {throughput:.1f} inf/s")
    print(f"Total spikes:   {spikes}")
    print(f"Firing rate:    {firing_rate:.2%}")
    print(f"\n--- Predições ---")
    print(f"pH:       {pred[0]:.4f}")
    print(f"OD:       {pred[1]:.4f}")
    print(f"Turbidez: {pred[2]:.4f}")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import os
    
    # Tenta carregar do JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "..", "esp32_micropython", "model_weights.json")
    
    if os.path.exists(json_path):
        print(f"Carregando modelo de: {json_path}")
        load_model_from_json(json_path)
    else:
        print("model_weights.json não encontrado, tentando model_data.py...")
        try:
            import sys
            sys.path.insert(0, os.path.join(script_dir, "esp32_micropython"))
            load_model_from_module()
        except ImportError:
            print("ERRO: Nenhum modelo encontrado!")
            print("Execute primeiro: python export_pt_to_json.py")
            exit(1)
    
    run_benchmark()
