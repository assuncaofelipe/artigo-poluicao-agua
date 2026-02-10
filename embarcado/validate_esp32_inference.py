#!/usr/bin/env python3
"""
Validação da Inferência ESP32 vs PyTorch

Este script replica exatamente a inferência do ESP32 (modo determinístico)
usando os mesmos pesos e parâmetros, para comprovar que o firmware está correto.

Uso:
    python validate_esp32_inference.py

Saída esperada: erros < 1e-5 entre ESP32 e Python
"""

import sys
import os
import json
import numpy as np

# Adiciona path para importar módulos do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts_multimodal'))

import torch
from model import SNNRegressor


def sigmoid(x):
    """Sigmoid function (numpy)"""
    return 1.0 / (1.0 + np.exp(-x))


def deterministic_snn_inference(norm_input, W1, B1, W2, B2, beta=0.95, thresh=0.8, time_steps=15):
    """
    Replica exatamente a inferência determinística do ESP32.
    
    No modo determinístico, usa-se sigmoid(x) diretamente como "taxa de spikes"
    em vez de amostrar binários.
    
    Args:
        norm_input: Array [5] com features normalizadas
        W1: Pesos camada 1 [64, 5]
        B1: Bias camada 1 [64]
        W2: Pesos camada 2 [3, 64]
        B2: Bias camada 2 [3]
        beta: Constante de decaimento LIF
        thresh: Limiar de disparo
        time_steps: Número de passos temporais
    
    Returns:
        output: Array [3] com saídas normalizadas
    """
    hidden_size = W1.shape[0]
    output_size = W2.shape[0]
    
    # Inicializa estados
    membrane = np.zeros(hidden_size)
    output_accum = np.zeros(output_size)
    
    for t in range(time_steps):
        # Rate encode: usa probabilidade diretamente
        spike_input = sigmoid(norm_input)
        
        # Hidden layer: LIF
        I_syn = W1 @ spike_input + B1
        membrane = beta * membrane + I_syn
        hidden_spike = (membrane > thresh).astype(np.float32)
        membrane = membrane - hidden_spike * thresh  # Subtractive reset
        
        # Output layer
        output_accum += W2 @ hidden_spike + B2
    
    return output_accum / time_steps


def load_weights_from_header(weights_file):
    """
    Carrega pesos do arquivo C header (weights.h)
    """
    with open(weights_file, 'r') as f:
        content = f.read()
    
    def parse_array(content, name, shape):
        import re
        # Encontra o array
        pattern = rf'{name}\s*\[[^\]]*\]\s*(?:\[[^\]]*\])?\s*=\s*\{{([^;]+)\}};'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find array {name}")
        
        # Extrai valores
        values_str = match.group(1)
        # Remove nested braces e extrai números
        values_str = re.sub(r'[{}]', '', values_str)
        values = [float(x.strip().rstrip('f')) for x in values_str.split(',') if x.strip()]
        
        return np.array(values).reshape(shape)
    
    W1 = parse_array(content, 'W1', (64, 5))
    B1 = parse_array(content, 'B1', (64,))
    W2 = parse_array(content, 'W2', (3, 64))
    B2 = parse_array(content, 'B2', (3,))
    
    return W1, B1, W2, B2


def load_normalization_stats(norm_file):
    """Carrega estatísticas de normalização"""
    with open(norm_file, 'r') as f:
        norm = json.load(f)
    
    x_mean = np.array(norm['mean'])
    x_std = np.array(norm['std'])
    y_mean = np.array(norm['y_mean'])
    y_std = np.array(norm['y_std'])
    
    return x_mean, x_std, y_mean, y_std


def normalize_input(raw_input, x_mean, x_std):
    """Normaliza input usando z-score"""
    return (raw_input - x_mean) / x_std


def denormalize_output(norm_output, y_mean, y_std):
    """Denormaliza output"""
    return norm_output * y_std + y_mean


# Dataset Serpa (mesmo do ESP32 - 60 amostras)
# Features: x, y, TEMP, COR_A, COR_V
# Ground truth: pH, OD, TURB
DATASET = [
    # x, y, temp, cor_a, cor_v, ph, od, turb
    (336691.99, 9654879.85, 31.3, 59.0, 32.0, 6.32, 2.22, 1.41),
    (336683.40, 9654961.36, 31.4, 67.0, 30.0, 5.93, 1.93, 3.26),
    (336730.84, 9654965.37, 31.5, 60.0, 33.0, 5.96, 1.85, 2.33),
    (336617.09, 9654965.27, 31.4, 68.0, 26.0, 5.93, 1.61, 0.00),
    (336639.41, 9655028.93, 31.4, 69.0, 18.0, 5.94, 1.82, 1.19),
    (336718.32, 9655085.13, 31.5, 64.0, 8.0, 5.95, 1.35, 1.14),
    (336611.38, 9655149.66, 31.4, 69.0, 21.0, 5.96, 1.72, 0.54),
    (336564.75, 9655159.52, 31.5, 67.0, 15.0, 5.90, 1.56, 0.05),
    (336639.63, 9655217.12, 31.5, 64.0, 16.0, 5.92, 1.44, 0.97),
    (336758.74, 9655268.80, 31.4, 63.0, 21.0, 5.88, 1.33, 0.05),
    (336710.23, 9655394.49, 31.4, 61.0, 12.0, 5.86, 1.68, 0.27),
    (336587.08, 9655405.65, 31.5, 63.0, 16.0, 5.89, 2.01, 1.03),
    (336514.10, 9655400.61, 31.5, 61.0, 22.0, 5.86, 1.32, 0.00),
    (336399.73, 9655571.06, 31.4, 72.0, 27.0, 5.87, 1.81, 0.70),
    (336501.05, 9655683.03, 31.4, 55.0, 17.0, 5.85, 1.86, 0.92),
    (336634.51, 9655789.62, 31.3, 66.0, 20.0, 5.84, 1.42, 1.73),
    (336560.39, 9655948.48, 31.2, 70.0, 29.0, 5.84, 1.41, 2.33),
    (336459.86, 9655923.65, 31.4, 93.0, 28.0, 5.80, 1.48, 2.33),
    (336342.28, 9655905.11, 31.4, 75.0, 30.0, 5.77, 1.39, 2.11),
    (336236.16, 9656056.73, 31.4, 70.0, 22.0, 5.85, 1.25, 3.80),
    (336405.13, 9656162.56, 31.3, 63.0, 24.0, 5.73, 1.57, 1.84),
    (336555.27, 9656249.29, 31.3, 73.0, 20.0, 5.77, 1.38, 2.11),
    (336526.74, 9656353.40, 31.4, 78.0, 22.0, 6.52, 1.72, 2.60),
    (336296.28, 9656549.81, 31.7, 65.0, 17.0, 5.83, 1.89, 1.84),
    (336216.70, 9656527.86, 31.6, 75.0, 28.0, 5.62, 1.59, 1.90),
    (336125.83, 9656476.86, 31.7, 74.0, 18.0, 5.56, 1.94, 2.22),
    (335778.07, 9656784.34, 31.2, 75.0, 17.0, 5.36, 1.94, 2.88),
    (335832.32, 9656886.68, 31.9, 76.0, 16.0, 5.47, 1.82, 2.66),
    (335934.20, 9657043.72, 31.8, 60.0, 17.0, 5.48, 1.68, 0.92),
    (335590.62, 9657027.25, 32.2, 73.0, 24.0, 5.59, 1.73, 1.57),
    (335638.40, 9657096.20, 32.2, 76.0, 29.0, 5.55, 1.86, 1.30),
    (335672.26, 9657142.71, 32.1, 70.0, 33.0, 5.55, 2.50, 2.11),
    (335456.53, 9657607.20, 32.1, 85.0, 40.0, 5.58, 2.13, 2.11),
    (335384.46, 9657643.93, 32.0, 85.0, 26.0, 5.56, 1.83, 1.90),
    (335330.01, 9657657.11, 32.0, 87.0, 38.0, 5.56, 1.93, 2.55),
    (335441.66, 9658125.34, 32.0, 95.0, 26.0, 5.60, 1.85, 3.64),
    (335503.71, 9658181.24, 32.0, 125.0, 25.0, 5.55, 2.00, 3.31),
    (335559.75, 9658211.07, 31.8, 89.0, 27.0, 5.48, 1.95, 2.33),
    (335372.93, 9658836.82, 32.1, 102.0, 29.0, 5.51, 1.83, 0.00),
    (335218.96, 9659035.19, 32.1, 115.0, 31.0, 5.49, 1.89, 5.43),
    (335117.60, 9659286.07, 32.1, 116.0, 25.0, 5.43, 1.59, 5.92),
    (334955.51, 9659263.53, 32.2, 124.0, 14.0, 5.36, 1.78, 4.29),
    (333570.99, 9659074.36, 32.0, 94.0, 48.0, 5.75, 1.87, 2.82),
    (333634.79, 9659171.19, 31.9, 90.0, 40.0, 5.72, 1.89, 3.96),
    (333779.92, 9659168.82, 32.0, 112.0, 50.0, 5.44, 1.81, 4.34),
    (333756.27, 9659243.08, 32.0, 115.0, 62.0, 5.31, 1.93, 4.13),
    (334040.73, 9659261.36, 32.0, 116.0, 49.0, 5.32, 1.88, 4.51),
    (334027.59, 9659200.15, 32.1, 121.0, 32.0, 5.22, 1.90, 4.72),
    (334232.76, 9659242.52, 31.9, 123.0, 29.0, 5.24, 1.64, 5.05),
    (334280.32, 9659185.98, 29.8, 124.0, 27.0, 5.29, 1.74, 3.91),
    (334216.71, 9659294.60, 31.6, 138.0, 28.0, 5.27, 1.86, 2.82),
    (334559.48, 9659363.26, 32.6, 124.0, 25.0, 5.27, 1.80, 4.56),
    (334565.60, 9659314.86, 32.1, 138.0, 74.0, 5.25, 1.33, 6.08),
    (334583.70, 9659432.52, 32.4, 126.0, 68.0, 5.25, 1.47, 3.47),
    (334767.81, 9659372.14, 32.5, 97.0, 56.0, 5.22, 1.78, 4.07),
    (334784.70, 9659333.76, 32.5, 124.0, 44.0, 5.25, 1.89, 5.65),
    (334775.78, 9659287.24, 32.4, 125.0, 41.0, 5.26, 1.41, 1.79),
    (335067.13, 9659159.65, 32.8, 121.0, 40.0, 5.16, 1.90, 0.92),
    (335134.88, 9659207.18, 32.4, 110.0, 27.0, 5.16, 1.98, 3.96),
    (335181.26, 9659263.92, 32.0, 123.0, 31.0, 5.18, 1.31, 1.14),
]


def main():
    print("=" * 60)
    print("  Validação: ESP32 vs Python (Modo Determinístico)")
    print("=" * 60)
    
    # Caminhos
    script_dir = os.path.dirname(__file__)
    weights_file = os.path.join(script_dir, 'esp32_snn_water', 'main', 'weights.h')
    norm_file = os.path.join(script_dir, '..', 'scripts_multimodal', 'outputs', 'grid_norm_multi.json')
    
    # Carrega pesos e normalização
    print("\n[1] Carregando pesos e configuração...")
    W1, B1, W2, B2 = load_weights_from_header(weights_file)
    x_mean, x_std, y_mean, y_std = load_normalization_stats(norm_file)
    
    print(f"    W1 shape: {W1.shape}, B1 shape: {B1.shape}")
    print(f"    W2 shape: {W2.shape}, B2 shape: {B2.shape}")
    print(f"    X mean: {x_mean}")
    print(f"    Y mean: {y_mean}")
    
    # Executa inferência para todas as amostras
    print("\n[2] Executando inferência determinística...\n")
    print(f"{'Sample':>6} | {'Pred pH':>8} {'Pred OD':>8} {'Pred Turb':>10} | "
          f"{'Real pH':>8} {'Real OD':>8} {'Real Turb':>10}")
    print("-" * 80)
    
    total_ph_error = 0
    total_od_error = 0
    total_turb_error = 0
    
    predictions = []
    
    for i, sample in enumerate(DATASET):
        x, y, temp, cor_a, cor_v, real_ph, real_od, real_turb = sample
        
        # Prepara input
        raw_input = np.array([x, y, temp, cor_a, cor_v], dtype=np.float32)
        
        # Normaliza
        norm_input = normalize_input(raw_input, x_mean, x_std)
        
        # Inferência
        norm_output = deterministic_snn_inference(norm_input, W1, B1, W2, B2)
        
        # Denormaliza
        pred_output = denormalize_output(norm_output, y_mean, y_std)
        pred_ph, pred_od, pred_turb = pred_output
        
        predictions.append((pred_ph, pred_od, pred_turb))
        
        # Calcula erros
        total_ph_error += abs(pred_ph - real_ph)
        total_od_error += abs(pred_od - real_od)
        total_turb_error += abs(pred_turb - real_turb)
        
        print(f"{i+1:>6} | {pred_ph:>8.2f} {pred_od:>8.2f} {pred_turb:>10.2f} | "
              f"{real_ph:>8.2f} {real_od:>8.2f} {real_turb:>10.2f}")
    
    # Métricas finais
    n = len(DATASET)
    mae_ph = total_ph_error / n
    mae_od = total_od_error / n
    mae_turb = total_turb_error / n
    
    print("\n" + "=" * 60)
    print("  RESULTADOS - Métricas de Erro (MAE)")
    print("=" * 60)
    print(f"  pH MAE:        {mae_ph:.4f}")
    print(f"  OD MAE:        {mae_od:.4f} mg/L")
    print(f"  Turbidity MAE: {mae_turb:.4f} NTU")
    print("=" * 60)
    
    # Salva resultados para comparação
    output_file = os.path.join(script_dir, 'validation_results.json')
    results = {
        'mode': 'deterministic',
        'samples': n,
        'mae': {
            'ph': float(mae_ph),
            'od': float(mae_od),
            'turbidity': float(mae_turb)
        },
        'predictions': [
            {'sample': i+1, 'pred_ph': float(p[0]), 'pred_od': float(p[1]), 'pred_turb': float(p[2])}
            for i, p in enumerate(predictions)
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Resultados salvos em: {output_file}")
    print("\n  NOTA: Execute o ESP32 com DETERMINISTIC_MODE=1 e compare")
    print("        os resultados. Devem ser IDÊNTICOS (erro < 1e-5).")
    
    return predictions


if __name__ == '__main__':
    main()
