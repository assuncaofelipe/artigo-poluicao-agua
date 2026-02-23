#!/usr/bin/env python3
"""
Export multimodal SNN model to C++
Carrega grid_best_multi.pt e exporta pesos para weights.h e configuração para model_config.h
Também exporta estatísticas de normalização do grid_norm_multi.json
"""
import sys
import os
import json

# Add scripts_multimodal to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts_multimodal"))

try:
    import torch
    import numpy as np
    from config import Config
    from model import SNNRegressor
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Tenha certeza que está rodando com o ambiente Python que tem torch instalado")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
cfg = Config()

# Carrega modelo multimodal (grid_best_multi.pt tem melhor desempenho)
checkpoint_path = os.path.join(script_dir, "..", "..", "scripts_multimodal", "outputs", "grid_best_multi.pt")
norm_path = os.path.join(script_dir, "..", "..", "scripts_multimodal", "outputs", "grid_norm_multi.json")
print(f"Carregando checkpoint multimodal de: {checkpoint_path}")
print(f"Carregando estatísticas de normalização de: {norm_path}")

if not os.path.exists(checkpoint_path):
    print(f"Erro: arquivo não encontrado: {checkpoint_path}")
    sys.exit(1)

if not os.path.exists(norm_path):
    print(f"Erro: arquivo não encontrado: {norm_path}")
    sys.exit(1)

# Carrega estatísticas de normalização
with open(norm_path, 'r') as f:
    norm_stats = json.load(f)

print(f"✓ Features: {norm_stats['feature_cols']}")
print(f"✓ Targets: {norm_stats['target_cols']}")
print(f"✓ X mean: {norm_stats['mean']}")
print(f"✓ X std: {norm_stats['std']}")
print(f"✓ Y mean: {norm_stats['y_mean']}")
print(f"✓ Y std: {norm_stats['y_std']}")

# Carrega o modelo
model = SNNRegressor(
    input_size=len(cfg.feature_cols),
    hidden_size=cfg.hidden_size,
    output_size=len(cfg.target_cols),  # 3: pH, OD, Turbidez
    beta=cfg.beta,
    thresh=cfg.thresh,
)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Extrai pesos
W1 = model.fc1.weight.detach().cpu().numpy()
B1 = model.fc1.bias.detach().cpu().numpy()
W2 = model.fc2.weight.detach().cpu().numpy()
B2 = model.fc2.bias.detach().cpu().numpy()

print(f"✓ W1 shape: {W1.shape}")
print(f"✓ B1 shape: {B1.shape}")
print(f"✓ W2 shape: {W2.shape}  <- 3 outputs (pH, OD, Turbidez)")
print(f"✓ B2 shape: {B2.shape}")

print(f"\nTarget columns: {cfg.target_cols}")
print(f"Number of outputs: {len(cfg.target_cols)}")

def dump_array(name, arr, dtype="float"):
    """Converte um array numpy para sintaxe C const array"""
    flat = arr.flatten()
    txt = f"const {dtype} {name}[{flat.size}] = {{\n"
    for i, v in enumerate(flat):
        txt += f"  {v:.8f}f,"
        if (i + 1) % 8 == 0:
            txt += "\n"
    if flat.size % 8 != 0:
        txt = txt.rstrip(",") + " "
    txt += "\n};\n"
    return txt

def dump_array_2d(name, arr, rows, cols, dtype="float"):
    """Converte um array 2D numpy para sintaxe C const array multi-linha"""
    flat = arr.flatten()
    txt = f"const {dtype} {name}[{rows}][{cols}] = {{\n"
    for r in range(rows):
        txt += "  { "
        for c in range(cols):
            idx = r * cols + c
            txt += f"{flat[idx]:.8f}f"
            if c < cols - 1:
                txt += ", "
        txt += " },\n"
    txt += "};\n"
    return txt

# Gerar weights.h
output_path = os.path.join(script_dir, "..", "esp32_snn_water", "main", "weights.h")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    f.write("#pragma once\n")
    f.write("/**\n")
    f.write(" * @file weights.h\n")
    f.write(" * @brief Pesos e biases da rede neural multimodal extraidos de best_multi.pt\n")
    f.write(" * \n")
    f.write(" * Gerado automaticamente por export_multimodal_to_c.py\n")
    f.write(" * Nao editar manualmente!\n")
    f.write(" * \n")
    f.write(" * Arquitetura do modelo:\n")
    f.write(f" *   - Camada 1: {W1.shape[1]} inputs -> {W1.shape[0]} neurônios hidden\n")
    f.write(f" *   - Camada 2: {W1.shape[0]} neurônios hidden -> {W2.shape[0]} outputs (pH, OD, Turbidez)\n")
    f.write(" */\n\n")
    f.write("#ifndef WEIGHTS_H\n")
    f.write("#define WEIGHTS_H\n\n")
    
    f.write(f"#define INPUT_SIZE {W1.shape[1]}\n")
    f.write(f"#define HIDDEN_SIZE {W1.shape[0]}\n")
    f.write(f"#define OUTPUT_SIZE {W2.shape[0]}\n\n")
    
    f.write("// Layer 1: Input -> Hidden\n")
    f.write(dump_array_2d("W1", W1, W1.shape[0], W1.shape[1]))
    f.write(dump_array("B1", B1))
    
    f.write("\n// Layer 2: Hidden -> Output (3 outputs: pH, OD, Turbidez)\n")
    f.write(dump_array_2d("W2", W2, W2.shape[0], W2.shape[1]))
    f.write(dump_array("B2", B2))
    
    f.write("\n#endif\n")

print(f"✔ Pesos exportados para {output_path}")

# Gerar model_config.h com metadados
config_path = os.path.join(script_dir, "..", "esp32_snn_water", "main", "model_config.h")
with open(config_path, "w") as f:
    f.write("#pragma once\n")
    f.write("/**\n")
    f.write(" * @file model_config.h\n")
    f.write(" * @brief Configuração do modelo SNN multimodal e estatísticas de normalização\n")
    f.write(" * \n")
    f.write(" * Gerado automaticamente por export_multimodal_to_c.py\n")
    f.write(" * Contém:\n")
    f.write(" *   - Dimensões da rede neural\n")
    f.write(" *   - Parâmetros do SNN (LIF)\n")
    f.write(" *   - Estatísticas de z-score normalization do treinamento\n")
    f.write(" *   - Constantes de inferência\n")
    f.write(" */\n\n")
    f.write("#ifndef MODEL_CONFIG_H\n")
    f.write("#define MODEL_CONFIG_H\n\n")
    
    f.write("// ============================================================================\n")
    f.write("// ARQUITETURA DO MODELO\n")
    f.write("// ============================================================================\n")
    f.write(f"#define MODEL_INPUT_SIZE    {W1.shape[1]}\n")
    f.write(f"#define MODEL_HIDDEN_SIZE   {W1.shape[0]}\n")
    f.write(f"#define MODEL_OUTPUT_SIZE   {W2.shape[0]}  // pH, OD, Turbidez\n\n")
    
    f.write("// ============================================================================\n")
    f.write("// PARÂMETROS DO SNN (Leaky Integrate-and-Fire)\n")
    f.write("// ============================================================================\n")
    f.write(f"#define SNN_TIME_STEPS      {cfg.time_steps}      // Número de passos temporais\n")
    f.write(f"#define SNN_BETA            {cfg.beta}f   // Constante de decaimento\n")
    f.write(f"#define SNN_THRESHOLD       {cfg.thresh}f    // Limiar de disparo\n")
    f.write("#define SNN_RESET_VALUE     0.0f    // Valor de reset após disparo\n\n")
    
    f.write("// ============================================================================\n")
    f.write("// ESTATÍSTICAS DE NORMALIZAÇÃO (Z-SCORE)\n")
    f.write("// ============================================================================\n")
    f.write("// Inputs: x, y, TEMP, COR_A (cor aparente), COR_V (cor verdadeira)\n")
    f.write("// Outputs: pH, OD (oxigênio dissolvido), Turbidez\n\n")
    
    # Usa estatísticas do arquivo grid_norm_multi.json
    f.write("// INPUT: Estatísticas do treinamento (de grid_norm_multi.json)\n")
    f.write(f"#define NORM_MEAN_X         {norm_stats['mean'][0]:.6f}f\n")
    f.write(f"#define NORM_MEAN_Y         {norm_stats['mean'][1]:.6f}f\n")
    f.write(f"#define NORM_MEAN_TEMP      {norm_stats['mean'][2]:.6f}f\n")
    f.write(f"#define NORM_MEAN_COR_A     {norm_stats['mean'][3]:.6f}f\n")
    f.write(f"#define NORM_MEAN_COR_V     {norm_stats['mean'][4]:.6f}f\n\n")
    
    f.write(f"#define NORM_STD_X          {norm_stats['std'][0]:.6f}f\n")
    f.write(f"#define NORM_STD_Y          {norm_stats['std'][1]:.6f}f\n")
    f.write(f"#define NORM_STD_TEMP       {norm_stats['std'][2]:.6f}f\n")
    f.write(f"#define NORM_STD_COR_A      {norm_stats['std'][3]:.6f}f\n")
    f.write(f"#define NORM_STD_COR_V      {norm_stats['std'][4]:.6f}f\n\n")
    
    f.write("// OUTPUT: Estatísticas do conjunto de treino\n")
    f.write(f"#define OUT_MEAN_PH         {norm_stats['y_mean'][0]:.6f}f\n")
    f.write(f"#define OUT_MEAN_OD         {norm_stats['y_mean'][1]:.6f}f\n")
    f.write(f"#define OUT_MEAN_TURBIDITY  {norm_stats['y_mean'][2]:.6f}f\n\n")
    
    f.write(f"#define OUT_STD_PH          {norm_stats['y_std'][0]:.6f}f\n")
    f.write(f"#define OUT_STD_OD          {norm_stats['y_std'][1]:.6f}f\n")
    f.write(f"#define OUT_STD_TURBIDITY   {norm_stats['y_std'][2]:.6f}f\n\n")
    
    f.write("#endif\n")

print(f"✔ Config exportado para {config_path}")
print(f"\n✨ Modelo multimodal exportado com sucesso!")
print(f"   - Arquitetura: {W1.shape[1]} inputs -> {W1.shape[0]} hidden -> {W2.shape[0]} outputs")
print(f"   - Outputs: pH, OD (O2 dissolvido), Turbidez")
