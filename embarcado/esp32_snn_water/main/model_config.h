#pragma once
/**
 * @file model_config.h
 * @brief Configuração do modelo SNN multimodal e estatísticas de normalização
 * 
 * Gerado automaticamente por export_multimodal_to_c.py
 * Contém:
 *   - Dimensões da rede neural
 *   - Parâmetros do SNN (LIF)
 *   - Estatísticas de z-score normalization do treinamento
 *   - Constantes de inferência
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// ============================================================================
// ARQUITETURA DO MODELO
// ============================================================================
#define MODEL_INPUT_SIZE    5
#define MODEL_HIDDEN_SIZE   64
#define MODEL_OUTPUT_SIZE   3  // pH, OD, Turbidez

// ============================================================================
// PARÂMETROS DO SNN (Leaky Integrate-and-Fire)
// ============================================================================
#define SNN_TIME_STEPS      15      // Numero de passos temporais
#define SNN_BETA            0.95f   // Constante de decaimento
#define SNN_THRESHOLD       0.8f    // Limiar de disparo
#define SNN_RESET_VALUE     0.0f    // Valor de reset após disparo

// ============================================================================
// ESTATÍSTICAS DE NORMALIZAÇÃO (Z-SCORE)
// ============================================================================
// Inputs: x, y, TEMP, COR_A (cor aparente), COR_V (cor verdadeira)
// Outputs: pH, OD (oxigênio dissolvido), Turbidez

// INPUT: Estatísticas do treinamento (de grid_norm_multi.json)
#define NORM_MEAN_X         335575.281250f
#define NORM_MEAN_Y         9657301.000000f
#define NORM_MEAN_TEMP      31.790476f
#define NORM_MEAN_COR_A     89.880951f
#define NORM_MEAN_COR_V     29.214285f

#define NORM_STD_X          1009.637268f
#define NORM_STD_Y          1739.921753f
#define NORM_STD_TEMP       0.390003f
#define NORM_STD_COR_A      25.901505f
#define NORM_STD_COR_V      13.846785f

// OUTPUT: Estatísticas do conjunto de treino
#define OUT_MEAN_PH         5.622857f
#define OUT_MEAN_OD         1.725238f
#define OUT_MEAN_TURBIDITY  2.527381f

#define OUT_STD_PH          0.316531f
#define OUT_STD_OD          0.227631f
#define OUT_STD_TURBIDITY   1.624802f

#endif
