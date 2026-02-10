/**
 * @file main.cpp
 * @brief SNN Multimodal para inferência de qualidade de água em ESP32
 * 
 * Implementa rede neural spiking (LIF) para predição de pH, OD e Turbidez
 * a partir de coordenadas geográficas e parâmetros de cor da água.
 * 
 * Features de entrada: x, y, TEMP, COR_A (cor aparente), COR_V (cor verdadeira)
 * Saídas preditas: pH, OD (oxigênio dissolvido), TURB (turbidez)
 * 
 * Dados: 60 amostras da Lagoa de Serpa, Amazônia
 */

#include <stdio.h>
#include <cmath>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_system.h"
#include "hal/cpu_hal.h"
#include "model_config.h"
#include "weights.h"

// Função para obter tempo em microsegundos usando ciclos de CPU
static inline uint32_t get_time_us() {
    return (uint32_t)(cpu_hal_get_cycle_count() / (CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ));
}

// Logging macros (ESP-IDF style)
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[%s WARN] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[%s ERROR] " fmt "\n", tag, ##__VA_ARGS__)

static const char* TAG = "SNN_MULTIMODAL";

// ============================================================================
// DATASET SERPA - 60 amostras
// Features: x, y, TEMP, COR_A, COR_V
// Ground truth: pH, OD, TURB (para comparação)
// ============================================================================
struct DataPoint {
    float x;           // Coordenada UTM X
    float y;           // Coordenada UTM Y
    float temp;        // Temperatura (°C)
    float cor_a;       // Cor aparente
    float cor_v;       // Cor verdadeira
    // Ground truth (valores reais do CSV para comparação)
    float ph_real;
    float od_real;
    float turb_real;
};

static const DataPoint dataset[] = {
    {336691.99f, 9654879.85f, 31.3f, 59.0f, 32.0f, 6.32f, 2.22f, 1.41f},
    {336683.40f, 9654961.36f, 31.4f, 67.0f, 30.0f, 5.93f, 1.93f, 3.26f},
    {336730.84f, 9654965.37f, 31.5f, 60.0f, 33.0f, 5.96f, 1.85f, 2.33f},
    {336617.09f, 9654965.27f, 31.4f, 68.0f, 26.0f, 5.93f, 1.61f, 0.00f},
    {336639.41f, 9655028.93f, 31.4f, 69.0f, 18.0f, 5.94f, 1.82f, 1.19f},
    {336718.32f, 9655085.13f, 31.5f, 64.0f, 8.0f, 5.95f, 1.35f, 1.14f},
    {336611.38f, 9655149.66f, 31.4f, 69.0f, 21.0f, 5.96f, 1.72f, 0.54f},
    {336564.75f, 9655159.52f, 31.5f, 67.0f, 15.0f, 5.90f, 1.56f, 0.05f},
    {336639.63f, 9655217.12f, 31.5f, 64.0f, 16.0f, 5.92f, 1.44f, 0.97f},
    {336758.74f, 9655268.80f, 31.4f, 63.0f, 21.0f, 5.88f, 1.33f, 0.05f},
    {336710.23f, 9655394.49f, 31.4f, 61.0f, 12.0f, 5.86f, 1.68f, 0.27f},
    {336587.08f, 9655405.65f, 31.5f, 63.0f, 16.0f, 5.89f, 2.01f, 1.03f},
    {336514.10f, 9655400.61f, 31.5f, 61.0f, 22.0f, 5.86f, 1.32f, 0.00f},
    {336399.73f, 9655571.06f, 31.4f, 72.0f, 27.0f, 5.87f, 1.81f, 0.70f},
    {336501.05f, 9655683.03f, 31.4f, 55.0f, 17.0f, 5.85f, 1.86f, 0.92f},
    {336634.51f, 9655789.62f, 31.3f, 66.0f, 20.0f, 5.84f, 1.42f, 1.73f},
    {336560.39f, 9655948.48f, 31.2f, 70.0f, 29.0f, 5.84f, 1.41f, 2.33f},
    {336459.86f, 9655923.65f, 31.4f, 93.0f, 28.0f, 5.80f, 1.48f, 2.33f},
    {336342.28f, 9655905.11f, 31.4f, 75.0f, 30.0f, 5.77f, 1.39f, 2.11f},
    {336236.16f, 9656056.73f, 31.4f, 70.0f, 22.0f, 5.85f, 1.25f, 3.80f},
    {336405.13f, 9656162.56f, 31.3f, 63.0f, 24.0f, 5.73f, 1.57f, 1.84f},
    {336555.27f, 9656249.29f, 31.3f, 73.0f, 20.0f, 5.77f, 1.38f, 2.11f},
    {336526.74f, 9656353.40f, 31.4f, 78.0f, 22.0f, 6.52f, 1.72f, 2.60f},
    {336296.28f, 9656549.81f, 31.7f, 65.0f, 17.0f, 5.83f, 1.89f, 1.84f},
    {336216.70f, 9656527.86f, 31.6f, 75.0f, 28.0f, 5.62f, 1.59f, 1.90f},
    {336125.83f, 9656476.86f, 31.7f, 74.0f, 18.0f, 5.56f, 1.94f, 2.22f},
    {335778.07f, 9656784.34f, 31.2f, 75.0f, 17.0f, 5.36f, 1.94f, 2.88f},
    {335832.32f, 9656886.68f, 31.9f, 76.0f, 16.0f, 5.47f, 1.82f, 2.66f},
    {335934.20f, 9657043.72f, 31.8f, 60.0f, 17.0f, 5.48f, 1.68f, 0.92f},
    {335590.62f, 9657027.25f, 32.2f, 73.0f, 24.0f, 5.59f, 1.73f, 1.57f},
    {335638.40f, 9657096.20f, 32.2f, 76.0f, 29.0f, 5.55f, 1.86f, 1.30f},
    {335672.26f, 9657142.71f, 32.1f, 70.0f, 33.0f, 5.55f, 2.50f, 2.11f},
    {335456.53f, 9657607.20f, 32.1f, 85.0f, 40.0f, 5.58f, 2.13f, 2.11f},
    {335384.46f, 9657643.93f, 32.0f, 85.0f, 26.0f, 5.56f, 1.83f, 1.90f},
    {335330.01f, 9657657.11f, 32.0f, 87.0f, 38.0f, 5.56f, 1.93f, 2.55f},
    {335441.66f, 9658125.34f, 32.0f, 95.0f, 26.0f, 5.60f, 1.85f, 3.64f},
    {335503.71f, 9658181.24f, 32.0f, 125.0f, 25.0f, 5.55f, 2.00f, 3.31f},
    {335559.75f, 9658211.07f, 31.8f, 89.0f, 27.0f, 5.48f, 1.95f, 2.33f},
    {335372.93f, 9658836.82f, 32.1f, 102.0f, 29.0f, 5.51f, 1.83f, 0.00f},
    {335218.96f, 9659035.19f, 32.1f, 115.0f, 31.0f, 5.49f, 1.89f, 5.43f},
    {335117.60f, 9659286.07f, 32.1f, 116.0f, 25.0f, 5.43f, 1.59f, 5.92f},
    {334955.51f, 9659263.53f, 32.2f, 124.0f, 14.0f, 5.36f, 1.78f, 4.29f},
    {333570.99f, 9659074.36f, 32.0f, 94.0f, 48.0f, 5.75f, 1.87f, 2.82f},
    {333634.79f, 9659171.19f, 31.9f, 90.0f, 40.0f, 5.72f, 1.89f, 3.96f},
    {333779.92f, 9659168.82f, 32.0f, 112.0f, 50.0f, 5.44f, 1.81f, 4.34f},
    {333756.27f, 9659243.08f, 32.0f, 115.0f, 62.0f, 5.31f, 1.93f, 4.13f},
    {334040.73f, 9659261.36f, 32.0f, 116.0f, 49.0f, 5.32f, 1.88f, 4.51f},
    {334027.59f, 9659200.15f, 32.1f, 121.0f, 32.0f, 5.22f, 1.90f, 4.72f},
    {334232.76f, 9659242.52f, 31.9f, 123.0f, 29.0f, 5.24f, 1.64f, 5.05f},
    {334280.32f, 9659185.98f, 29.8f, 124.0f, 27.0f, 5.29f, 1.74f, 3.91f},
    {334216.71f, 9659294.60f, 31.6f, 138.0f, 28.0f, 5.27f, 1.86f, 2.82f},
    {334559.48f, 9659363.26f, 32.6f, 124.0f, 25.0f, 5.27f, 1.80f, 4.56f},
    {334565.60f, 9659314.86f, 32.1f, 138.0f, 74.0f, 5.25f, 1.33f, 6.08f},
    {334583.70f, 9659432.52f, 32.4f, 126.0f, 68.0f, 5.25f, 1.47f, 3.47f},
    {334767.81f, 9659372.14f, 32.5f, 97.0f, 56.0f, 5.22f, 1.78f, 4.07f},
    {334784.70f, 9659333.76f, 32.5f, 124.0f, 44.0f, 5.25f, 1.89f, 5.65f},
    {334775.78f, 9659287.24f, 32.4f, 125.0f, 41.0f, 5.26f, 1.41f, 1.79f},
    {335067.13f, 9659159.65f, 32.8f, 121.0f, 40.0f, 5.16f, 1.90f, 0.92f},
    {335134.88f, 9659207.18f, 32.4f, 110.0f, 27.0f, 5.16f, 1.98f, 3.96f},
    {335181.26f, 9659263.92f, 32.0f, 123.0f, 31.0f, 5.18f, 1.31f, 1.14f},
};

static const int DATASET_SIZE = sizeof(dataset) / sizeof(dataset[0]);

// ============================================================================
// LIF SPIKING NEURAL NETWORK
// ============================================================================
struct LIFNeuron {
    float membrane;  // Potencial de membrana
    float spike;     // Spike (0 ou 1)
};

static LIFNeuron hidden_neurons[HIDDEN_SIZE];

/**
 * @brief Normaliza input usando Z-score com estatísticas do treinamento
 */
void normalize_input(const float* raw, float* normalized) {
    normalized[0] = (raw[0] - NORM_MEAN_X) / NORM_STD_X;
    normalized[1] = (raw[1] - NORM_MEAN_Y) / NORM_STD_Y;
    normalized[2] = (raw[2] - NORM_MEAN_TEMP) / NORM_STD_TEMP;
    normalized[3] = (raw[3] - NORM_MEAN_COR_A) / NORM_STD_COR_A;
    normalized[4] = (raw[4] - NORM_MEAN_COR_V) / NORM_STD_COR_V;
}

/**
 * @brief Denormaliza output para valores reais
 */
void denormalize_output(const float* normalized, float* real) {
    real[0] = normalized[0] * OUT_STD_PH + OUT_MEAN_PH;           // pH
    real[1] = normalized[1] * OUT_STD_OD + OUT_MEAN_OD;           // OD
    real[2] = normalized[2] * OUT_STD_TURBIDITY + OUT_MEAN_TURBIDITY;  // Turbidez
}

// Variáveis globais para diagnóstico
static int g_total_spikes = 0;
static int g_debug_sample = 0;

// Controle de reprodutibilidade
#define DETERMINISTIC_MODE 1  // 1 = determinístico (média), 0 = estocástico
#define ENSEMBLE_RUNS 10      // Número de execuções para ensemble (se estocástico)

/**
 * @brief Sigmoid function
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Fast pseudo-random number generator (Xorshift32)
 */
static uint32_t xorshift_state = 12345;
inline void reset_random_seed(uint32_t seed) {
    xorshift_state = seed;
}
inline float fast_random() {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    return (float)(xorshift_state & 0xFFFF) / 65535.0f;  // 0.0 to 1.0
}

/**
 * @brief Rate encode: converte valores normalizados em spikes
 * 
 * Modo determinístico: usa probabilidade diretamente (spike = sigmoid(x))
 * Modo estocástico: amostra binária (spike = random() < sigmoid(x))
 */
void rate_encode_input(const float* norm_input, float* spike_input, bool deterministic) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        float prob = sigmoid(norm_input[i]);
        if (deterministic) {
            spike_input[i] = prob;  // Usa probabilidade diretamente
        } else {
            spike_input[i] = (fast_random() < prob) ? 1.0f : 0.0f;
        }
    }
}

/**
 * @brief Executa uma única inferência SNN (core)
 * 
 * @param norm_input Array de 5 features normalizadas
 * @param output Array de 3 saídas normalizadas
 * @param deterministic Se true, usa probabilidades; se false, amostra spikes
 */
void snn_inference_single(const float* norm_input, float* output, bool deterministic) {
    // Reset membrane potentials
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        hidden_neurons[h].membrane = 0.0f;
        hidden_neurons[h].spike = 0.0f;
    }
    
    float output_accum[OUTPUT_SIZE] = {0.0f, 0.0f, 0.0f};
    float spike_input[INPUT_SIZE];
    
    for (int t = 0; t < SNN_TIME_STEPS; t++) {
        // Rate encode
        rate_encode_input(norm_input, spike_input, deterministic);
        
        // Hidden layer: LIF neurons
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            float I_syn = B1[h];
            for (int i = 0; i < INPUT_SIZE; i++) {
                I_syn += W1[h][i] * spike_input[i];
            }
            
            hidden_neurons[h].membrane = SNN_BETA * hidden_neurons[h].membrane + I_syn;
            hidden_neurons[h].spike = (hidden_neurons[h].membrane > SNN_THRESHOLD) ? 1.0f : 0.0f;
            if (hidden_neurons[h].spike > 0.0f) {
                g_total_spikes++;  // Contabiliza spike para estatísticas
            }
            hidden_neurons[h].membrane -= hidden_neurons[h].spike * SNN_THRESHOLD;
        }
        
        // Output layer: weighted sum of spikes
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            float sum = B2[o];
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                sum += W2[o][h] * hidden_neurons[h].spike;
            }
            output_accum[o] += sum;
        }
    }
    
    // Average over timesteps
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output[o] = output_accum[o] / SNN_TIME_STEPS;
    }
}

/**
 * @brief Executa inferência SNN com modo selecionável
 * 
 * DETERMINISTIC_MODE=1: Usa probabilidades sigmoid diretamente (sempre igual)
 * DETERMINISTIC_MODE=0: Ensemble de múltiplas execuções estocásticas
 * 
 * @param norm_input Array de 5 features normalizadas [x, y, TEMP, COR_A, COR_V]
 * @param output Array de 3 saídas normalizadas [pH, OD, TURB]
 */
void snn_inference(const float* norm_input, float* output) {
#if DETERMINISTIC_MODE
    // Modo determinístico: usa sigmoid diretamente como "taxa" de spikes
    snn_inference_single(norm_input, output, true);
#else
    // Modo ensemble: múltiplas inferências estocásticas com média
    float ensemble_output[OUTPUT_SIZE] = {0.0f, 0.0f, 0.0f};
    float single_output[OUTPUT_SIZE];
    
    for (int run = 0; run < ENSEMBLE_RUNS; run++) {
        // Seed diferente para cada run (baseado em sample + run)
        reset_random_seed(12345 + g_debug_sample * 100 + run);
        snn_inference_single(norm_input, single_output, false);
        
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            ensemble_output[o] += single_output[o];
        }
    }
    
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        output[o] = ensemble_output[o] / ENSEMBLE_RUNS;
    }
#endif
}

/**
 * @brief Calcula índice composto de poluição conforme artigo
 * 
 * Baseado em:
 * (i) desvio do pH do neutro (7.0)
 * (ii) OD inverso normalizado (menor OD = maior poluição)
 * (iii) turbidez normalizada
 * 
 * @param ph pH predito
 * @param od Oxigênio dissolvido predito (mg/L)
 * @param turbidity Turbidez predita (NTU)
 * @return Índice de poluição normalizado [0, 1]
 */
float calculate_pollution_index(float ph, float od, float turbidity) {
    // Limita valores para faixas realistas
    if (ph < 4.0f) ph = 4.0f;
    if (ph > 10.0f) ph = 10.0f;
    if (od < 0.0f) od = 0.0f;
    if (od > 15.0f) od = 15.0f;
    if (turbidity < 0.0f) turbidity = 0.0f;
    if (turbidity > 100.0f) turbidity = 100.0f;
    
    // (i) Desvio do pH neutro normalizado [0, 1]
    float ph_deviation = fabsf(ph - 7.0f) / 3.0f;
    if (ph_deviation > 1.0f) ph_deviation = 1.0f;
    
    // (ii) OD inverso normalizado [0, 1]
    // OD alto = boa qualidade, OD baixo = poluição
    // OD típico para água saudável: 6-8 mg/L
    // OD < 2 mg/L é crítico para vida aquática
    float od_inverse = 1.0f - (od / 8.0f);
    if (od_inverse < 0.0f) od_inverse = 0.0f;
    if (od_inverse > 1.0f) od_inverse = 1.0f;
    
    // (iii) Turbidez normalizada [0, 1]
    // Turbidez até 5 NTU é considerada boa
    float turb_norm = turbidity / 10.0f;
    if (turb_norm > 1.0f) turb_norm = 1.0f;
    
    // Índice composto com pesos (conforme config: 0.4, 0.4, 0.2)
    float index = 0.4f * ph_deviation + 0.4f * od_inverse + 0.2f * turb_norm;
    
    return index;
}

/**
 * @brief Classifica nível de poluição
 */
const char* classify_pollution_level(float index) {
    if (index < 0.33f) return "LOW";
    if (index < 0.66f) return "MEDIUM";
    return "HIGH";
}

/**
 * @brief Executa benchmark de latência da SNN
 * 
 * Similar ao bench_tflite.py - mede:
 * - Latência média por inferência
 * - Desvio padrão
 * - Throughput (amostras/segundo)
 * 
 * Saída em formato JSON para fácil parsing
 */
void run_benchmark() {
    const int WARMUP_RUNS = 10;
    const int BENCHMARK_RUNS = 1000;
    const int SAMPLE_RUNS = 200;
    
    // Prepara input de teste (primeira amostra do dataset)
    float raw_input[INPUT_SIZE] = {
        dataset[0].x,
        dataset[0].y,
        dataset[0].temp,
        dataset[0].cor_a,
        dataset[0].cor_v
    };
    float norm_input[INPUT_SIZE];
    normalize_input(raw_input, norm_input);
    float norm_output[OUTPUT_SIZE];
    
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== BENCHMARK MODE ===");
    ESP_LOGI(TAG, "Warmup: %d runs", WARMUP_RUNS);
    ESP_LOGI(TAG, "Benchmark: %d runs", BENCHMARK_RUNS);
    ESP_LOGI(TAG, "Sample latencies: %d runs", SAMPLE_RUNS);
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        snn_inference(norm_input, norm_output);
    }
    
    // Benchmark principal - mede tempo total
    uint32_t t0 = get_time_us();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        snn_inference(norm_input, norm_output);
    }
    uint32_t t1 = get_time_us();
    
    double total_time_ms = (t1 - t0) / 1000.0;
    double latency_mean_ms = total_time_ms / BENCHMARK_RUNS;
    double throughput = BENCHMARK_RUNS / (total_time_ms / 1000.0);
    
    // Coleta amostras individuais para calcular desvio padrão
    float samples[SAMPLE_RUNS];
    for (int i = 0; i < SAMPLE_RUNS; i++) {
        uint32_t s0 = get_time_us();
        snn_inference(norm_input, norm_output);
        uint32_t s1 = get_time_us();
        samples[i] = (s1 - s0) / 1000.0f;  // ms
    }
    
    // Calcula desvio padrão
    float sum = 0.0f;
    for (int i = 0; i < SAMPLE_RUNS; i++) {
        sum += samples[i];
    }
    float mean = sum / SAMPLE_RUNS;
    
    float variance = 0.0f;
    for (int i = 0; i < SAMPLE_RUNS; i++) {
        float diff = samples[i] - mean;
        variance += diff * diff;
    }
    float std_dev = sqrtf(variance / SAMPLE_RUNS);
    
    // Informações do chip
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    const char* chip_model = "ESP32";
    if (chip_info.model == CHIP_ESP32S2) chip_model = "ESP32-S2";
    else if (chip_info.model == CHIP_ESP32S3) chip_model = "ESP32-S3";
    else if (chip_info.model == CHIP_ESP32C3) chip_model = "ESP32-C3";
    
    // Output formatado para captura
    printf("\n");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "BENCHMARK RESULTS");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Device: %s", chip_model);
    ESP_LOGI(TAG, "Cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "CPU freq: %d MHz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    ESP_LOGI(TAG, "Features: %d inputs, %d hidden, %d outputs", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    ESP_LOGI(TAG, "Time steps: %d", SNN_TIME_STEPS);
    ESP_LOGI(TAG, "Latency mean: %.4f ms/sample", latency_mean_ms);
    ESP_LOGI(TAG, "Latency std: %.4f ms", std_dev);
    ESP_LOGI(TAG, "Throughput: %.2f samples/s", throughput);
    ESP_LOGI(TAG, "========================================");
    
    // JSON para parsing automático
    printf("\n<<<BENCHMARK_JSON>>>\n");
    printf("{\"device\":\"%s\",\"cores\":%d,\"cpu_mhz\":%d,\"features\":%d,\"hidden\":%d,\"outputs\":%d,\"time_steps\":%d,\"latency_mean_ms\":%.4f,\"latency_std_ms\":%.4f,\"throughput\":%.2f,\"warmup_runs\":%d,\"benchmark_runs\":%d}\n",
           chip_model,
           chip_info.cores,
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
           INPUT_SIZE,
           HIDDEN_SIZE,
           OUTPUT_SIZE,
           SNN_TIME_STEPS,
           latency_mean_ms,
           std_dev,
           throughput,
           WARMUP_RUNS,
           BENCHMARK_RUNS);
    printf("<<<END_BENCHMARK_JSON>>>\n\n");
}

/**
 * @brief Task principal de inferência
 */
void inference_task(void *arg) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "SNN Multimodal Water Quality Inference");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Model: 5 inputs -> 64 hidden LIF -> 3 outputs");
    ESP_LOGI(TAG, "Features: x, y, TEMP, COR_A, COR_V");
    ESP_LOGI(TAG, "Outputs: pH, OD, Turbidity");
    ESP_LOGI(TAG, "Dataset: %d samples from Serpa Lake", DATASET_SIZE);
    ESP_LOGI(TAG, "Time steps: %d, Beta: %.2f, Threshold: %.2f", 
             SNN_TIME_STEPS, SNN_BETA, SNN_THRESHOLD);
#if DETERMINISTIC_MODE
    ESP_LOGI(TAG, "Mode: DETERMINISTIC (reproducible results)");
#else
    ESP_LOGI(TAG, "Mode: ENSEMBLE (%d runs per sample)", ENSEMBLE_RUNS);
#endif
    ESP_LOGI(TAG, "========================================\n");
    
    // EXECUTA BENCHMARK PRIMEIRO
    run_benchmark();
    
    // Estatísticas globais
    float total_ph_error = 0.0f;
    float total_od_error = 0.0f;
    float total_turb_error = 0.0f;
    g_total_spikes = 0;  // Reset contador de spikes
    
    // Processa cada amostra
    for (int i = 0; i < DATASET_SIZE; i++) {
        // Prepara input
        float raw_input[INPUT_SIZE] = {
            dataset[i].x,
            dataset[i].y,
            dataset[i].temp,
            dataset[i].cor_a,
            dataset[i].cor_v
        };
        
        // Normaliza input
        float norm_input[INPUT_SIZE];
        normalize_input(raw_input, norm_input);
        
        // Executa inferência SNN
        float norm_output[OUTPUT_SIZE];
        snn_inference(norm_input, norm_output);
        
        // Denormaliza output
        float pred_output[OUTPUT_SIZE];
        denormalize_output(norm_output, pred_output);
        
        float pred_ph = pred_output[0];
        float pred_od = pred_output[1];
        float pred_turb = pred_output[2];
        
        // Ground truth
        float real_ph = dataset[i].ph_real;
        float real_od = dataset[i].od_real;
        float real_turb = dataset[i].turb_real;
        
        // Calcula erros
        float ph_error = fabsf(pred_ph - real_ph);
        float od_error = fabsf(pred_od - real_od);
        float turb_error = fabsf(pred_turb - real_turb);
        
        total_ph_error += ph_error;
        total_od_error += od_error;
        total_turb_error += turb_error;
        
        // Calcula índice de poluição
        float pollution_idx = calculate_pollution_index(pred_ph, pred_od, pred_turb);
        const char* level = classify_pollution_level(pollution_idx);
        
        // Log resultado
        printf("[Sample %2d] Pred: pH=%.2f OD=%.2f Turb=%.2f | "
               "Real: pH=%.2f OD=%.2f Turb=%.2f | "
               "Pollution: %.3f (%s)\n",
               i + 1,
               pred_ph, pred_od, pred_turb,
               real_ph, real_od, real_turb,
               pollution_idx, level);
        
        vTaskDelay(pdMS_TO_TICKS(100));  // Delay para visualização
    }
    
    // Estatísticas finais
    float mae_ph = total_ph_error / DATASET_SIZE;
    float mae_od = total_od_error / DATASET_SIZE;
    float mae_turb = total_turb_error / DATASET_SIZE;
    
    printf("\n");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "INFERENCE COMPLETE - ERROR METRICS (MAE)");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "pH MAE:        %.4f", mae_ph);
    ESP_LOGI(TAG, "OD MAE:        %.4f mg/L", mae_od);
    ESP_LOGI(TAG, "Turbidity MAE: %.4f NTU", mae_turb);
    ESP_LOGI(TAG, "========================================");
    
    // Estatísticas de spikes
    float avg_spikes_per_sample = (float)g_total_spikes / DATASET_SIZE;
    float firing_rate = 100.0f * g_total_spikes / (DATASET_SIZE * HIDDEN_SIZE * SNN_TIME_STEPS);
    ESP_LOGI(TAG, "Total spikes: %d (%.1f per sample, %.2f%% firing rate)", 
             g_total_spikes, avg_spikes_per_sample, firing_rate);
    
    // ============================================================
    // MODO CONTÍNUO: Demonstra que a inferência é dinâmica
    // A cada 5 segundos, faz inferência com inputs diferentes
    // ============================================================
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== CONTINUOUS INFERENCE MODE ===");
    ESP_LOGI(TAG, "Demonstrating real-time SNN predictions...");
    ESP_LOGI(TAG, "Each prediction uses the trained weights from .pt");
    ESP_LOGI(TAG, "");
    
    int demo_cycle = 0;
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(5000));  // 5 segundos entre inferências
        
        // Seleciona amostra rotativa do dataset
        int sample_idx = demo_cycle % DATASET_SIZE;
        
        // Prepara input da amostra selecionada
        float raw_input[INPUT_SIZE] = {
            dataset[sample_idx].x,
            dataset[sample_idx].y,
            dataset[sample_idx].temp,
            dataset[sample_idx].cor_a,
            dataset[sample_idx].cor_v
        };
        
        // Normaliza e executa inferência SNN
        float norm_input[INPUT_SIZE];
        normalize_input(raw_input, norm_input);
        
        g_total_spikes = 0;  // Reset para contar spikes desta inferência
        float norm_output[OUTPUT_SIZE];
        snn_inference(norm_input, norm_output);
        
        // Denormaliza output
        float pred_output[OUTPUT_SIZE];
        denormalize_output(norm_output, pred_output);
        
        // Calcula índice de poluição e classifica
        float pollution_idx = calculate_pollution_index(pred_output[0], pred_output[1], pred_output[2]);
        const char* level = classify_pollution_level(pollution_idx);
        
        // Exibe resultado detalhado
        printf("\n");
        ESP_LOGI(TAG, ">>> LIVE INFERENCE #%d (sample %d) <<<", demo_cycle + 1, sample_idx + 1);
        ESP_LOGI(TAG, "Input:  TEMP=%.1fC COR_A=%.0f COR_V=%.0f", 
                 dataset[sample_idx].temp, dataset[sample_idx].cor_a, dataset[sample_idx].cor_v);
        ESP_LOGI(TAG, "Output: pH=%.2f  OD=%.2f mg/L  Turb=%.2f NTU", 
                 pred_output[0], pred_output[1], pred_output[2]);
        ESP_LOGI(TAG, "Real:   pH=%.2f  OD=%.2f mg/L  Turb=%.2f NTU",
                 dataset[sample_idx].ph_real, dataset[sample_idx].od_real, dataset[sample_idx].turb_real);
        ESP_LOGI(TAG, "Spikes: %d  |  Pollution Index: %.3f  |  Level: ** %s **", 
                 g_total_spikes, pollution_idx, level);
        
        demo_cycle++;
    }
}

extern "C" void app_main(void) {
    printf("\n\n");
    printf("=================================================\n");
    printf("   ESP32 SNN Multimodal Water Quality Monitor\n");
    printf("   Serpa Lake - Amazon Region\n");
    printf("=================================================\n\n");
    
    // Cria task de inferência no core 1
    xTaskCreatePinnedToCore(
        inference_task,
        "snn_inference",
        8192,
        NULL,
        5,
        NULL,
        1
    );
    
    ESP_LOGI(TAG, "Inference task started on core 1");
}
