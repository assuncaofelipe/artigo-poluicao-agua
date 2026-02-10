# SNN Water Quality Monitoring on ESP32

## 📋 Visão Geral

Sistema completo de monitoramento de qualidade de água usando **Spiking Neural Networks (SNN)** com neurônios Leaky Integrate-and-Fire (LIF) em um microcontrolador ESP32.

### Arquitetura

```
PyTorch Model (grid_best.pt)
        ↓
   export_to_c_enhanced.py
        ↓
   ┌─────────────────┐
   │  weights.h      │
   │ model_config.h  │
   └────────┬────────┘
            ↓
    esp32_snn_water/
        ↓
    ┌──────────────────┐
    │   main.cpp       │  (FreeRTOS tasks)
    │   snn.h          │  (SNN engine)
    │   CMakeLists.txt │  (ESP-IDF build)
    └──────────────────┘
```

## 🔄 Pipeline Completo

### 1. **Extração de Pesos (Python)**

O script `export_to_c_enhanced.py` executa:

```bash
cd embarcado
python export_to_c_enhanced.py
```

**O que faz:**
- Carrega `../outputs/grid_best.pt` (checkpoint PyTorch)
- Extrai pesos (W1, B1, W2, B2) das camadas fully-connected
- Carrega estatísticas de normalização de `../outputs/norm.json`
- Gera C headers estáticos:
  - `main/weights.h` → Arrays const com todos os pesos
  - `main/model_config.h` → Hiperparâmetros e estatísticas

**Estrutura do modelo extraído:**
```
Input Size:    7 features (x, y, TEMP, COR_A, TURB, COR_V, OD)
Hidden Layer:  64 neurônios LIF
Output Size:   1 (predição de pH)
```

### 2. **SNN Engine (C++ em main/snn.h)**

Motor de inferência neuromorfo implementado como uma classe C++ pura:

```cpp
class SpikingNeuralNetwork {
    // Forward pass com LIF neurons
    // v(t+1) = beta * v(t) + input
    // spike = v(t) >= threshold
};
```

**Características:**
- ✅ Zero alocação dinâmica (buffers estáticos)
- ✅ Determinístico (sem randomicidade)
- ✅ Otimizado para ESP32 (sem floating-point pesado)
- ✅ Tempo de integração configurável (T steps)

**Inferência completa:**
```
Input bruto (sensores)
    ↓
Z-score normalization
    ↓
Injeção de corrente (W1 @ input + B1)
    ↓
T passos de integração LIF
    ↓
Acúmulo de spikes (neuron firing counts)
    ↓
Readout linear (W2 @ spike_count + B2)
    ↓
Output (predição)
```

### 3. **Firmware ESP32 (esp32_snn_water/)**

Aplicação FreeRTOS com múltiplas tasks:

```cpp
// Task 1: Inferência contínua
inference_task() {
    - Lê sensores (pH, temperatura, turbidez)
    - Valida dados
    - Executa SNN.infer()
    - Calcula pollution_index
    - Classifica (LOW/MEDIUM/HIGH)
    - Loga resultados
}

// Task 2: Calibração de sensores
calibration_task() {
    - Lê múltiplos pontos
    - Detecta drift
    - Atualiza zero/span
}
```

## 📐 Dimensões e Parâmetros

| Parâmetro | Valor |
|-----------|-------|
| Input Features | 7 |
| Hidden Neurons (LIF) | 64 |
| Output Size | 1 |
| SNN Time Steps | 10 |
| LIF Beta (decay) | 0.95 |
| LIF Threshold | 1.0 |

### Estatísticas de Normalização

Extraídas do `norm.json`:

```c
// Temperature normalization
NORM_MEAN_TEMP = 31.7905f
NORM_STD_TEMP = 0.3900f

// Turbidity normalization
NORM_MEAN_TURB = 2.5274f
NORM_STD_TURB = 1.6248f

// pH (padrão, não está nas features)
NORM_MEAN_PH = 6.5f
NORM_STD_PH = 1.0f
```

## 🚀 Como Compilar e Executar

### Pré-requisitos

```
- ESP-IDF toolchain (v5.0+)
- VS Code + ESP-IDF extension
- CMake 3.16+
- Python 3.8+
```

### Passos

1. **Gerar headers do PyTorch:**
   ```bash
   cd embarcado/
   python export_to_c_enhanced.py
   ```
   → Cria `main/weights.h` e `main/model_config.h`

2. **Compilar para ESP32:**
   ```bash
   cd esp32_snn_water/
   idf.py set-target esp32
   idf.py build
   ```

3. **Flashear no dispositivo:**
   ```bash
   idf.py -p /dev/ttyUSB0 flash monitor
   ```

4. **Ver logs:**
   ```
   [SNN_INFERENCE] Iniciando SNN Water Quality Monitor
   [SNN_INFERENCE] Inicializando rede neural...
   [SNN_INFERENCE] ✓ SNN inicializado
   [SNN_INFERENCE] Predições: pH=6.8234, Temp=26.4521, Turb=3.1892
   [SNN_INFERENCE] Poluição: Index=0.3421 (MEDIUM)
   ```

## 📊 Índice de Poluição

### Cálculo

```
pollution_index = weighted_sum(
    0.33 * |pH - 7.0| / 7.0,           # Desvio de pH neutro
    0.33 * temperature / 32.0,          # Normalizado por limite
    0.34 * turbidity / 1.0              # Turbidez normalizada
)
```

### Classificação

| Índice | Nível | Status |
|--------|-------|--------|
| 0.0 - 0.33 | **LOW** | 🟢 Água segura |
| 0.33 - 0.67 | **MEDIUM** | 🟡 Atenção necessária |
| 0.67 - 1.0 | **HIGH** | 🔴 Altamente poluída |

## 💾 Estrutura de Arquivos

```
embarcado/
├── export_to_c.py              # Script básico (backup)
├── export_to_c_enhanced.py     # Script principal
├── inspect_checkpoint.py       # Debug do checkpoint
├── main/
│   ├── snn.h                   # Engine SNN (classe C++)
│   ├── main.cpp                # Aplicação FreeRTOS
│   ├── weights.h               # ⚙ Gerado automaticamente
│   ├── model_config.h          # ⚙ Gerado automaticamente
│   ├── CMakeLists.txt
│   └── model_config.h          # Original (pode sobrescrever)
└── esp32_snn_water/
    ├── CMakeLists.txt          # Build config
    ├── sdkconfig               # ESP-IDF config
    └── main/
        ├── snn.h               # Motor neuromorfo
        ├── main.cpp            # Firmware principal
        └── model_config.h      # Metadados do modelo
```

## 🔧 Customização

### Alterar período de inferência

Em `main/main.cpp`:
```cpp
const uint32_t inference_period_ms = 1000;  // 1000ms = 1s
```

### Ajustar parâmetros SNN

Em `main/model_config.h` (gerado):
```cpp
#define SNN_TIME_STEPS      10      // Aumentar para mais precisão
#define SNN_BETA            0.95f   // Aumentar para memória mais longa
#define SNN_THRESHOLD       1.0f    // Aumentar para menos spikes
```

### Modificar limites de poluição

Em `main/model_config.h`:
```cpp
#define POLLUTION_THRESHOLD_LOW    0.33f
#define POLLUTION_THRESHOLD_HIGH   0.67f
```

## 🐛 Debug e Benchmarking

### Modo verbose

Em `main/main.cpp`:
```cpp
ESP_LOGI(TAG, "Debug: membrane[0]=%.4f, spike_count[0]=%.0f",
         snn.get_state().membrane[0],
         snn.get_state().spike_count[0]);
```

### Medir latência

```cpp
uint32_t ts_start = esp_timer_get_time();
snn.infer(input, output);
uint32_t ts_end = esp_timer_get_time();
printf("Latência: %ld µs", ts_end - ts_start);
```

## 📚 Referências

### Papers
- **SNN Basics**: Gerstner & Kistler, "Spiking Neuron Models" (2002)
- **LIF Neurons**: Lapicque's equation, membrane potential dynamics
- **Neuromorphic Computing**: IBM TrueNorth, Intel Loihi

### Documentação
- ESP-IDF: https://docs.espressif.com/projects/esp-idf/
- FreeRTOS: https://www.freertos.org/
- PyTorch: https://pytorch.org/docs/

## 📝 Notas Importantes

### ⚠ Restrições Hard
- **Sem malloc/new**: Todos os buffers são estáticos
- **Sem dynamic allocation**: Fixed-size arrays
- **Determinístico**: Sem randomicidade em inferência
- **Offline**: Sem conexão necessária para funcionar

### ✅ Boas Práticas
1. **Sempre reexecute `export_to_c_enhanced.py`** após retreinar o modelo
2. **Valide os ranges de entrada** antes de passar para SNN.infer()
3. **Use FreeRTOS tasks** para não bloquear o loop principal
4. **Logue timestamps** para análise de latência

### 🎯 Otimizações Futuras
- [ ] Quantização INT8 para pesos
- [ ] Lookup tables (LUT) para ReLU/sigmoid
- [ ] SIMD assembly para operações de matriz
- [ ] DMA para I2C/ADC
- [ ] Ultra-low power mode (ULP coprocessor)

## 📦 Dependências

### Python
```
torch>=1.9.0
numpy>=1.20.0
```

### C++
```
<cstring>
<cmath>
freertos/FreeRTOS.h
freertos/task.h
esp_log.h
```

## 📞 Suporte

Para problemas:
1. Verifique `inspect_checkpoint.py` para validar estrutura do modelo
2. Compile com `idf.py build -v` para mensagens detalhadas
3. Monitore logs com `idf.py monitor`

---

**Última atualização**: Fevereiro 2026
**Versão**: 1.0.0
**Status**: Produção
