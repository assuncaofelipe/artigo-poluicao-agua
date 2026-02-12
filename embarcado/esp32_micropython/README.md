# ESP32 MicroPython - Inferência SNN em Python Puro

Inferência de rede neural spiking (SNN) para ESP32 usando **Python puro** — sem PyTorch, sem NumPy.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Script de inferência e benchmark |
| `model_weights.json` | Pesos e configuração do modelo (12.7 KB) |
| `requirements.txt` | Documentação de dependências |
| `setup_esp32.bat` | Setup automático (Windows) |
| `setup_esp32.sh` | Setup automático (Linux/Mac) |

---

## Instruções Passo a Passo (Linux)

### 1. Instalar MicroPython na ESP32

```bash
# Instalar ferramenta de flash
pip install esptool

# Baixar firmware MicroPython
wget https://micropython.org/resources/firmware/esp32-20240105-v1.22.0.bin

# Flash na ESP32 (trocar /dev/ttyUSB0 pela sua porta)
esptool.py --chip esp32 --port /dev/ttyUSB0 erase_flash
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-20240105-v1.22.0.bin
```

### 2. Instalar ferramentas de upload

```bash
pip install mpremote
# ou
pip install adafruit-ampy
```

### 3. Fazer upload dos arquivos

**Opção A: Com script automático**
```bash
bash setup_esp32.sh /dev/ttyUSB0
```

**Opção B: Manual com mpremote**
```bash
mpremote connect /dev/ttyUSB0 fs cp main.py :main.py
mpremote connect /dev/ttyUSB0 fs cp model_weights.json :model_weights.json
```

**Opção C: Manual com ampy**
```bash
ampy --port /dev/ttyUSB0 put main.py
ampy --port /dev/ttyUSB0 put model_weights.json
```

### 4. Testar na ESP32

```bash
# Conectar ao terminal interativo
mpremote connect /dev/ttyUSB0
```

No REPL da ESP32:
```python
>>> import main
>>> main.benchmark()           # Roda benchmark completo
>>> pred, spikes = main.inference([335575.0, 9657300.0, 32.0, 90.0, 30.0])
>>> print(f"pH: {pred[0]:.3f}, OD: {pred[1]:.3f}, Turb: {pred[2]:.3f}")
```

---

## Resultados Reais (Benchmarked)

### PC (Python Puro)
```
=== BENCHMARK SNN PYTHON PURO ===
Arquitetura: 5 -> 64 -> 3
Time steps: 15, Beta: 0.95, Thresh: 0.8

--- Resultados ---
Latência:   0.434 ± 0.548 ms
Min/Max:    0.000 / 3.007 ms
Throughput: 2304.2 inf/s
Spikes:     136 (14.2%)
pH:         5.533
OD:         1.791
Turbidez:   2.453
```

### ESP32 (MicroPython)
```
=== BENCHMARK ESP32 SNN ===
Arch: 5 -> 64 -> 3
Steps: 15, Beta: 0.95, Thresh: 0.8

--- Resultados ---
Latência:   183.776 ± 9.448 ms
Min/Max:    170.345 / 198.147 ms
Throughput: 5.4 inf/s
Spikes:     136 (14.2%)
pH:         5.533
OD:         1.791
Turbidez:   2.453
Memória:    133,360 bytes
```

### Comparação

| Métrica | PC | ESP32 | Razão |
|---------|-------|-----------|-------|
| **Latência** | 0.43 ms | 183.78 ms | 428x |
| **Throughput** | 2,304 inf/s | 5.4 inf/s | 427x |
| **Predições** | 5.533, 1.791, 2.453 | 5.533, 1.791, 2.453 | ✅ Idênticas |
| **Spikes** | 136 | 136 | ✅ Iguais |

**Conclusão:** O modelo roda corretamente na ESP32 com as mesmas predições que no PC, 
confirmando que a implementação Python pura é fiel ao modelo original.

---

## Setup Rápido (Windows)

```bash
setup_esp32.bat COM4
```

---

## Ferramentas (pasta `../tools/`)

| Script | Descrição |
|--------|-----------|
| `snn_pure.py` | Benchmark Python puro (PC) |
| `benchmark_pytorch_vs_pure.py` | Compara PyTorch vs Python puro |
| `export_multimodal_to_c.py` | Exporta modelo para C++ (ESP-IDF) |
| `export_multimodal_to_micropython.py` | Regenera model_data.py |

---

## Resolução de Problemas

### `FileNotFoundError: Nao encontrou model_weights.json`
- Verifique que `model_weights.json` foi feito upload com sucesso
- Na ESP32, liste os arquivos: `>>> import os; print(os.listdir())`

### `'module' object has no attribute 'path'` 
- MicroPython não tem `os.path` completo
- Já foi corrigido na versão atual do `main.py`

### Porta COM não encontrada
- Linux: `ls /dev/ttyUSB*` ou `ls /dev/ttyACM*`
- Verifique permissões: `sudo chmod 666 /dev/ttyUSB0`

---

## Observações

- Latência esperada na ESP32: ~170-200 ms por inferência (na prática)
- Throughput: ~5-6 inf/s na ESP32
- O modelo roda totalmente em Python puro, sem dependências externas
- Memória usada: ~133 KB (cabe facilmente em ESP32 com 4 MB PSRAM)
