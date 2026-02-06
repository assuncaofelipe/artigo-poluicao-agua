# Benchmark de Energia e Latência (Raspberry Pi 4B)

Este diretório contém o script mínimo para medir **latência de inferência** do modelo
TFLite na Raspberry Pi 4B e registrar informações do sistema para o artigo.

## 1) O que você precisa copiar para a Raspberry

Copie estes arquivos para a Raspberry Pi (exemplo com `scp`):

```bash
scp energy_benchmark/bench_tflite.py pi@IP_DA_RASPBERRY:/home/pi/energy_benchmark/
scp energy_benchmark/requirements.txt pi@IP_DA_RASPBERRY:/home/pi/energy_benchmark/
scp serpa_predict/assets/model/model.tflite pi@IP_DA_RASPBERRY:/home/pi/energy_benchmark/
scp serpa_predict/assets/model/norm_multi.json pi@IP_DA_RASPBERRY:/home/pi/energy_benchmark/
```

## 2) Rodar na Raspberry Pi

```bash
cd /home/pi/energy_benchmark
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python bench_tflite.py
```

Se aparecer erro do NumPy 2.x com `tflite-runtime`, confirme que o `requirements.txt`
está com `numpy<2` e reinstale:

```bash
pip uninstall -y numpy
pip install "numpy<2"
```

## 3) Saída esperada

O script imprime:

- Latência média (ms por amostra)
- Desvio padrão (ms)
- Throughput (amostras/s)
- Info do hardware

Esses valores podem ser usados direto no artigo.
