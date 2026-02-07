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

Se aparecer erro `AttributeError: module 'serial' has no attribute 'Serial'`,
é porque o pacote errado foi instalado. Corrija assim:

```bash
pip uninstall -y serial
pip install pyserial
```

## 3) Medição de Energia (corrente/tensão)

**Aviso ACS712:** a sensibilidade típica é ~66 mV/A para o modelo 30A e ~100 mV/A
para o modelo 20A. No seu código Arduino, use `SENSIBILIDADE = 0.066` (30A) ou
`SENSIBILIDADE = 0.100` (20A), conforme o sensor.

O script `bench_energy.py` mede energia média por inferência usando três modos:

- `sysfs`: lê `current_now` e `voltage_now` em `/sys/class/power_supply`
- `csv`: lê um CSV com colunas `timestamp,current_mA,voltage_V`
- `manual`: usa valores fixos informados via CLI

### a) Modo sysfs (padrão)

```bash
python bench_energy.py --provider sysfs \
  --current-unit uA --voltage-unit uV
```

Se os caminhos forem diferentes, use:

```bash
python bench_energy.py --provider sysfs \
  --current-path /sys/class/power_supply/BAT0/current_now \
  --voltage-path /sys/class/power_supply/BAT0/voltage_now \
  --current-unit uA --voltage-unit uV
```

### b) Modo CSV (ex.: medidor externo exportando)

```bash
python bench_energy.py --provider csv --csv power_samples.csv
```

### c) Modo manual (valor fixo)

```bash
python bench_energy.py --provider manual --current-mA 400 --voltage-V 5.0
```

Saída: `results_energy.csv` com timestamp e energia por inferência.

## 4) Log via Arduino (ACS712) -> CSV

Se você usa o ACS712 com Arduino, rode o logger serial e depois alimente o
`bench_energy.py` com `--provider csv`.

### a) Log serial

```bash
python serial_power_logger.py --port /dev/ttyUSB0 --baud 9600 --voltage 5.0
```

Se o Arduino reiniciar ao abrir a porta, use:

```bash
python serial_power_logger.py --port /dev/ttyACM0 --baud 9600 --voltage 5.0 \
  --no-reset --skip-seconds 10
```

Isso gera `power_samples.csv` com `timestamp,current_mA,voltage_V`.

### b) Energia a partir do CSV

```bash
python bench_energy.py --provider csv --csv power_samples.csv
```

### b.1) Converter log bruto do Arduino em CSV

Se você já tem o texto bruto do Serial Monitor, salve em `log.txt` e converta:

```bash
python parse_arduino_log.py --in log.txt --out power_samples_parsed.csv --summary power_summary.csv
```

Você pode forçar tensão fixa com `--voltage 5.0` se preferir.

### c) Energia incremental (idle vs carga)

Se você mede corrente em idle e em carga separadamente, salve em dois CSVs
(`power_samples_idle.csv` e `power_samples_load.csv`) e calcule a energia
incremental usando o throughput medido:

```bash
python compute_energy_from_csvs.py \
  --idle power_samples_idle.csv \
  --load power_samples_load.csv \
  --throughput 11639.71 \
  --out results_energy_manual.csv
```

## 5) Android: log de corrente via ADB

Se o seu Android expõe `current_now` e `voltage_now`, você pode registrar via ADB:

```bash
python adb_power_logger.py --out android_power_samples.csv --interval 0.5 --duration 60
```

Depois envie o CSV para eu calcular a média e a energia por inferência.

## 3) Saída esperada

O script imprime:

- Latência média (ms por amostra)
- Desvio padrão (ms)
- Throughput (amostras/s)
- Info do hardware

Esses valores podem ser usados direto no artigo.
