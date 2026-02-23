"""
SNN inference para ESP32 com MicroPython.
Carrega pesos de model_weights.json

Uso:
    >>> import main
    >>> main.benchmark()
    >>> pred, spikes = main.inference([x, y, temp, cor_a, cor_v])
"""

import math
import time
import gc

try:
    import ujson as json
except ImportError:
    import json

# ==============================================================================
# GLOBALS (carregadas do modelo)
# ==============================================================================

INPUT_SIZE = 5
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3
TIME_STEPS = 15
BETA = 0.95
THRESH = 0.8

NORM_MEAN = None
NORM_STD = None
OUT_MEAN = None
OUT_STD = None
W1 = None
B1 = None
W2 = None
B2 = None

_model_loaded = False

# ==============================================================================
# TIMING (compatível MicroPython)
# ==============================================================================

def _ticks_us():
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.time() * 1_000_000)


def _ticks_diff(new, old):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(new, old)
    return new - old


# ==============================================================================
# CARREGAMENTO
# ==============================================================================

def load_json(path="model_weights.json"):
    """Carrega modelo de arquivo JSON com fallback para múltiplos caminhos"""
    global INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    global TIME_STEPS, BETA, THRESH
    global NORM_MEAN, NORM_STD, OUT_MEAN, OUT_STD
    global W1, B1, W2, B2, _model_loaded
    
    gc.collect()
    
    # Tenta vários caminhos possíveis (compatível com MicroPython)
    paths_to_try = [
        path,
        "model_weights.json",
        "/model_weights.json",
    ]
    
    file_content = None
    loaded_from = None
    last_error = None
    
    for try_path in paths_to_try:
        try:
            with open(try_path, 'r') as f:
                file_content = f.read()
            loaded_from = try_path
            break
        except (FileNotFoundError, OSError) as e:
            last_error = e
            continue
    
    if file_content is None:
        raise FileNotFoundError(
            f"Nao encontrou model_weights.json em:\n" +
            "  " + "\n  ".join(paths_to_try) + "\n" +
            f"Erro: {last_error}"
        )
    
    try:
        d = json.loads(file_content)
    except Exception as e:
        raise Exception(f"Erro ao parsear JSON: {e}")
    
    # Extrai campos do JSON
    try:
        INPUT_SIZE = d["input_size"]
        HIDDEN_SIZE = d["hidden_size"]
        OUTPUT_SIZE = d["output_size"]
        TIME_STEPS = d["time_steps"]
        BETA = d["beta"]
        THRESH = d["thresh"]
        NORM_MEAN = d["norm_mean"]
        NORM_STD = d["norm_std"]
        OUT_MEAN = d["out_mean"]
        OUT_STD = d["out_std"]
        W1 = d["W1"]
        B1 = d["B1"]
        W2 = d["W2"]
        B2 = d["B2"]
    except KeyError as e:
        raise KeyError(f"Campo ausente no JSON: {e}")
    
    _model_loaded = True
    gc.collect()
    print(f"[OK] Modelo: {INPUT_SIZE}->{HIDDEN_SIZE}->{OUTPUT_SIZE} de {loaded_from}")
    return True


def auto_load():
    """Tenta carregar modelo. Falha explicitamente se nao conseguir."""
    global _model_loaded
    if _model_loaded:
        return True
    
    try:
        load_json("model_weights.json")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}")
        return False


# ==============================================================================
# INFERÊNCIA
# ==============================================================================

def _sigmoid(x):
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def inference(raw_input):
    """
    Inferencia completa: normaliza -> SNN -> desnormaliza
    
    Args:
        raw_input: [x, y, temp, cor_a, cor_v]
        
    Returns:
        ([pH, OD, Turb], spikes)
    """
    if not _model_loaded:
        if not auto_load():
            raise RuntimeError("Falha: modelo nao foi carregado! Verifique se model_weights.json existe.")
    
    # Normaliza entrada
    norm_in = [(raw_input[i] - NORM_MEAN[i]) / NORM_STD[i] for i in range(INPUT_SIZE)]
    
    # Estados
    mem = [0.0] * HIDDEN_SIZE
    out_acc = [0.0] * OUTPUT_SIZE
    spikes = 0
    
    for _ in range(TIME_STEPS):
        # Input encoding
        spike_in = [_sigmoid(norm_in[i]) for i in range(INPUT_SIZE)]
        
        # Hidden layer (fc1 + LIF)
        h_spk = [0.0] * HIDDEN_SIZE
        for h in range(HIDDEN_SIZE):
            i_syn = B1[h]
            for i in range(INPUT_SIZE):
                i_syn += W1[h][i] * spike_in[i]
            
            v = BETA * mem[h] + i_syn
            if v > THRESH:
                h_spk[h] = 1.0
                spikes += 1
                v -= THRESH
            mem[h] = v
        
        # Output layer (fc2)
        for o in range(OUTPUT_SIZE):
            acc = B2[o]
            for h in range(HIDDEN_SIZE):
                acc += W2[o][h] * h_spk[h]
            out_acc[o] += acc
    
    # Media temporal e desnormaliza
    norm_out = [out_acc[o] / TIME_STEPS for o in range(OUTPUT_SIZE)]
    pred = [norm_out[i] * OUT_STD[i] + OUT_MEAN[i] for i in range(OUTPUT_SIZE)]
    
    return pred, spikes
    
    # Estados
    mem = [0.0] * HIDDEN_SIZE
    out_acc = [0.0] * OUTPUT_SIZE
    spikes = 0
    
    for _ in range(TIME_STEPS):
        # Input encoding
        spike_in = [_sigmoid(norm_in[i]) for i in range(INPUT_SIZE)]
        
        # Hidden layer (fc1 + LIF)
        h_spk = [0.0] * HIDDEN_SIZE
        for h in range(HIDDEN_SIZE):
            i_syn = B1[h]
            for i in range(INPUT_SIZE):
                i_syn += W1[h][i] * spike_in[i]
            
            v = BETA * mem[h] + i_syn
            if v > THRESH:
                h_spk[h] = 1.0
                spikes += 1
                v -= THRESH
            mem[h] = v
        
        # Output layer (fc2)
        for o in range(OUTPUT_SIZE):
            acc = B2[o]
            for h in range(HIDDEN_SIZE):
                acc += W2[o][h] * h_spk[h]
            out_acc[o] += acc
    
    # Média temporal e desnormaliza
    norm_out = [out_acc[o] / TIME_STEPS for o in range(OUTPUT_SIZE)]
    pred = [norm_out[i] * OUT_STD[i] + OUT_MEAN[i] for i in range(OUTPUT_SIZE)]
    
    return pred, spikes


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark(warmup=10, runs=500, test_input=None):
    """
    Benchmark de performance.
    
    Retorna dict com latencia (ms), throughput (inf/s), predicoes.
    """
    if not auto_load():
        raise RuntimeError("Falha ao carregar modelo!")
    
    if test_input is None:
        test_input = [335575.0, 9657300.0, 32.0, 90.0, 30.0]
    
    gc.collect()
    mem_before = gc.mem_free() if hasattr(gc, 'mem_free') else 0
    
    print("\n=== BENCHMARK ESP32 SNN ===")
    print(f"Arch: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print(f"Steps: {TIME_STEPS}, Beta: {BETA}, Thresh: {THRESH}")
    print(f"Warmup: {warmup}, Runs: {runs}")
    
    # Warmup
    for _ in range(warmup):
        inference(test_input)
    
    gc.collect()
    
    # Benchmark com coleta de amostras
    samples = []
    for _ in range(runs):
        s0 = _ticks_us()
        inference(test_input)
        s1 = _ticks_us()
        samples.append(_ticks_diff(s1, s0) / 1000.0)
    
    lat_mean = sum(samples) / len(samples)
    lat_min = min(samples)
    lat_max = max(samples)
    variance = sum((s - lat_mean) ** 2 for s in samples) / len(samples)
    lat_std = variance ** 0.5
    throughput = runs / (sum(samples) / 1000.0)
    
    # Predicao de teste
    pred, spikes = inference(test_input)
    
    gc.collect()
    mem_after = gc.mem_free() if hasattr(gc, 'mem_free') else 0
    
    print("\n--- Resultados ---")
    print(f"Latencia:   {lat_mean:.3f} +- {lat_std:.3f} ms")
    print(f"Min/Max:    {lat_min:.3f} / {lat_max:.3f} ms")
    print(f"Throughput: {throughput:.1f} inf/s")
    print(f"Spikes:     {spikes} ({100.0*spikes/(TIME_STEPS*HIDDEN_SIZE):.1f}%)")
    print(f"pH:         {pred[0]:.3f}")
    print(f"OD:         {pred[1]:.3f}")
    print(f"Turbidez:   {pred[2]:.3f}")
    if mem_before > 0:
        print(f"Memoria:    {mem_after} bytes")
    
    return {
        "latency_ms": lat_mean,
        "latency_std": lat_std,
        "latency_min": lat_min,
        "latency_max": lat_max,
        "throughput": throughput,
        "spikes": spikes,
        "predictions": pred,
        "mem_free": mem_after
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    benchmark()
