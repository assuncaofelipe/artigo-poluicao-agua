#!/usr/bin/env python3
"""
Exporta os artefatos C++ atuais do ESP32 para um arquivo model_data.py
compatível com MicroPython.

Fonte:
- esp32_snn_water/main/weights.h
- esp32_snn_water/main/model_config.h
- esp32_snn_water/main/main.cpp (dataset)
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_H = ROOT / "esp32_snn_water" / "main" / "weights.h"
MODEL_CONFIG_H = ROOT / "esp32_snn_water" / "main" / "model_config.h"
MAIN_CPP = ROOT / "esp32_snn_water" / "main" / "main.cpp"
OUT_FILE = ROOT / "esp32_micropython" / "model_data.py"


def parse_c_number(token: str) -> float:
    return float(token.strip().rstrip("f"))


def extract_define_int(content: str, name: str) -> int:
    m = re.search(rf"#define\s+{name}\s+(-?\d+)", content)
    if not m:
        raise ValueError(f"Define nao encontrado: {name}")
    return int(m.group(1))


def extract_define_float(content: str, name: str) -> float:
    m = re.search(rf"#define\s+{name}\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)f?", content)
    if not m:
        raise ValueError(f"Define nao encontrado: {name}")
    return float(m.group(1))


def extract_c_array(content: str, name: str, expected_size: int | None = None) -> list[float]:
    pattern = rf"const\s+float\s+{name}\s*\[[^\]]+\](?:\[[^\]]+\])?\s*=\s*\{{(.*?)\}};"
    m = re.search(pattern, content, re.DOTALL)
    if not m:
        raise ValueError(f"Array nao encontrado: {name}")
    raw = re.sub(r"[{}]", "", m.group(1))
    values = [parse_c_number(tok) for tok in raw.split(",") if tok.strip()]
    if expected_size is not None and len(values) != expected_size:
        raise ValueError(
            f"Tamanho inesperado para {name}: {len(values)} (esperado {expected_size})"
        )
    return values


def reshape_2d(flat: list[float], rows: int, cols: int) -> list[list[float]]:
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def extract_dataset(main_cpp: str) -> list[list[float]]:
    block_match = re.search(
        r"static\s+const\s+DataPoint\s+dataset\[\]\s*=\s*\{(.*?)\};",
        main_cpp,
        re.DOTALL,
    )
    if not block_match:
        raise ValueError("Bloco dataset[] nao encontrado em main.cpp")

    block = block_match.group(1)
    rows = []
    for row_match in re.finditer(r"\{([^{}]+)\}", block):
        parts = [p.strip() for p in row_match.group(1).split(",") if p.strip()]
        if len(parts) < 8:
            continue
        rows.append([parse_c_number(p) for p in parts[:8]])
    if not rows:
        raise ValueError("Nenhuma linha do dataset foi extraida")
    return rows


def fmt_tuple(values: list[float], indent: str = "    ", per_line: int = 6) -> str:
    out = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        out.append(indent + ", ".join(f"{v:.8f}" for v in chunk))
    return ",\n".join(out)


def fmt_2d(values: list[list[float]], indent: str = "    ") -> str:
    lines = []
    for row in values:
        lines.append(f"{indent}({', '.join(f'{v:.8f}' for v in row)}),")
    return "\n".join(lines)


def read_text_auto(path: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("auto", b"", 0, 1, f"Falha de decode em {path}")


def main() -> None:
    weights_txt = read_text_auto(WEIGHTS_H)
    cfg_txt = read_text_auto(MODEL_CONFIG_H)
    main_cpp_txt = read_text_auto(MAIN_CPP)

    input_size = extract_define_int(weights_txt, "INPUT_SIZE")
    hidden_size = extract_define_int(weights_txt, "HIDDEN_SIZE")
    output_size = extract_define_int(weights_txt, "OUTPUT_SIZE")

    w1_flat = extract_c_array(weights_txt, "W1", hidden_size * input_size)
    b1 = extract_c_array(weights_txt, "B1", hidden_size)
    w2_flat = extract_c_array(weights_txt, "W2", output_size * hidden_size)
    b2 = extract_c_array(weights_txt, "B2", output_size)

    w1 = reshape_2d(w1_flat, hidden_size, input_size)
    w2 = reshape_2d(w2_flat, output_size, hidden_size)

    snn_time_steps = extract_define_int(cfg_txt, "SNN_TIME_STEPS")
    snn_beta = extract_define_float(cfg_txt, "SNN_BETA")
    snn_threshold = extract_define_float(cfg_txt, "SNN_THRESHOLD")

    norm_mean = [
        extract_define_float(cfg_txt, "NORM_MEAN_X"),
        extract_define_float(cfg_txt, "NORM_MEAN_Y"),
        extract_define_float(cfg_txt, "NORM_MEAN_TEMP"),
        extract_define_float(cfg_txt, "NORM_MEAN_COR_A"),
        extract_define_float(cfg_txt, "NORM_MEAN_COR_V"),
    ]
    norm_std = [
        extract_define_float(cfg_txt, "NORM_STD_X"),
        extract_define_float(cfg_txt, "NORM_STD_Y"),
        extract_define_float(cfg_txt, "NORM_STD_TEMP"),
        extract_define_float(cfg_txt, "NORM_STD_COR_A"),
        extract_define_float(cfg_txt, "NORM_STD_COR_V"),
    ]
    out_mean = [
        extract_define_float(cfg_txt, "OUT_MEAN_PH"),
        extract_define_float(cfg_txt, "OUT_MEAN_OD"),
        extract_define_float(cfg_txt, "OUT_MEAN_TURBIDITY"),
    ]
    out_std = [
        extract_define_float(cfg_txt, "OUT_STD_PH"),
        extract_define_float(cfg_txt, "OUT_STD_OD"),
        extract_define_float(cfg_txt, "OUT_STD_TURBIDITY"),
    ]

    dataset = extract_dataset(main_cpp_txt)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Auto-generated by export_multimodal_to_micropython.py\n")
        f.write("# Do not edit manually.\n\n")
        f.write(f"INPUT_SIZE = {input_size}\n")
        f.write(f"HIDDEN_SIZE = {hidden_size}\n")
        f.write(f"OUTPUT_SIZE = {output_size}\n\n")
        f.write(f"SNN_TIME_STEPS = {snn_time_steps}\n")
        f.write(f"SNN_BETA = {snn_beta:.8f}\n")
        f.write(f"SNN_THRESHOLD = {snn_threshold:.8f}\n\n")

        f.write("NORM_MEAN = (\n")
        f.write(fmt_tuple(norm_mean, indent="    ", per_line=5) + ",\n")
        f.write(")\n")
        f.write("NORM_STD = (\n")
        f.write(fmt_tuple(norm_std, indent="    ", per_line=5) + ",\n")
        f.write(")\n")
        f.write("OUT_MEAN = (\n")
        f.write(fmt_tuple(out_mean, indent="    ", per_line=3) + ",\n")
        f.write(")\n")
        f.write("OUT_STD = (\n")
        f.write(fmt_tuple(out_std, indent="    ", per_line=3) + ",\n")
        f.write(")\n\n")

        f.write("W1 = (\n")
        f.write(fmt_2d(w1))
        f.write("\n)\n\n")
        f.write("B1 = (\n")
        f.write(fmt_tuple(b1, indent="    ", per_line=6) + ",\n")
        f.write(")\n\n")
        f.write("W2 = (\n")
        f.write(fmt_2d(w2))
        f.write("\n)\n\n")
        f.write("B2 = (\n")
        f.write(fmt_tuple(b2, indent="    ", per_line=3) + ",\n")
        f.write(")\n\n")

        f.write("# x, y, temp, cor_a, cor_v, ph_real, od_real, turb_real\n")
        f.write("DATASET = (\n")
        for row in dataset:
            f.write("    (" + ", ".join(f"{v:.8f}" for v in row) + "),\n")
        f.write(")\n")

    print(f"OK: {OUT_FILE}")
    print(
        f"Modelo exportado para MicroPython: "
        f"{input_size}->{hidden_size}->{output_size}, dataset={len(dataset)}"
    )


if __name__ == "__main__":
    main()
