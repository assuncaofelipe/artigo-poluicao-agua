# Projeto Lago do Serpa — Artigo, Modelos e Deploy

Este repositório reúne o artigo, os scripts de treinamento/avaliação, o pipeline
multimodal, o app Flutter e o fluxo de deploy para ESP32 e benchmarks de energia.

## Estrutura
- `artigo/`: LaTeX do artigo (`main.tex`) + PDFs gerados
- `scripts/`: pipeline SNN single-target (pH/OD/TURB), figures 600 dpi
- `scripts_multimodal/`: pipeline multi-saida (pH, OD, TURB), export TFLite
- `serpa_predict/`: app Flutter offline com inferência TFLite
- `embarcado/`: export para C/ESP32 + firmware ESP-IDF
- `energy_benchmark/`: benchmarks de latência/energia (Raspberry/Android/Arduino)
- `outputs/`: modelos e artefatos gerados (ex.: `grid_best.pt`, `norm.json`)
- `figuras/`: imagens finais (PNG 600 dpi)
- `Bibliografias/`: PDFs usados na revisão bibliográfica
- `objetivos.tex` / `objetivos.pdf`: objetivos do trabalho
- `ACKNOWLEDGMENTS_EN.txt`, `BIBLIOGRAFIA_AUDIT.txt`, `IEEE_ACCESS_CHECKLIST.txt`,
  `CHECKLIST_AMANHA_ARTIGO.txt`: apoio editorial e checklist

## Requisitos
- Python 3.10+
- (Opcional) CUDA/driver NVIDIA para treinar na GPU

## Execução rápida (SNN single-target)
1) Criar e ativar ambiente virtual
```bash
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
```

2) Instalar dependências
```bash
pip install -r scripts/requirements.txt
```

3) Treinar, avaliar e gerar figuras
```bash
python scripts/train.py
python scripts/eval.py
python scripts/generate_figures.py
```

## Multimodal (multi-saida)
```bash
source scripts/.venv/bin/activate
python scripts_multimodal/train_multimodal.py
python scripts_multimodal/eval_multimodal.py
python scripts_multimodal/generate_figures_multimodal.py
```
Mais detalhes em `scripts_multimodal/README.md`.

## Grid search e cross-validation (opcionais)
```bash
python scripts/grid_search.py
python scripts/cross_validation.py
```
Saidas em `outputs/` (ex.: `grid_search.json`, `grid_best.pt`).

## Artigo (LaTeX)
Arquivos em `artigo/` (principal: `artigo/main.tex`).
O PDF atual esta em `artigo/main.pdf` e `artigo/paper-official.pdf`.

## Deploy e benchmarks
- ESP32: `embarcado/README_SNN_ESP32.md`
- Benchmark de energia/latencia: `energy_benchmark/README.md`
- App Flutter: `serpa_predict/README.md`

## Dados e artefatos
- Dados: `scripts/data/dados_serpa.csv`
- Figuras geradas: `figuras/` e `scripts/figuras/`
- Modelos/normalizacao: `outputs/`

## Rodar via Conda (opcional)
```bash
conda create -n serpa_lago python=3.10 -y
source /home/alfredo/miniconda3/etc/profile.d/conda.sh
conda activate serpa_lago
pip install -r scripts/requirements.txt
python scripts/train.py
python scripts/eval.py
python scripts/generate_figures.py
```
