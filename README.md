# Projeto Lago do Serpa — Execução rápida

Este repositório contém o artigo e os scripts para treinar uma SNN e gerar
figuras (600 dpi) a partir do `dados_serpa.csv`.

## Requisitos
- Python 3.10+
- (Opcional) CUDA/driver NVIDIA se quiser treinar na GPU

## Passo a passo

1) Criar e ativar ambiente virtual
```bash
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
```

2) Instalar dependências
```bash
pip install -r scripts/requirements.txt
```

3) Treinar a SNN
```bash
python scripts/train.py
```

4) Avaliar e gerar scatter real vs predito
```bash
python scripts/eval.py
```

5) Gerar figuras descritivas (600 dpi)
```bash
python scripts/generate_figures.py
```

## Grid search (opcional)
Para buscar hiperparâmetros:
```bash
python scripts/grid_search.py
```
Saídas em `outputs/`:
- `grid_search.json` (todas as combinações)
- `grid_best.pt` (melhor modelo)

## Cross-validation (opcional)
```bash
python scripts/cross_validation.py
```
Saída: `outputs/cross_validation.json`

## Saídas
- Modelos e logs: `scripts/outputs/`
- Figuras (PNG 600 dpi): `scripts/figuras/`

## Configuração
Edite `scripts/config.py` para:
- trocar o alvo (`target_col`, ex.: `pH`, `OD`, `TURB`)
- ajustar hiperparâmetros
- habilitar CUDA (`device="cuda"`)
