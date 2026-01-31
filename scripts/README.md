SNN training scaffold (Lago do Serpa)

This folder contains a minimal, modular PyTorch SNN pipeline for regression on
`dados_serpa.csv`. It is designed to run on CPU by default and use CUDA if
available.

Quick start (Linux/macOS):

1) Create a virtual env
   python3 -m venv .venv
   source .venv/bin/activate

2) Install deps
   pip install -r requirements.txt

3) Train (default target is pH)
   python train.py

4) Evaluate
   python eval.py

5) Generate figures
   python generate_figures.py

6) Grid search (optional)
   python grid_search.py

7) Cross-validation (optional)
   python cross_validation.py

Notes:
- If CUDA is available, set `device` in config.py to "cuda".
- You can change the target variable in config.py (e.g., "OD", "TURB").
- The figure outputs (600 dpi) go to `scripts/figuras/`.
