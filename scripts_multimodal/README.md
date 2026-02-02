Pipeline multimodal (multi-saida) - Lago do Serpa

Objetivo:
- Treinar um modelo multi-saida para prever pH, OD e TURB
- Gerar graficos de desempenho
- Gerar mapas e superficies interpoladas

Importante:
- Para "manchas de poluicao" ou "nivel de poluicao", e necessario definir
  uma regra (ex.: limiares ou indice composto). Sem essa definicao, o modelo
  nao tem rótulo de poluicao.

Passos rapidos:
1) Ativar env
   source ../scripts/.venv/bin/activate

2) Instalar deps (usa o mesmo requirements do scripts/)
   pip install -r ../scripts/requirements.txt

3) Treinar
   python train_multimodal.py

4) Avaliar e gerar graficos
   python eval_multimodal.py
   # mapas de poluicao sao gerados automaticamente

5) Grid search (opcional)
   python grid_search_multimodal.py

6) Metricas adicionais (MAE/MSE/R2 e tolerancia)
   python metrics.py

7) Gerar todas as figuras do multimodal
   python generate_figures_multimodal.py

8) Exportar modelo para TFLite (para o app Flutter)
   # dependencias: onnx, onnx2tf, tensorflow
   python export_tflite.py
   # OBS: a exportacao usa codificacao deterministica (sem aleatoriedade)

Arquivos principais:
- config.py
- data.py
- model.py
- train_multimodal.py
- eval_multimodal.py
- figures.py

Mapas e manchas:
- Para estimar "nivel de poluicao", defina um indice ou regra (ex.: combinacao
  de pH + OD + TURB). Sem isso, o sistema apenas preve variaveis fisico-quimicas.
- Este pipeline ja calcula um indice composto (configuravel em config.py) e
  gera mapas de pontos e IDW do indice.

Saidas:
- outputs/ (modelos e logs)
- figuras/ (imagens 600 dpi)
