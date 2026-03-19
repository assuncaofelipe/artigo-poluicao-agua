# Artigo SBA (Modular)

Versão do artigo migrada para o template `SBAconf-LATEX-2` (classe `ifacconf`) em estrutura modular.

## Estrutura
- `main.tex`
- `sections/*.tex`
- `figuras/*`
- `references.bib`
- `ifacconf.cls`
- `ifacconf.bst`

## Compilação
```bash
cd artigo_sba_modular
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Observação de paginação
- Limite informado: mínimo 4 e máximo 8 páginas.
- Taxa adicional informada: R$ 200,00 por página após a 6ª.
- Situação atual desta versão: **7 páginas**.
