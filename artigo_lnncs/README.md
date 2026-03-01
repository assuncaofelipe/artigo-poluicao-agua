# LNCS Modular Paper

This folder contains a modular LNCS version of the paper.

## Structure
- `main.tex`
- `sections/introduction.tex`
- `sections/related_work.tex`
- `sections/dataset_description.tex`
- `sections/methodology.tex`
- `sections/energy_measurement_setup.tex`
- `sections/experimental_results.tex`
- `sections/discussion.tex`
- `sections/conclusion.tex`
- `references.bib`
- `llncs.cls`
- `splncs04.bst`

## Compile
```bash
cd artigo_lnncs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
