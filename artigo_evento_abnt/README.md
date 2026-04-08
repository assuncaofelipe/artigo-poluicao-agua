# Artigo em formato do evento

Esta pasta cont\'em uma vers\~ao modularizada em LaTeX adaptada para a normativa do evento:

- papel A4;
- margens 3 cm (superior/esquerda) e 2 cm (inferior/direita);
- fonte em fam\'ilia Times;
- espa\c{c}amento 1,5;
- estrutura compat\'ivel com artigo cient\'ifico em ABNT;
- cita\c{c}\~oes autor-data com `abntex2cite`.

## Estrutura

- `main.tex`: arquivo principal.
- `sections/`: se\c{c}\~oes do artigo.
- `referencias.bib`: base bibliogr\'afica usada nesta vers\~ao.

## Compila\c{c}\~ao

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Ajustes pendentes antes da submiss\~ao

- substituir autores, institui\c{c}\~oes e e-mails;
- confirmar o eixo tem\'atico correto no sistema do evento;
- revisar o total de p\'aginas da vers\~ao final compilada;
- conferir se as figuras externas devem ser copiadas para dentro desta pasta.
