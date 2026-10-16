# Banner LaTeX modular

Template de banner vertical criado a partir das normas e do modelo oficial do **II Encontro Científico e Tecnológico do HUGV / I Seminário Conectando Saberes**.

## Formato e conformidade

- Tamanho final: `90 cm x 120 cm`, orientação vertical, conforme as normas de submissão. A moldura do PPTX oficial foi adaptada a essa dimensão sem alterar sua identidade visual.
- Título: 72 pt; subtítulos: 48 pt; afiliações: 40 pt; corpo: 36 pt.
- A tipografia segue literalmente as instruções do slide-modelo. Para preservar a legibilidade, reduza o conteúdo antes de alterar esses tamanhos.
- As referências seguem o formato ABNT simplificado e devem ser revisadas conforme a NBR 6023:2018 antes da submissão.
- O cabeçalho e o rodapé institucional são preservados a partir do modelo oficial do evento.
- `assets/template/modelo_oficial.pdf` é a moldura exportada do PPTX fornecido e é usada diretamente como plano de fundo; não redesenhe o cabeçalho, a borda ou o rodapé.
- As imagens incluídas são cópias locais das figuras do estudo; substitua somente mantendo resolução adequada para impressão (a normativa recomenda 300 dpi).

## Estrutura

```text
banner_latex/
  main.tex                 arquivo principal
  config.tex               tamanho do pôster, cores e macros reutilizáveis
  sections/                conteúdo modular por seção
  assets/logos/            recursos institucionais locais usados na moldura
  assets/figures/          figuras usadas no banner
  data/                    conjunto original e resumos de benchmark
  referencias_evento/      cópias das normas e do PPTX fornecidos
```

## Compilação

Execute no diretório `banner_latex`:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

O arquivo resultante é `main.pdf`. A prévia de conferência é `preview.png`. Para uma impressão em escala correta, use a opção de impressão em `100%` ou `tamanho real`, sem ajuste automático à página.

## Versões cega e identificada

- `sem_blinded/main.tex`: versão identificada, com autores, afiliações, referência do conjunto de dados e agradecimentos.
- `blinded/main.tex`: versão para avaliação cega, sem autores, afiliações, agradecimentos e referência que identifica a autoria.

Compile a partir da pasta `banner_latex` para que os caminhos compartilhados de figuras sejam resolvidos:

```bash
latexmk -xelatex -outdir=sem_blinded sem_blinded/main.tex
latexmk -xelatex -outdir=blinded blinded/main.tex
```

## Personalização rápida

Edite apenas os arquivos em `sections/` para alterar título, autores, eixo temático, texto e referências. A composição institucional, as margens, as dimensões e as logos ficam centralizadas em `config.tex`.

Os arquivos de referência na pasta `referencias_evento/` foram preservados para conferência das regras do evento.
