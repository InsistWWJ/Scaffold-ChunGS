# Scaffold-ChunGS Thesis Makefile
# 中国地质大学（武汉）学士学位论文
#
# Prerequisites:
#   - TeX Live 2023+ (xelatex, biber, ctex, biblatex-gb7714-2005)
#   - latexmk (optional, for watch mode)
#
# Usage:
#   make          - Build thesis PDF (full: xelatex -> biber -> xelatex -> xelatex)
#   make clean    - Remove auxiliary files
#   make watch    - Continuous build (requires latexmk -pvc)

MAIN = main
PDF  = $(MAIN).pdf

.PHONY: all clean watch

all: $(PDF)

$(PDF): $(MAIN).tex chapters/*.tex thesis.bib
	xelatex -synctex=1 -interaction=nonstopmode $(MAIN).tex
	biber $(MAIN)
	xelatex -synctex=1 -interaction=nonstopmode $(MAIN).tex
	xelatex -synctex=1 -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.bcf *.run.xml *.synctex.gz

watch:
	latexmk -pvc -xelatex -bibtex=biber $(MAIN).tex
