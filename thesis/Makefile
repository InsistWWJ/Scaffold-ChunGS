# Scaffold-ChunGS Thesis Makefile
#
# Prerequisites:
#   - TeX Live (with xelatex, biber, ctex, biblatex-gb7714-2015)
#   - latexmk
#
# Usage:
#   make          - Build the thesis (PDF)
#   make clean    - Remove auxiliary files
#   make watch    - Continuous build (requires latexmk -pvc)

MAIN = main
PDF  = $(MAIN).pdf

.PHONY: all clean watch

all: $(PDF)

$(PDF): $(MAIN).tex chapters/*.tex thesis.bib
	latexmk -xelatex -bibtex -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.bcf *.run.xml *.synctex.gz

watch:
	latexmk -pvc -xelatex -bibtex $(MAIN).tex
