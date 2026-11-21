.PHONY: pdf parker-pdf parker-equations nlte-pdf nlte-equations coupled-pdf coupled-equations cloudy-pdf cloudy-equations atom-pdf atom-equations clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build model.tex
	cp build/model.pdf model.pdf

atom-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build atom_convergence.tex
	cp build/atom_convergence.pdf atom_convergence.pdf

atom-equations:
	python -m mceg equations ../mceg/contracts/escape.atom_convergence.yaml --output atom_convergence_equations.tex

parker-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build parker_wind.tex
	cp build/parker_wind.pdf parker_wind.pdf

parker-equations:
	python -m mceg equations ../mceg/contracts/escape.parker_wind.yaml --output parker_wind_equations.tex

nlte-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build hydrogen_nlte.tex
	cp build/hydrogen_nlte.pdf hydrogen_nlte.pdf

# Run only in the separate authoring environment with MCEG installed.
nlte-equations:
	python -m mceg equations ../mceg/contracts/escape.hydrogen_nlte.yaml --output hydrogen_nlte_equations.tex

coupled-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build coupled_model.tex
	cp build/coupled_model.pdf coupled_model.pdf

coupled-equations:
	python -m mceg equations ../mceg/contracts/escape.coupled_hydrogen.yaml --output coupled_model_equations.tex

clean:
	rm -rf build

cloudy-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build cloudy_investigation.tex
	cp build/cloudy_investigation.pdf cloudy_investigation.pdf

cloudy-equations:
	python -m mceg equations ../mceg/contracts/escape.cloudy_investigation.yaml --output cloudy_investigation_equations.tex
