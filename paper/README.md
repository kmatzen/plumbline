# Plumbline paper

The manuscript reports the verified reproduction ledger in `site/explore.html`.
It does not treat the verified cells as a random sample or use them to estimate a
reproduction success rate.

Build from the repository root:

```sh
uv run python paper/make_figures.py
uv run python paper/run_experiments.py
uv run python paper/check_claims.py
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The release PDF is built separately into `output/pdf/` so generated files do not
need to be committed.
