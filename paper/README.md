# Plumbline paper

This is an incomplete research draft about camera evaluation. The reproduction
ledger is supporting infrastructure. Real tracking comparisons and independent
judgments are still required before submission.

The original 92.9% versus 79.6% comparison is withdrawn: the cross-family defect
order was an arbitrary design choice, not an independently established target.
`controlled_evaluator.json` and `method_validation.png` are retained historical
exploratory outputs and must not support paper claims. The replacement sensitivity
experiment compares amplitudes only within each family and includes standard RPE.
The synthetic search experiment tests the optimizer, not tracking improvements.

## Real-data pilot

`inventory_pose_cache.py` inventories camera predictions on Anima without loading
the large depth and point-cloud arrays. `export_realestate_poses.py` runs in the
original `/home/kmatzen/git/geobench/.venv` environment and reconstructs seed-0,
ten-frame samples from `/data3/kmatzen/data/re10k_sub50`. It matches the old input
fingerprints and exports poses for native input and Gaussian blur sigma 1.5.
All 300 predictions matched (50 clips, three methods, two conditions).

`analyze_realestate.py` recomputes metrics locally. The recovered AUC aggregates
match the original blur-control records. ATE and AUC disagree on 27 of 149
decisive native-input method pairs, with one additional tied pair. This measures
disagreement, not accuracy against a quality target.

The legacy cache fingerprint covers image shape, intrinsics, and only the first
1 KB of image bytes. It cannot prove historical frame identity. Exported
timestamps are reconstructed; they were not stored at inference time. The
export includes the original loader's source hash for audit. Treat this as a
pilot until fresh runs store complete input hashes and ordered timestamps.

Run the local analysis with:

```sh
uv run python paper/analyze_realestate.py
```

The blur level was selected during earlier development on these clips. Bootstrap
intervals describe paired variation within this selected subset, not held-out
generalization or correction for the earlier parameter search. Ten sparse views
are suitable for geometric comparisons but insufficient to validate temporal
jitter judgments on dense video. The camera-evaluation contribution still needs
dense outputs, independent outcomes or reviewer labels, and held-out comparison
against ATE, RPE, pose AUC, and a combined metric baseline.

Build from the repository root:

```sh
uv run python paper/make_figures.py
uv run python paper/run_experiments.py
uv run python paper/run_sensitivity.py
uv run python paper/check_claims.py
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The release PDF is built separately into `output/pdf/` so generated files do not
need to be committed.
