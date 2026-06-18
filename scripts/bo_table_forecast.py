"""Bayesian completion + active-learning for the reproduction table (prototype).

The site is a reproduction harness, not a leaderboard: cells use different
protocols/targets, so absolute scores aren't cross-comparable. But WITHIN a
task-consistent slice (here: mono-depth AbsRel) we can model the matrix and:

  1. PREDICT every empty (model, dataset) cell with calibrated uncertainty
     ("fill in the rest of the table").
  2. RANK the empty cells by an acquisition function ("what to run next").

Model: Bayesian additive latent-factor (probabilistic matrix factorisation,
rank-0 + closed-form posterior). log(AbsRel) ~ global + model_effect +
dataset_effect, Gaussian priors, exact Gaussian posterior → predictive mean +
variance for ANY cell, including unmeasured pairings (via the learned per-model
"quality" and per-dataset "difficulty"). Honest: variance is large where a model
or dataset is thinly measured — which is exactly what the acquisition targets.

  python scripts/bo_table_forecast.py            # fill-in + next-eval ranking
  python scripts/bo_table_forecast.py --topk 12
"""

import argparse
import json
import re

import numpy as np

EXPLORE = "site/explore.html"


def load_depth_absrel_cells():
    h = open(EXPLORE).read()
    cells = []
    for kind in ("verified-cells", "companion-cells", "info-cells"):
        m = re.search(rf'id="{kind}">(.*?)</script>', h, re.S)
        for c in json.loads(m.group(1)):
            if c.get("task") == "depth" and c.get("metric") == "abs_rel":
                cells.append({"model": c["model"], "ds": c["ds"], "y": float(c["obs"])})
    # de-dup (model, ds): keep first (verified beats info)
    seen, out = set(), []
    for c in cells:
        k = (c["model"], c["ds"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def fit_predict(cells, prior_sd=1.0, noise_sd=0.15):
    """Bayesian linear regression on one-hot [1, model, dataset], target log(y)."""
    models = sorted({c["model"] for c in cells})
    datasets = sorted({c["ds"] for c in cells})
    mi = {m: i for i, m in enumerate(models)}
    di = {d: i for i, d in enumerate(datasets)}
    nM, nD = len(models), len(datasets)
    P = 1 + nM + nD  # intercept + model effects + dataset effects

    def feat(m, d):
        x = np.zeros(P)
        x[0] = 1.0
        x[1 + mi[m]] = 1.0
        x[1 + nM + di[d]] = 1.0
        return x

    X = np.array([feat(c["model"], c["ds"]) for c in cells])
    y = np.log(np.array([c["y"] for c in cells]))

    # Gaussian prior N(0, prior_sd^2 I) (looser on intercept), noise noise_sd^2.
    tau = np.full(P, prior_sd**2)
    tau[0] = 10.0**2
    A = X.T @ X / noise_sd**2 + np.diag(1.0 / tau)
    Sigma = np.linalg.inv(A)
    mu = Sigma @ (X.T @ y) / noise_sd**2  # posterior mean of weights

    def predict(m, d):
        x = feat(m, d)
        mean = float(x @ mu)
        var = float(x @ Sigma @ x) + noise_sd**2
        return mean, np.sqrt(var)

    # Expected info gain of measuring cell x*: the rank-1 (Sherman-Morrison)
    # reduction in posterior covariance, summed over EVERY other empty cell —
    # i.e. how much does this one eval shrink uncertainty across the whole table.
    grid = [feat(m, d) for m in models for d in datasets]

    def info_gain(m, d):
        xs = feat(m, d)
        Sx = Sigma @ xs
        denom = noise_sd**2 + xs @ Sx  # scalar
        # reduction for any cell z: (z·Sx)^2 / denom
        return float(sum((z @ Sx) ** 2 for z in grid) / denom)

    return models, datasets, predict, info_gain


def emit_json(cells, measured, models, datasets, predict):
    """Predictions for empty cells, for embedding in the site as ghosted fill-in."""
    out = []
    for m in models:
        for d in datasets:
            if (m, d) in measured:
                continue
            lm, ls = predict(m, d)
            out.append({
                "model": m, "ds": d, "metric": "abs_rel",
                "pred": round(float(np.exp(lm)), 4),
                "lo": round(float(np.exp(lm - ls)), 4),
                "hi": round(float(np.exp(lm + ls)), 4),
            })
    print(json.dumps(out, indent=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="emit predicted empty cells as JSON (for the site)")
    args = ap.parse_args()

    cells = load_depth_absrel_cells()
    measured = {(c["model"], c["ds"]) for c in cells}
    models, datasets, predict, info_gain = fit_predict(cells)

    if args.json:
        emit_json(cells, measured, models, datasets, predict)
        return

    print(f"mono-depth AbsRel slice: {len(cells)} measured cells, "
          f"{len(models)} models × {len(datasets)} datasets "
          f"= {len(models)*len(datasets)} grid ({len(measured)} filled, "
          f"{len(models)*len(datasets)-len(measured)} empty)\n")

    # --- completed table: predicted AbsRel for empty cells ---
    print("PREDICTED fill-in (empty cells, AbsRel ± 1σ, geometric):")
    rows = []
    for m in models:
        for d in datasets:
            if (m, d) in measured:
                continue
            lm, ls = predict(m, d)
            rows.append((m, d, np.exp(lm), np.exp(lm + ls) - np.exp(lm), ls))
    # show a few representative predictions
    for m, d, mean, sd, ls in sorted(rows, key=lambda r: r[2])[:8]:
        print(f"  {m:14} {d:10} ~{mean:.3f}  (+{sd:.3f}/-{np.exp(np.log(mean)-ls)-mean:+.3f})")

    # --- acquisition: what to run next (max expected info gain over the table) ---
    print(f"\nACQUISITION — top {args.topk} evals to run next (max expected info "
          "gain = most uncertainty removed across the WHOLE table per run):")
    cand = []
    for m in models:
        for d in datasets:
            if (m, d) in measured:
                continue
            lm, ls = predict(m, d)
            cand.append((m, d, float(np.exp(lm)), ls, info_gain(m, d)))
    ranked = sorted(cand, key=lambda r: r[4], reverse=True)[: args.topk]
    gmax = ranked[0][4]
    print(f"  {'rank':4} {'model':14} {'dataset':10} {'pred AbsRel':>11} {'σ':>5} {'infogain':>9}  why")
    for i, (m, d, mean, ls, ig) in enumerate(ranked, 1):
        nm = sum(1 for c in cells if c["model"] == m)
        nd = sum(1 for c in cells if c["ds"] == d)
        why = f"model {nm}×, dataset {nd}× → teaches {ig/gmax:.0%} of best"
        print(f"  {i:<4} {m:14} {d:10} {mean:11.3f} {ls:5.2f} {ig:9.2f}  {why}")


if __name__ == "__main__":
    main()
