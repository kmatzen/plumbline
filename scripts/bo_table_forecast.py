"""Bayesian score prediction + active-learning, with PROTOCOL as a discrete variable.

Each reproduction cell is (model, dataset, protocol) → score. Scores are only
comparable WITHIN one protocol, so the model treats protocol as a third discrete
factor alongside model and dataset:

    log(score) = global + model_effect + dataset_effect + protocol_effect + noise

with Gaussian priors → exact Gaussian posterior → predicted score + uncertainty
for ANY (model, dataset, protocol), including protocol counterfactuals. Then:

  1. PREDICT — per protocol, fill the gaps in that protocol's cohort
     (its models × its datasets) with a calibrated score ± 1σ.
  2. ACQUIRE — rank (model, dataset, protocol) evals to run next by expected
     information gain across the whole table.

  python scripts/bo_table_forecast.py                 # per-protocol fill-in + next-evals
  python scripts/bo_table_forecast.py --json          # predictions for the site
"""

import argparse
import json
import re

import numpy as np

EXPLORE = "site/explore.html"

# Protocol = the eval RECIPE, read from the recorded recipe (which paper's eval
# TABLE the score is scored under, + dataset) — NOT the method. This reproduces the
# plumbline `protocol:` family per cell: the eval source `src` is the paper whose
# recipe was used (e.g. a DA-V2 baseline row in MoGe's Table 3 has src=MoGe →
# moge-eval, regardless that the method is DA-V2). Method never appears here, so
# protocol is an independent tensor axis. Mirrored by recipeOf() in the site.
def proto_of(model, ds, src, loc):
    s = re.sub(r"\s*\(.*?\)", "", src or "").strip()
    if s == "MoGe":
        return "moge-eval (affine)"                       # *_moge
    if s == "DUSt3R":
        return "dust3r-table2 (eigen+ratio-med)" if ds == "NYU" else "dust3r-lineage (median, no-crop)"
    if s == "MonST3R":
        return "dust3r-lineage (median, no-crop)"         # *_dust3r_lineage
    if s == "CUT3R":
        return "video (per-sequence)" if ds == "Bonn" else "dust3r-lineage (median, no-crop)"
    if s == "UniK3D":
        return "metric (no-align)"                        # *_metric
    if s == "Depth Pro":
        return "metric (no-align)" if ds in ("Booster", "ETH3D", "iBims-1", "Sun-RGBD") else "eigen-2014 (crop+median)"
    return "eigen-2014 (crop+median)"                     # nyu_eigen_2014 / kitti_eigen_garg


def load_cells():
    h = open(EXPLORE).read()
    out, seen = [], set()
    for kind in ("verified-cells", "companion-cells", "info-cells"):
        m = re.search(rf'id="{kind}">(.*?)</script>', h, re.S)
        for c in json.loads(m.group(1)):
            if c.get("task") != "depth" or c.get("metric") != "abs_rel":
                continue
            cite = c.get("cite") or {}
            proto = proto_of(c["model"], c["ds"], cite.get("src", ""), cite.get("loc", ""))
            k = (c["model"], c["ds"], proto)  # a model+dataset can be measured under 2 recipes
            if k in seen:
                continue
            seen.add(k)
            out.append({"model": c["model"], "ds": c["ds"], "y": float(c["obs"]), "proto": proto})
    return out


def fit(cells, prior_sd=0.7, noise_sd=0.12):
    models = sorted({c["model"] for c in cells})
    datasets = sorted({c["ds"] for c in cells})
    protos = sorted({c["proto"] for c in cells})
    mi = {m: i for i, m in enumerate(models)}
    di = {d: i for i, d in enumerate(datasets)}
    pi = {p: i for i, p in enumerate(protos)}
    nM, nD, nP = len(models), len(datasets), len(protos)
    P = 1 + nM + nD + nP

    def feat(m, d, p):
        x = np.zeros(P)
        x[0] = 1.0
        x[1 + mi[m]] = 1.0
        x[1 + nM + di[d]] = 1.0
        x[1 + nM + nD + pi[p]] = 1.0
        return x

    X = np.array([feat(c["model"], c["ds"], c["proto"]) for c in cells])
    y = np.log(np.array([c["y"] for c in cells]))
    tau = np.full(P, prior_sd**2)
    tau[0] = 10.0**2
    A = X.T @ X / noise_sd**2 + np.diag(1.0 / tau)
    Sigma = np.linalg.inv(A)
    mu = Sigma @ (X.T @ y) / noise_sd**2

    def predict(m, d, p):
        x = feat(m, d, p)
        return float(np.exp(x @ mu)), float(np.sqrt(x @ Sigma @ x + noise_sd**2))

    grid = [feat(m, d, p) for m in models for d in datasets for p in protos]

    def info_gain(m, d, p):
        Sx = Sigma @ feat(m, d, p)
        denom = noise_sd**2 + feat(m, d, p) @ Sx
        return float(sum((z @ Sx) ** 2 for z in grid) / denom)

    return models, datasets, protos, predict, info_gain


def proto_datasets(cells):
    """Per protocol: the datasets it covers (a protocol's table reports these)."""
    out = {}
    for c in cells:
        out.setdefault(c["proto"], set()).add(c["ds"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cells = load_cells()
    measured = {(c["model"], c["ds"], c["proto"]) for c in cells}
    models, datasets, protos, predict, info_gain = fit(cells)

    # FULL TENSOR: every (model, dataset, protocol) — measured or predicted. No
    # restriction to a protocol's "own" datasets; the protocol is a free axis.
    def gaps_for(p):
        return [(m, d) for m in models for d in datasets if (m, d, p) not in measured]

    if args.json:
        out = []
        for p in protos:
            for m, d in gaps_for(p):
                mean, sd = predict(m, d, p)
                out.append({
                    "model": m, "ds": d, "proto": p, "metric": "abs_rel",
                    "pred": round(mean, 4),
                    "lo": round(float(np.exp(np.log(mean) - sd)), 4),
                    "hi": round(float(np.exp(np.log(mean) + sd)), 4),
                })
        print(json.dumps(out, separators=(",", ":")))
        return

    cube = len(models) * len(datasets) * len(protos)
    print(f"{len(cells)} observed cells → full tensor {len(models)} models × "
          f"{len(datasets)} datasets × {len(protos)} protocols = {cube} cells "
          f"({len(cells)} observed, {cube - len(cells)} predicted)\n")
    print("PROTOCOLS (recipe, read per-cell from the recorded eval — never the method):")
    for p in protos:
        n = sum(1 for c in cells if c["proto"] == p)
        ms = sorted({c["model"] for c in cells if c["proto"] == p})
        print(f"  ◇ {p:32} {n:2} obs · methods: {', '.join(ms)}")

    print(f"\nACQUISITION — top {args.topk} (model, dataset, protocol) by info gain:")
    cand = [(m, d, p, predict(m, d, p)[0], info_gain(m, d, p))
            for p in protos for m, d in gaps_for(p)]
    for i, (m, d, p, mean, ig) in enumerate(sorted(cand, key=lambda r: -r[4])[: args.topk], 1):
        print(f"  {i:<3} {m:14} {d:9} ·{p:32} ~{mean:.4f}   infogain {ig:.2f}")


if __name__ == "__main__":
    main()
