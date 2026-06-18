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

# Protocol identity = (citing paper, table). Two cells share a protocol iff they
# were scored under the same paper's same eval table — the recipe that makes scores
# comparable. (MoGe Table 3 unifies MoGe + DA-V2 baselines across 8 datasets; DUSt3R
# Table 2 and MonST3R Table 3 stay separate — different crops.) Mirrored by the site.
def proto_of(model, ds, src, loc):
    src = re.sub(r"\s*\(.*?\)", "", src or "").strip()  # drop parentheticals
    src = {"Depth Anything V2": "DA-V2", "Metric3D v2": "Metric3D", "Depth Anything 3": "DA3"}.get(src, src)
    m = re.search(r"Table [IVXLC]+\b|Table \d+\w*", loc or "")
    return f"{src} {m.group(0) if m else 'Table ?'}"


def load_cells():
    h = open(EXPLORE).read()
    out, seen = [], set()
    for kind in ("verified-cells", "companion-cells", "info-cells"):
        m = re.search(rf'id="{kind}">(.*?)</script>', h, re.S)
        for c in json.loads(m.group(1)):
            if c.get("task") != "depth" or c.get("metric") != "abs_rel":
                continue
            k = (c["model"], c["ds"])
            if k in seen:
                continue
            seen.add(k)
            cite = c.get("cite") or {}
            out.append({
                "model": c["model"], "ds": c["ds"], "y": float(c["obs"]),
                "proto": proto_of(c["model"], c["ds"], cite.get("src", ""), cite.get("loc", "")),
            })
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
    measured = {(c["model"], c["ds"], c["proto"]) for c in cells}  # measured is PER protocol
    models, datasets, protos, predict, info_gain = fit(cells)
    pds = proto_datasets(cells)

    # Per protocol, fill its table for ALL models on its datasets (the leaderboard);
    # cells measured under a *different* protocol are counterfactual predictions here.
    def gaps_for(p):
        return [(m, d) for m in models for d in sorted(pds[p]) if (m, d, p) not in measured]

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

    print(f"{len(cells)} cells · {len(models)} models × {len(datasets)} datasets × "
          f"{len(protos)} protocols\n")
    print("PER-PROTOCOL leaderboard fill-in (one protocol per table → scores comparable):")
    for p in sorted(protos, key=lambda p: -len(pds[p])):
        ds = sorted(pds[p])
        run = sum(1 for m in models for d in ds if (m, d, p) in measured)
        gaps = gaps_for(p)
        print(f"\n  ◇ {p}  ({len(models)} models × {len(ds)} datasets · {run} run, {len(gaps)} predicted)")
        for m, d in gaps[:3]:
            mean, sd = predict(m, d, p)
            print(f"      {m:14} {d:9} ~{mean:.4f}  (1σ {np.exp(np.log(mean)-sd):.4f}–{np.exp(np.log(mean)+sd):.4f})")
        if len(gaps) > 3:
            print(f"      … +{len(gaps)-3} more")

    print(f"\nACQUISITION — top {args.topk} (model, dataset, protocol) evals by info gain:")
    cand = [(m, d, p, *predict(m, d, p)[:1], info_gain(m, d, p))
            for p in protos for m, d in gaps_for(p)]
    for i, (m, d, p, mean, ig) in enumerate(sorted(cand, key=lambda r: -r[4])[: args.topk], 1):
        print(f"  {i:<3} {m:14} {d:9} ·{p:18} ~{mean:.4f}   infogain {ig:.2f}")


if __name__ == "__main__":
    main()
