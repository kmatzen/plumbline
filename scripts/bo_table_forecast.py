"""Bayesian completion of the 4-D reproduction tensor: method × dataset × protocol × metric.

Every observed cell is (method, dataset, protocol, metric) -> score. The four axes
are orthogonal; protocol is the eval RECIPE (read from the recorded eval source +
dataset, never the method), and metric is the 4th axis.

Metrics differ in scale and direction (AbsRel lower-better ~0.05; δ₁ higher-better
~0.95; mAA higher-better; ATE lower-better). We standardise each metric to a
direction-aware "goodness" g = sign·(log score − μ_metric)/σ_metric (higher g =
better, comparable across metrics), then fit

    g ~ method + dataset + protocol          (Gaussian priors, exact posterior)

so method-skill / dataset-difficulty / protocol-leniency are SHARED across metrics
— a method observed only in δ₁ (e.g. DA3) still gets an AbsRel prediction. Predict
g for any (method, dataset, protocol, metric), back-transform to that metric's scale.

  python scripts/bo_table_forecast.py            # tensor summary + acquisition
  python scripts/bo_table_forecast.py --json      # predictions for the site
"""

import argparse
import json
import re

import numpy as np

EXPLORE = "site/explore.html"
PRIMARY = ("abs_rel", "delta_1", "maa", "ate")
SIGN = {"delta_1": +1, "maa": +1, "abs_rel": -1, "ate": -1}  # +1 = higher is better
METRIC_TASK = {"abs_rel": "depth", "delta_1": "depth", "maa": "pose", "ate": "pose"}

# What each method can actually DO — from the adapters' `tasks` (mono_depth vs pose).
# We never predict a metric for a method that doesn't do that task.
CAPABILITY = {
    "DA-V2 Small": ["depth"], "DA-V2 Base": ["depth"], "DA-V2 Large": ["depth"],
    "MoGe ViT-L": ["depth"], "Metric3D-v2 L": ["depth"], "Metric3D-v2 G": ["depth"],
    "Marigold v1-1": ["depth"], "Depth Pro": ["depth"], "UniK3D Large": ["depth"],
    "DA3": ["depth", "pose"], "CUT3R": ["depth", "pose"], "DUSt3R": ["depth", "pose"],
    "MonST3R": ["depth", "pose"], "VGGT": ["depth", "pose"], "MASt3R": ["depth", "pose"],
    "DAGE": ["pose"],
}


def can_do(method, metric):
    return METRIC_TASK[metric] in CAPABILITY.get(method, ["depth", "pose"])

DEPTH_DROP = ("Booster", "ETH3D", "iBims-1", "Sun-RGBD")


def proto_of(model, ds, metric, src, loc):
    """Protocol = eval recipe, from the eval SOURCE + dataset — never the method."""
    s = re.sub(r"\s*\(.*?\)", "", src or "").strip()
    if metric in ("abs_rel", "delta_1"):
        if ds in ("GSO", "DDAD"):
            return "moge-eval (affine)"  # these datasets are only ever the MoGe affine eval
        if s == "MoGe":
            return "moge-eval (affine)"
        if s == "DUSt3R":
            return "dust3r-table2 (eigen+ratio-med)" if ds == "NYU" else "dust3r-lineage (median, no-crop)"
        if s == "MonST3R":
            return "dust3r-lineage (median, no-crop)"
        if s == "CUT3R":
            return "video (per-sequence)" if ds == "Bonn" else "dust3r-lineage (median, no-crop)"
        if s == "UniK3D":
            return "metric (no-align)"
        if s == "Depth Pro":
            return "metric (no-align)" if ds in DEPTH_DROP else "eigen-2014 (crop+median)"
        return "eigen-2014 (crop+median)"
    if metric == "maa":
        return "RE10K wide-baseline" if ds == "RE10K" else "PoseDiff (10-frame)"
    if metric == "ate":
        return "trajectory (Sim3 ATE)"
    return "other"


def load_cells():
    h = open(EXPLORE).read()
    out, seen = [], set()
    for kind in ("verified-cells", "companion-cells", "info-cells"):
        m = re.search(rf'id="{kind}">(.*?)</script>', h, re.S)
        for c in json.loads(m.group(1)):
            if c.get("metric") not in PRIMARY or c.get("obs") in (None, ""):
                continue
            # read the baked single-source `proto`; proto_of() is only the baker's
            # logic now (scripts/site_protocols.py), kept here as a fallback.
            cite = c.get("cite") or {}
            proto = c.get("proto") or proto_of(c["model"], c["ds"], c["metric"], cite.get("src", ""), cite.get("loc", ""))
            k = (c["model"], c["ds"], c["metric"], proto)
            if k in seen:
                continue
            seen.add(k)
            out.append({"model": c["model"], "ds": c["ds"], "metric": c["metric"],
                        "proto": proto, "y": float(c["obs"])})
    return out


def fit(cells, prior_sd=0.8, noise_sd=0.35):
    methods = sorted({c["model"] for c in cells})
    datasets = sorted({c["ds"] for c in cells})
    protos = sorted({c["proto"] for c in cells})
    mi = {m: i for i, m in enumerate(methods)}
    di = {d: i for i, d in enumerate(datasets)}
    pi = {p: i for i, p in enumerate(protos)}
    nM, nD, nP = len(methods), len(datasets), len(protos)
    P = 1 + nM + nD + nP

    # per-metric standardisation of log-score → direction-aware goodness g
    stats = {}
    for k in PRIMARY:
        ys = np.log([c["y"] for c in cells if c["metric"] == k])
        if len(ys):
            stats[k] = (float(ys.mean()), float(max(ys.std(), 0.15)))

    def good(c):
        mu, sd = stats[c["metric"]]
        return SIGN[c["metric"]] * (np.log(c["y"]) - mu) / sd

    def feat(m, d, p):
        x = np.zeros(P)
        x[0] = 1.0
        x[1 + mi[m]] = 1.0
        x[1 + nM + di[d]] = 1.0
        x[1 + nM + nD + pi[p]] = 1.0
        return x

    X = np.array([feat(c["model"], c["ds"], c["proto"]) for c in cells])
    g = np.array([good(c) for c in cells])
    tau = np.full(P, prior_sd**2)
    tau[0] = 10.0**2
    Sigma = np.linalg.inv(X.T @ X / noise_sd**2 + np.diag(1.0 / tau))
    mu_w = Sigma @ (X.T @ g) / noise_sd**2

    def predict(m, d, p, metric):
        x = feat(m, d, p)
        gm = float(x @ mu_w)
        gs = float(np.sqrt(x @ Sigma @ x + noise_sd**2))
        mu, sd = stats[metric]
        lm = mu + SIGN[metric] * sd * gm          # back to log-score
        ls = sd * gs                               # log-score std
        return float(np.exp(lm)), float(np.exp(lm - ls)), float(np.exp(lm + ls))

    grid = [feat(m, d, p) for m in methods for d in datasets for p in protos]

    def info_gain(m, d, p):
        Sx = Sigma @ feat(m, d, p)
        return float(sum((z @ Sx) ** 2 for z in grid) / (noise_sd**2 + feat(m, d, p) @ Sx))

    model = {  # the fitted model, small enough to embed + evaluate in JS
        "methods": methods, "datasets": datasets, "protos": protos,
        "nM": nM, "nD": nD, "noise2": noise_sd**2,
        "mu": [round(float(v), 5) for v in mu_w],
        "Sigma": [[round(float(v), 5) for v in row] for row in Sigma],
        "stats": {k: [round(stats[k][0], 5), round(stats[k][1], 5), SIGN[k]] for k in stats},
        "metric_task": METRIC_TASK,
        "cap": {m: CAPABILITY.get(m, ["depth", "pose"]) for m in methods},
    }
    return methods, datasets, protos, predict, info_gain, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", action="store_true", help="emit the fitted model for in-browser prediction")
    ap.add_argument("--loo", action="store_true", help="leave-one-out cross-validation")
    args = ap.parse_args()

    cells = load_cells()
    measured = {(c["model"], c["ds"], c["metric"], c["proto"]) for c in cells}
    methods, datasets, protos, predict, info_gain, model = fit(cells)

    if args.model:
        print(json.dumps(model, separators=(",", ":")))
        return

    if args.loo:
        import statistics
        errs = {k: [] for k in PRIMARY}
        hits = tot = cold = 0
        for i in range(len(cells)):
            c = cells[i]
            _, _, _, pr, _, _ = fit(cells[:i] + cells[i + 1:])
            try:
                mean, lo, hi = pr(c["model"], c["ds"], c["proto"], c["metric"])
            except KeyError:  # held-out cell was the sole observation of some level
                cold += 1
                continue
            errs[c["metric"]].append(abs(mean - c["y"]) / c["y"])
            tot += 1
            hits += lo <= c["y"] <= hi
        print(f"LEAVE-ONE-OUT CV  ({len(cells)} cells)\n")
        for k in PRIMARY:
            if errs[k]:
                print(f"  {k:8} median rel-err {statistics.median(errs[k]) * 100:5.1f}%  "
                      f"(max {max(errs[k]) * 100:.0f}%)  n={len(errs[k])}")
        print(f"\n  calibration: {hits}/{tot} held-out scores fell within ±1σ "
              f"({100*hits/tot:.0f}%; well-calibrated ≈ 68%)")
        print(f"  cold-start: {cold}/{len(cells)} cells are the SOLE observation of some "
              f"method/dataset/protocol → unpredictable when held out")
        return

    # 4-D cube MASKED by capability: skip metrics a method can't do (no pose for a
    # depth-only model, no depth for a pose-only model).
    def gaps():
        for k in PRIMARY:
            for p in protos:
                for d in datasets:
                    for m in methods:
                        if can_do(m, k) and (m, d, k, p) not in measured:
                            yield (m, d, p, k)

    if args.json:
        out = []
        for m, d, p, k in gaps():
            mean, lo, hi = predict(m, d, p, k)
            out.append({"model": m, "ds": d, "proto": p, "metric": k,
                        "pred": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
        print(json.dumps(out, separators=(",", ":")))
        return

    obs = len(cells)
    cube = len(methods) * len(datasets) * len(protos) * len(PRIMARY)
    print(f"FULL 4-D cube {len(methods)} methods × {len(datasets)} datasets × "
          f"{len(protos)} protocols × {len(PRIMARY)} metrics = {cube} cells "
          f"({obs} observed, {cube - obs} predicted)\n")
    print("Every (metric, protocol) view is the SAME complete methods × datasets grid.")
    print(f"protocols: {', '.join(protos)}")

    print(f"\nACQUISITION — top {args.topk} (method, dataset, protocol) by info gain:")
    seen = {(c["model"], c["ds"], c["proto"]) for c in cells}
    cand = sorted(((m, d, p, info_gain(m, d, p)) for m in methods for d in datasets for p in protos
                   if (m, d, p) not in seen), key=lambda r: -r[3])[: args.topk]
    for i, (m, d, p, ig) in enumerate(cand, 1):
        print(f"  {i:<3} {m:13} {d:8} ·{p:28} ig {ig:.2f}")


if __name__ == "__main__":
    main()
