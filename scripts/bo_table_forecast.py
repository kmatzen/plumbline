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
import statistics
from collections import Counter

import numpy as np
import yaml

# display dataset -> gpu_queue.yaml dataset key (for the cost lookup)
DS_QUEUE = {"NYU": "nyuv2", "KITTI": "kitti", "DIODE": "diode", "ETH3D": "eth3d", "GSO": "gso",
            "iBims-1": "ibims1", "DDAD": "ddad", "Sintel": "sintel", "Bonn": "bonn",
            "Sun-RGBD": "sun_rgbd", "Booster": "booster", "CO3Dv2": "co3dv2",
            "RE10K": "realestate10k", "TUM-Dyn": "tum-dynamics"}
SLOW = {"Marigold v1-1"}  # diffusion (multi-step) → ~3× wall

EXPLORE = "site/explore.html"
PRIMARY = ("abs_rel", "delta_1", "maa", "ate")
SIGN = {"delta_1": +1, "maa": +1, "abs_rel": -1, "ate": -1}  # +1 = higher is better
METRIC_TASK = {"abs_rel": "depth", "delta_1": "depth", "maa": "pose", "ate": "pose"}
# valid output range per metric — predictions (and their bands) clamp here. δ₁ and
# mAA are accuracy fractions ≤ 1; abs_rel/ate are unbounded-positive (exp ⇒ ≥0).
RANGE = {"delta_1": (0.0, 1.0), "maa": (0.0, 1.0)}

# What each method can actually DO — from the adapters' `tasks` (mono_depth vs pose).
# We never predict a metric for a method that doesn't do that task.
CAPABILITY = {
    "DA-V2 Small": ["depth"], "DA-V2 Base": ["depth"], "DA-V2 Large": ["depth"],
    "MoGe ViT-L": ["depth"], "Metric3D-v2 L": ["depth"], "Metric3D-v2 G": ["depth"],
    "Marigold v1-1": ["depth"], "Depth Pro": ["depth"], "UniK3D Large": ["depth"],
    "DA3": ["depth", "pose"], "CUT3R": ["depth", "pose"], "DUSt3R": ["depth", "pose"],
    "MonST3R": ["depth", "pose"], "VGGT": ["depth", "pose"], "MASt3R": ["depth", "pose"],
    "StreamVGGT": ["depth", "pose"], "Pi3": ["depth", "pose"], "DAGE": ["pose"],
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


def fit(cells, prior_sd=0.8, noise_sd=0.35, calib=1.0):
    methods = sorted({c["model"] for c in cells})
    datasets = sorted({c["ds"] for c in cells})
    protos = sorted({c["proto"] for c in cells})
    # skill is PER TASK: (method, task) is the unit, so a method's pose results do
    # NOT inform its depth predictions. An unobserved (method, task) — e.g. MASt3R
    # depth — reverts to the prior (≈ average) with wide uncertainty instead of
    # inheriting the method's skill from the other task.
    mtasks = sorted({(m, t) for m in methods for t in CAPABILITY.get(m, ["depth", "pose"])})
    mti = {mt: i for i, mt in enumerate(mtasks)}
    di = {d: i for i, d in enumerate(datasets)}
    pi = {p: i for i, p in enumerate(protos)}
    nMT, nD, nP = len(mtasks), len(datasets), len(protos)
    P = 1 + nMT + nD + nP

    # per-metric standardisation of log-score → direction-aware goodness g
    stats = {}
    for k in PRIMARY:
        ys = np.log([c["y"] for c in cells if c["metric"] == k])
        if len(ys):
            stats[k] = (float(ys.mean()), float(max(ys.std(), 0.15)))

    def good(c):
        mu, sd = stats[c["metric"]]
        return SIGN[c["metric"]] * (np.log(c["y"]) - mu) / sd

    def feat(m, d, p, task):
        x = np.zeros(P)
        x[0] = 1.0
        x[1 + mti[(m, task)]] = 1.0
        x[1 + nMT + di[d]] = 1.0
        x[1 + nMT + nD + pi[p]] = 1.0
        return x

    X = np.array([feat(c["model"], c["ds"], c["proto"], METRIC_TASK[c["metric"]]) for c in cells])
    g = np.array([good(c) for c in cells])
    tau = np.full(P, prior_sd**2)
    tau[0] = 10.0**2
    Sigma = np.linalg.inv(X.T @ X / noise_sd**2 + np.diag(1.0 / tau))
    mu_w = Sigma @ (X.T @ g) / noise_sd**2

    def predict(m, d, p, metric):
        task = METRIC_TASK[metric]
        if (m, task) not in mti:
            return None
        x = feat(m, d, p, task)
        gm = float(x @ mu_w)
        gs = float(np.sqrt(x @ Sigma @ x + noise_sd**2))
        mu, sd = stats[metric]
        lm = mu + SIGN[metric] * sd * gm           # back to log-score
        ls = sd * gs * calib                       # log-score std, LOO-calibrated
        lo_b, hi_b = RANGE.get(metric, (0.0, float("inf")))
        clip = lambda v: min(max(v, lo_b), hi_b)   # never predict outside the metric's range
        return clip(float(np.exp(lm))), clip(float(np.exp(lm - ls))), clip(float(np.exp(lm + ls)))

    grid = [feat(m, d, p, t) for (m, t) in mtasks for d in datasets for p in protos]

    def info_gain(m, d, p, task):
        x = feat(m, d, p, task)
        Sx = Sigma @ x
        return float(sum((z @ Sx) ** 2 for z in grid) / (noise_sd**2 + x @ Sx))

    model = {  # the fitted model, small enough to embed + evaluate in JS
        "methods": methods, "datasets": datasets, "protos": protos,
        "method_tasks": [f"{m}|{t}" for (m, t) in mtasks],
        "nMT": nMT, "nD": nD, "noise2": noise_sd**2,
        "mu": [round(float(v), 5) for v in mu_w],
        "Sigma": [[round(float(v), 5) for v in row] for row in Sigma],
        "stats": {k: [round(stats[k][0], 5), round(stats[k][1], 5), SIGN[k]] for k in stats},
        "metric_task": METRIC_TASK,
        "range": {k: list(v) for k, v in RANGE.items()},   # clamp predictions to valid range
        "cap": {m: CAPABILITY.get(m, ["depth", "pose"]) for m in methods},
        "calib": round(calib, 4),
        # which metrics each dataset actually scores (it has GT for) — the site masks
        # the cube to these, so no AbsRel-on-CO3Dv2 / mAA-on-NYU nonsense is shown.
        "supports": {d: sorted({c["metric"] for c in cells if c["ds"] == d}) for d in datasets},
    }
    return methods, datasets, protos, predict, info_gain, model


def calibrate(cells):
    """LOO factor κ s.t. the inflated ±1σ band covers ~68% of held-out scores."""
    z = []
    for i in range(len(cells)):
        c = cells[i]
        try:
            _, _, _, pr, _, _ = fit(cells[:i] + cells[i + 1:])
            out = pr(c["model"], c["ds"], c["proto"], c["metric"])
        except KeyError:
            continue
        if out is None:        # held-out cell was the only obs of its (method, task) — skip
            continue
        mean, lo, hi = out
        ls = max((np.log(hi) - np.log(lo)) / 2, 1e-9)        # log-space predictive std
        z.append(abs(np.log(c["y"]) - np.log(mean)) / ls)    # |standardised residual|
    return float(np.percentile(z, 68)) if z else 1.0          # 68th pct → 68% coverage at ±1σ


def suggest(cells, measured, methods, datasets, predict, info_gain, topk=15):
    """Rank RUNNABLE next-evals by expected info-gain per GPU-minute."""
    supports = {}  # dataset -> metrics it actually scores (data-driven mask)
    for c in cells:
        supports.setdefault(c["ds"], set()).add(c["metric"])
    home = {}  # method -> home DEPTH protocol (its most common depth recipe)
    for c in cells:
        if METRIC_TASK[c["metric"]] == "depth":
            home.setdefault(c["model"], Counter())[c["proto"]] += 1
    home = {k: v.most_common(1)[0][0] for k, v in home.items()}

    def proto_for(m, d, k):  # the recipe we'd actually run this eval under
        if k in ("abs_rel", "delta_1"):
            return home.get(m)
        if k == "maa":
            return "RE10K wide-baseline" if d == "RE10K" else "PoseDiff (10-frame)"
        return "trajectory (Sim3 ATE)"  # ate
    nseen = Counter()  # observations per factor level (for the cold-start note)
    for c in cells:
        nseen[("m", c["model"])] += 1
        nseen[("d", c["ds"])] += 1
        nseen[("p", c["proto"])] += 1

    q = yaml.safe_load(open("reproductions/gpu_queue.yaml"))
    dscost = {}
    for j in q.get("jobs", []):
        if j.get("est_wall_min"):
            for d in j.get("datasets") or []:
                dscost.setdefault(d, []).append(j["est_wall_min"])
    dscost = {d: statistics.median(v) for d, v in dscost.items()}

    def cost(ds, m):
        return dscost.get(DS_QUEUE.get(ds, ""), 60) * (3 if m in SLOW else 1)

    cand = []
    for m in methods:
        for d in datasets:
            for k in PRIMARY:
                if not can_do(m, k) or k not in supports.get(d, ()):  # capability + dataset×metric mask
                    continue
                p = proto_for(m, d, k)
                if not p or (m, d, k, p) in measured:
                    continue
                try:
                    ig, pred = info_gain(m, d, p, METRIC_TASK[k]), predict(m, d, p, k)[0]
                except KeyError:
                    continue
                c = cost(d, m)
                cold = min(nseen[("m", m)], nseen[("d", d)], nseen[("p", p)])
                cand.append((ig / c, m, d, k, p, pred, ig, c, cold))
    cand.sort(reverse=True)
    print("NEXT EVALS — runnable, capability+dataset masked, ranked by info-gain / GPU-minute\n")
    print(f"  {'#':<3}{'method':14}{'dataset':9}{'metric':8}{'~pred':>8}{'min':>5}{'ig/min':>8}  note")
    for i, (s, m, d, k, p, pred, ig, c, cold) in enumerate(cand[:topk], 1):
        note = f"·{p[:22]}" + ("  ⟵ cold-start unlock" if cold <= 1 else "")
        print(f"  {i:<3}{m:14}{d:9}{k:8}{pred:8.3f}{c:5.0f}{s:8.3f}  {note}")
    print(f"\n  ({len(cand)} runnable candidates total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", action="store_true", help="emit the fitted model for in-browser prediction")
    ap.add_argument("--loo", action="store_true", help="leave-one-out cross-validation")
    ap.add_argument("--suggest", action="store_true", help="rank runnable next-evals by info-gain per GPU-minute")
    args = ap.parse_args()

    cells = load_cells()
    measured = {(c["model"], c["ds"], c["metric"], c["proto"]) for c in cells}
    kappa = calibrate(cells)
    methods, datasets, protos, predict, info_gain, model = fit(cells, calib=kappa)

    if args.model:
        print(json.dumps(model, separators=(",", ":")))
        return

    if args.loo:
        errs = {k: [] for k in PRIMARY}
        hits = tot = cold = 0
        for i in range(len(cells)):
            c = cells[i]
            _, _, _, pr, _, _ = fit(cells[:i] + cells[i + 1:], calib=kappa)
            try:
                out = pr(c["model"], c["ds"], c["proto"], c["metric"])
            except KeyError:  # held-out cell was the sole observation of some level
                out = None
            if out is None:   # also sole obs of its (method, task) → can't predict
                cold += 1
                continue
            mean, lo, hi = out
            errs[c["metric"]].append(abs(mean - c["y"]) / c["y"])
            tot += 1
            hits += lo <= c["y"] <= hi
        print(f"LEAVE-ONE-OUT CV  ({len(cells)} cells, calibration κ={kappa:.2f})\n")
        for k in PRIMARY:
            if errs[k]:
                print(f"  {k:8} median rel-err {statistics.median(errs[k]) * 100:5.1f}%  "
                      f"(max {max(errs[k]) * 100:.0f}%)  n={len(errs[k])}")
        print(f"\n  calibration: {hits}/{tot} held-out scores fell within ±1σ "
              f"({100*hits/tot:.0f}%; well-calibrated ≈ 68%)")
        print(f"  cold-start: {cold}/{len(cells)} cells are the SOLE observation of some "
              f"method/dataset/protocol → unpredictable when held out")
        return

    if args.suggest:
        suggest(cells, measured, methods, datasets, predict, info_gain)
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
            pr = predict(m, d, p, k)
            if pr is None:
                continue
            mean, lo, hi = pr
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

    def ig_mdp(m, d, p):  # best info-gain over the tasks this method can do
        return max(info_gain(m, d, p, t) for t in CAPABILITY.get(m, ["depth", "pose"]))
    cand = sorted(((m, d, p, ig_mdp(m, d, p)) for m in methods for d in datasets for p in protos
                   if (m, d, p) not in seen), key=lambda r: -r[3])[: args.topk]
    for i, (m, d, p, ig) in enumerate(cand, 1):
        print(f"  {i:<3} {m:13} {d:8} ·{p:28} ig {ig:.2f}")


if __name__ == "__main__":
    main()
