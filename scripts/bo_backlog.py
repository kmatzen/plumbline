#!/usr/bin/env python3
"""Emit the full runnable BO active-learning backlog, ranked by uncertainty.

Every unmeasured-but-well-supported cell the prediction model can estimate, so
"what's queueable" is transparent and reproducible instead of hand-picked. Two
classes are surfaced:

  * NEW       — (model, dataset, metric) never measured in any protocol.
  * CROSSPROTO — measured under one protocol, but shown predicted under another
                 (e.g. MoGe is moge-eval-native, so its eigen-2014 column is a
                 prediction). These are easy to miss because a (model, dataset,
                 metric) key looks "measured" while a specific protocol cell is not.

Gate (per the campaign mandate): the model must already do this task (have a
measured cell for it), and the dataset and protocol must each be ✅-verified by
some neighbor — so model, dataset, protocol, metric are each independently proven.

Usage:  python scripts/bo_backlog.py [--task depth|pose] [--md]
Reads the embedded model-fit + cells from site/explore.html (the single source).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site" / "explore.html"
DEPTH_METRICS = ("abs_rel", "delta_1")


def _load():
    h = SITE.read_text(encoding="utf-8")

    def arr(name):
        m = re.search(rf'id="{name}">\s*(\[.*?\])\s*</script>', h, re.S)
        return json.loads(m.group(1))

    fit = json.loads(re.search(r'id="model-fit">\s*(\{.*?\})\s*</script>', h, re.S).group(1))
    cells = arr("verified-cells") + arr("companion-cells") + arr("info-cells")
    return fit, cells, arr("verified-cells")


def build(task: str = "depth"):
    fit, cells, verified = _load()
    MT = fit["metric_task"]
    po = lambda c: c.get("proto", "other")
    metrics = [k for k, t in MT.items() if t == task and k in fit["stats"]]

    # measured keys (protocol-agnostic) and protocol-specific
    measured_k = {(c["model"], c["ds"], c.get("metric")) for c in cells}
    measured_kp = {(c["model"], c["ds"], c.get("metric"), po(c)) for c in cells}
    # a model "does" this task iff it has a measured cell for it
    task_models = {c["model"] for c in cells if MT.get(c.get("metric")) == task}
    V_ds = {c["ds"] for c in verified}
    V_proto = {po(c) for c in verified}
    task_protos = sorted({po(c) for c in cells if MT.get(c.get("metric")) == task})
    # a (dataset, protocol) is a REAL grid combo only if some model was evaluated
    # there — keeps cross-protocol cells (NYU is run under eigen-2014) but rejects
    # nonsense (NYU under a video-per-sequence protocol). This is the constraint to
    # keep; only the per-model (model,protocol) requirement is dropped.
    dp_ok = {(c["ds"], po(c)) for c in cells}

    FMI = {m: i for i, m in enumerate(fit["methods"])}
    FDI = {d: i for i, d in enumerate(fit["datasets"])}
    FPI = {p: i for i, p in enumerate(fit["protos"])}
    FMTI = {mt: i for i, mt in enumerate(fit["method_tasks"])}
    mu, Sig = fit["mu"], fit["Sigma"]
    nMT, nD, noise2, calib = fit["nMT"], fit["nD"], fit["noise2"], fit.get("calib", 1)

    def predict(m, d, p, k):
        mt = f"{m}|{task}"
        if mt not in FMTI or m not in FMI or d not in FDI or p not in FPI:
            return None
        idx = [0, 1 + FMTI[mt], 1 + nMT + FDI[d], 1 + nMT + nD + FPI[p]]
        gm = sum(mu[i] for i in idx)
        gv = noise2 + sum(Sig[i][j] for i in idx for j in idx)
        m_, sd, sign = fit["stats"][k]
        lm = m_ + sign * sd * gm
        ls = sd * math.sqrt(gv) * calib
        lob, hib = fit.get("range", {}).get(k, [0, float("inf")])
        e = lambda x: min(max(math.exp(x), lob), hib)
        return ls, e(lm), e(lm - 1.96 * ls), e(lm + 1.96 * ls)

    rows = []
    for m in task_models:
        cap = fit["cap"].get(m, [])
        if task not in cap:
            continue
        for d in fit["datasets"]:
            if d not in V_ds:
                continue
            for p in task_protos:
                if p not in V_proto or (d, p) not in dp_ok:
                    continue
                for k in metrics:
                    if k not in fit["supports"].get(d, []):
                        continue
                    if (m, d, k, p) in measured_kp:
                        continue
                    r = predict(m, d, p, k)
                    if not r:
                        continue
                    kind = "NEW" if (m, d, k) not in measured_k else "CROSSPROTO"
                    rows.append(dict(sigma=r[0], pred=r[1], lo=r[2], hi=r[3],
                                     model=m, ds=d, metric=k, proto=p, kind=kind))
    rows.sort(key=lambda r: r["sigma"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="depth", choices=["depth", "pose"])
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = build(args.task)
    if args.limit:
        rows = rows[: args.limit]
    n_new = sum(r["kind"] == "NEW" for r in rows)
    n_x = len(rows) - n_new
    if args.md:
        print(f"# BO backlog — {args.task} ({len(rows)} runnable: {n_new} new, {n_x} cross-protocol)\n")
        print("| σ | predicted | 95% CI | model | dataset | metric | protocol | kind |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['sigma']:.2f} | ~{r['pred']:.4f} | [{r['lo']:.3f}, {r['hi']:.3f}] | "
                  f"{r['model']} | {r['ds']} | {r['metric']} | {r['proto']} | {r['kind']} |")
    else:
        print(f"{len(rows)} runnable {args.task} cells ({n_new} new, {n_x} cross-protocol), by σ:")
        for r in rows:
            print(f"  σ={r['sigma']:.2f} ~{r['pred']:.4f} [{r['lo']:.3f},{r['hi']:.3f}]  "
                  f"{r['model']} / {r['ds']} / {r['metric']}  [{r['proto']}]  {r['kind']}")


if __name__ == "__main__":
    main()
