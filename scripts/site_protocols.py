"""Single source of truth for cell protocols: bake the recipe family into each site
cell's `proto` field, so the site (protoOf) and forecaster (proto_of) just READ it —
no duplicated heuristic that drifts as cells are added.

Source priority per cell:
  1. the recorded `protocol:` of its reproduction YAML, IF exactly one reproduction
     matches (same model, dataset, task) — authoritative, and auto-corrects the cite
     heuristic (e.g. DA3-GSO → moge-eval);
  2. otherwise the eval source from the citation (`proto_of`), which correctly
     captures baseline cells reported inside another paper's table (DA-V2 in MoGe's
     Table 3 → moge-eval), where no dedicated YAML exists.

    python scripts/site_protocols.py            # report
    python scripts/site_protocols.py --write    # bake `proto` into explore.html
"""

import argparse
import glob
import importlib.util
import json
import re

import yaml

_spec = importlib.util.spec_from_file_location("bo", "scripts/bo_table_forecast.py")
bo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bo)

DISPLAY2REG = {
    "DA-V2 Small": "depth-anything-v2", "DA-V2 Base": "depth-anything-v2", "DA-V2 Large": "depth-anything-v2",
    "MoGe ViT-L": "moge", "Metric3D-v2 L": "metric3d-v2", "Metric3D-v2 G": "metric3d-v2",
    "Marigold v1-1": "marigold", "DA3": "depth-anything-3", "Depth Pro": "depth-pro",
    "CUT3R": "cut3r", "MonST3R": "monst3r", "DUSt3R": "dust3r", "MASt3R": "mast3r",
    "VGGT": "vggt", "DAGE": "dage", "UniK3D Large": "unik3d",
}
DS_NORM = {"NYU": "nyu", "KITTI": "kitti", "DIODE": "diode", "ETH3D": "eth3d", "GSO": "gso",
           "iBims-1": "ibims", "DDAD": "ddad", "Sintel": "sintel", "Bonn": "bonn",
           "Sun-RGBD": "sun_rgbd", "Booster": "booster", "CO3Dv2": "co3dv2",
           "RE10K": "realestate", "TUM-Dyn": "tum"}
DEPTH_METRICS = ("abs_rel", "delta_1")


def task_of(metric):
    return "depth" if metric in DEPTH_METRICS else "pose"


def family_of_protocol(proto, align, dsl):
    if proto:
        p = proto
        if "moge" in p:
            return "moge-eval (affine)"
        if p.endswith("_dav2"):
            return "dav2-native (affine)"
        if "dust3r_table2" in p:
            return "dust3r-table2 (eigen+ratio-med)"
        if "dust3r_lineage" in p or p.startswith("bonn_lineage") or p.startswith("bonn_dust3r"):
            return "dust3r-lineage (median, no-crop)"
        if p.endswith("_metric"):
            return "metric (no-align)"
        if "eigen" in p or p == "marigold_kitti_eval":
            return "eigen-2014 (crop+median)"
        if "co3dv2" in p or "vggt_pose" in p:
            return "PoseDiff (10-frame)"
        if "realestate10k" in p:
            return "RE10K wide-baseline"
    if align == "median_lineage" or (align == "median" and "bonn" in dsl):
        return "dust3r-lineage (median, no-crop)"
    if align == "scale_weiszfeld":
        return "video (per-sequence)"
    if align in ("scale_shift", "scale_shift_clamped"):
        return "dav2-native (affine)"
    return None


def pose_family(metric, ds):
    if metric in ("maa", "rra", "rta"):
        return "RE10K wide-baseline" if ds == "RE10K" else "PoseDiff (10-frame)"
    if metric in ("ate", "rpe_t", "rpe_r"):
        return "trajectory (Sim3 ATE)"
    return None


def ds_of(d):
    proto = (d.get("protocol") or "").lower()
    ds = (d.get("dataset") or {}).get("name") if isinstance(d.get("dataset"), dict) else (d.get("dataset") or "")
    src = (proto + " " + str(ds)).lower()
    for norm in DS_NORM.values():
        if norm in src:
            return norm
    return None


def build_index():
    """(reg-model, ds-norm, task) -> set of families across reproductions."""
    idx = {}
    for f in glob.glob("reproductions/*.yaml"):
        try:
            d = yaml.safe_load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        model = (d.get("model") or {}).get("name")
        dsn = ds_of(d)
        pm = (d.get("paper_reference") or {}).get("primary_metric") or ""
        task = "depth" if pm in ("abs_rel", "rmse", "delta_1", "log10", "silog") else "pose"
        dsl = (str(d.get("dataset") or "") + (d.get("protocol") or "")).lower()
        fam = family_of_protocol(d.get("protocol"), d.get("scale_alignment"), dsl)
        if model and dsn and fam:
            idx.setdefault((model, dsn, task), set()).add(fam)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    idx = build_index()
    h = open("site/explore.html").read()
    baked = warns = 0

    for kind in ("verified-cells", "companion-cells", "info-cells"):
        m = re.search(rf'id="{kind}">(.*?)</script>', h, re.S)
        arr = json.loads(m.group(1))
        for c in arr:
            cite = c.get("cite") or {}
            fam = (bo.proto_of(c["model"], c["ds"], c["metric"], cite.get("src", ""), cite.get("loc", ""))
                   if c["metric"] in ("abs_rel", "delta_1", "maa", "ate") else pose_family(c["metric"], c["ds"]))
            c["proto"] = fam or "other"
            baked += 1
            # GUARD: warn if the *only* task-matched reproduction's recorded protocol
            # disagrees AND this isn't a cross-paper baseline cell (src == the cell's
            # own model) — that's the signature of true drift, not a legit baseline.
            cand = idx.get((DISPLAY2REG.get(c["model"]), DS_NORM.get(c["ds"]), task_of(c["metric"])))
            src = re.sub(r"\s*\(.*?\)", "", cite.get("src", "")).strip()
            is_baseline = DISPLAY2REG.get(src) != DISPLAY2REG.get(c["model"])  # reported in another paper's table
            if not is_baseline and cand and len(cand) == 1 and next(iter(cand)) != fam:
                warns += 1
                print(f"  ⚠ DRIFT {c['model']} {c['ds']} {c['metric']}: cite={fam!r} but YAML={next(iter(cand))!r}")
        if args.write:
            block = ",\n".join(json.dumps(c, separators=(",", ":")) for c in arr)
            h = h[:m.start(1)] + "\n[\n" + block + "\n]\n" + h[m.end(1):]

    if args.write:
        open("site/explore.html", "w").write(h)
        print("WROTE proto into site/explore.html")
    print(f"\nbaked {baked} cells; {warns} YAML-vs-cite warnings for review")


if __name__ == "__main__":
    main()
