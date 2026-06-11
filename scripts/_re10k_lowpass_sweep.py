"""RE10K pose native-resolution LOW-PASS sweep (1080 Ti diagnostic, NOT a paper-match).

Decisive companion to scripts/_re10k_resolution_sweep.py. The downsample sweep
showed VGGT pose AUC@30 tracing an inverted-U that peaks at 288px (0.7731) vs
native 640 (0.6724), and concluded the gap is high-frequency / re-encode-artifact
driven (a low-pass helps) rather than pixel-count driven. But that downsample knob
confounds two effects: (a) attenuating the HF band, and (b) the bicubic-down /
VGGT-up resize round-trip. This sweep isolates (a): it Gaussian-blurs each frame IN
PLACE at native resolution (the `lowpass_sigma` loader kwarg) — same pixel count,
aspect, and VGGT internal resize path as native, only the HF band is attenuated.

If AUC@30 recovers toward the 0.7731 downsample peak under a pure native-res
low-pass, the gap is artifact/HF-driven and the actionable fix is an anti-alias
prefilter (NOT a re-scrape, NOT a resolution change). If AUC stays flat near the
native 0.6724, the HF-artifact framing is wrong and the 288px result is about the
resize round-trip or a VGGT resolution prior instead.

sigma=0.0 is the native control and MUST reproduce the recorded 0.6724 — it is the
apparatus-validation gate for the reconstructed `re10k_sub50` subset.

Run:  python scripts/_re10k_lowpass_sweep.py <model> <root> <out.json> [sigma,sigma,...]
"""

import json
import sys

from plumbline.cli import _eager_import_adapters
from plumbline.datasets.registry import DATASET_REGISTRY
from plumbline.models.registry import MODEL_REGISTRY
from plumbline.runner import evaluate

# Exact pose recipe from reproductions/vggt_realestate10k_pose.yaml.
POSE_KW = dict(
    tasks=["pose"],
    scale_alignment="none",
    max_views=10,
    device="cuda:0",
    pose_translation_antipodal=True,
    pose_auc_mode="vggt_co3d_histogram",
    pose_auc_thresholds=(3.0, 5.0, 15.0, 30.0),
    pose_acc_thresholds=(5.0, 15.0),
)

# re10k_sub50 = first 50 sorted usable clips. The downsample sweep ran on a
# dedicated 50-clip root; here we reconstruct the identical subset from the full
# 1800-clip root so sigma=0 reproduces the recorded native 0.6724.
SUB50_N = 50


def main() -> None:
    model_name, root, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if len(sys.argv) > 4:
        sigma_list = [float(s) for s in sys.argv[4].split(",")]
    else:
        sigma_list = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    _eager_import_adapters()

    model_cls = MODEL_REGISTRY[model_name]
    ds_cls = DATASET_REGISTRY["realestate10k-pose-eval"]

    # Reconstruct re10k_sub50: instantiate the full loader and take the first 50
    # usable clips in the loader's own sorted order (identical to the dedicated
    # 50-clip root the downsample sweep used).
    full = ds_cls(root=root, num_frames=10, seed=0)
    sub50 = [rec["clip_id"] for rec in full._records[:SUB50_N]]
    print(f"reconstructed re10k_sub50: {len(sub50)} clips (of {len(full._records)} usable)", flush=True)

    model_kwargs = {"device": "cuda:0"}
    if model_name == "vggt":
        model_kwargs["dtype"] = "float32"  # Pascal sm_61 has no bf16
    model = model_cls(**model_kwargs)

    rows = []
    for sigma in sigma_list:
        ds = ds_cls(root=root, clips=sub50, num_frames=10, seed=0, lowpass_sigma=sigma)
        report = evaluate(model=model, dataset=ds, **POSE_KW)
        m = report.aggregate_metrics
        row = {
            "model": model_name,
            "lowpass_sigma": sigma,
            "n_samples": len(ds),
            "auc@30": m.get("pairwise_pose_auc@30"),
            "auc@15": m.get("pairwise_pose_auc@15"),
            "RRA@15": m.get("pairwise_RRA@15"),
            "RTA@15": m.get("pairwise_RTA@15"),
            "rot_err_deg": m.get("pairwise_rot_err_deg_mean"),
            "trans_cos_err_deg": m.get("pairwise_trans_cos_err_deg_mean"),
        }
        rows.append(row)
        print(
            f"[{model_name}] sigma={sigma:>4}  AUC@30={row['auc@30']:.4f}  "
            f"trans_cos={row['trans_cos_err_deg']:.2f}deg  "
            f"RRA@15={row['RRA@15']:.3f}  n={row['n_samples']}",
            flush=True,
        )

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
