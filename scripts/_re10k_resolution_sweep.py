"""RE10K pose resolution-sensitivity sweep (1080 Ti diagnostic, NOT a paper-match).

Tests the documented PRIME SUSPECT for the vggt-realestate10k-pose gap: that
VGGT's feed-forward translation regression is resolution/softness-sensitive on
RE10K-style content, while correspondence-based MASt3R/DUSt3R are robust.

Pose GT is resolution-independent and the protocol uses no GT focals, so feeding
progressively softer frames (BICUBIC downsample of the staged 640x360 clips, via
the new `downsample_long_side` loader kwarg) isolates each model's sensitivity to
input resolution. Same loader + pose-AUC settings as the reproduction config so
the native-res point matches the recorded cross-model control.

Run:  python scripts/_re10k_resolution_sweep.py <model> <root> <out.json> [res,res,...]
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


def main() -> None:
    model_name, root, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if len(sys.argv) > 4:
        res_list = [None if r.lower() in ("none", "native") else int(r) for r in sys.argv[4].split(",")]
    else:
        res_list = [None, 384, 288, 192, 128, 96]

    _eager_import_adapters()

    model_cls = MODEL_REGISTRY[model_name]
    ds_cls = DATASET_REGISTRY["realestate10k-pose-eval"]

    model_kwargs = {"device": "cuda:0"}
    if model_name == "vggt":
        model_kwargs["dtype"] = "float32"  # Pascal sm_61 has no bf16
    model = model_cls(**model_kwargs)

    rows = []
    for res in res_list:
        ds = ds_cls(root=root, num_frames=10, seed=0, downsample_long_side=res)
        report = evaluate(model=model, dataset=ds, **POSE_KW)
        m = report.aggregate_metrics
        row = {
            "model": model_name,
            "downsample_long_side": res,
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
            f"[{model_name}] res={str(res):>5}  AUC@30={row['auc@30']:.4f}  "
            f"trans_cos={row['trans_cos_err_deg']:.2f}deg  "
            f"RRA@15={row['RRA@15']:.3f}  n={row['n_samples']}",
            flush=True,
        )

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
