# BO active-learning backlog (auto-generated)

The full ranked set of **runnable, well-supported, unmeasured** prediction-matrix cells —
so "what's queueable" is transparent, not hand-picked. Regenerate with:

```
python scripts/bo_backlog.py --task depth --md   # or --task pose
```

Gate: the model already does the task (has a measured cell), and the dataset, protocol,
and metric are each independently ✅-verified by a neighbor. **NEW** = model×dataset never
measured; **CROSSPROTO** = measured under one protocol, predicted under another (e.g. MoGe is
moge-eval-native, so its eigen-2014 column is a prediction).

# BO backlog — depth (522 runnable: 230 new, 292 cross-protocol)

Top 50 by uncertainty (σ); full list via the script:

| σ | predicted | 95% CI | model | dataset | metric | protocol | kind |
|---|---|---|---|---|---|---|---|
| 0.73 | ~0.0577 | [0.014, 0.243] | UniK3D Large | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0512 | [0.013, 0.207] | MoGe ViT-L | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0577 | [0.014, 0.232] | DA-V2 Small | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0557 | [0.014, 0.224] | DA-V2 Base | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0821 | [0.021, 0.329] | Metric3D-v2 L | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0703 | [0.018, 0.281] | Metric3D-v2 G | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.1031 | [0.026, 0.413] | Marigold v1-1 | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0625 | [0.016, 0.250] | DA3 | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.71 | ~0.0422 | [0.011, 0.168] | UniK3D Large | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.71 | ~0.1350 | [0.034, 0.538] | Depth Pro | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.70 | ~0.0576 | [0.014, 0.229] | DA-V2 Large | Bonn | abs_rel | video (per-sequence) | NEW |
| 0.70 | ~0.0683 | [0.017, 0.270] | UniK3D Large | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.68 | ~0.0374 | [0.010, 0.143] | MoGe ViT-L | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0422 | [0.011, 0.160] | DA-V2 Small | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0407 | [0.011, 0.154] | DA-V2 Base | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.1020 | [0.027, 0.386] | DUSt3R | iBims-1 | abs_rel | metric (no-align) | NEW |
| 0.68 | ~0.0605 | [0.016, 0.229] | MoGe ViT-L | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.68 | ~0.0599 | [0.016, 0.227] | Metric3D-v2 L | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0513 | [0.014, 0.194] | Metric3D-v2 G | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0753 | [0.020, 0.285] | Marigold v1-1 | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0456 | [0.012, 0.172] | DA3 | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0911 | [0.024, 0.343] | CUT3R | iBims-1 | abs_rel | metric (no-align) | NEW |
| 0.68 | ~0.0743 | [0.020, 0.280] | MonST3R | iBims-1 | abs_rel | metric (no-align) | NEW |
| 0.68 | ~0.0986 | [0.026, 0.371] | Depth Pro | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.68 | ~0.0421 | [0.011, 0.158] | DA-V2 Large | NYU | abs_rel | dust3r-table2 (eigen+ratio-med) | CROSSPROTO |
| 0.67 | ~0.0683 | [0.018, 0.256] | DA-V2 Small | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0659 | [0.018, 0.247] | DA-V2 Base | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0972 | [0.026, 0.362] | Metric3D-v2 L | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0832 | [0.022, 0.310] | Metric3D-v2 G | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.1221 | [0.033, 0.455] | Marigold v1-1 | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0740 | [0.020, 0.275] | DA3 | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.1598 | [0.043, 0.594] | Depth Pro | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.1452 | [0.039, 0.540] | DUSt3R | DIODE | abs_rel | metric (no-align) | NEW |
| 0.67 | ~0.1026 | [0.028, 0.381] | DUSt3R | NYU | abs_rel | metric (no-align) | CROSSPROTO |
| 0.67 | ~0.1315 | [0.035, 0.488] | DUSt3R | DDAD | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.0682 | [0.018, 0.253] | DA-V2 Large | Bonn | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0865 | [0.023, 0.321] | MonST3R | KITTI | abs_rel | metric (no-align) | CROSSPROTO |
| 0.67 | ~0.1174 | [0.032, 0.435] | CUT3R | DDAD | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.1057 | [0.029, 0.392] | MonST3R | DIODE | abs_rel | metric (no-align) | NEW |
| 0.67 | ~0.1296 | [0.035, 0.480] | CUT3R | DIODE | abs_rel | metric (no-align) | NEW |
| 0.67 | ~0.0556 | [0.015, 0.206] | DUSt3R | iBims-1 | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.0957 | [0.026, 0.354] | MonST3R | DDAD | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.1060 | [0.029, 0.392] | CUT3R | KITTI | abs_rel | metric (no-align) | CROSSPROTO |
| 0.67 | ~0.3119 | [0.084, 1.152] | UniK3D Large | Sintel | abs_rel | dust3r-lineage (median, no-crop) | NEW |
| 0.67 | ~0.0497 | [0.013, 0.183] | CUT3R | iBims-1 | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.0747 | [0.020, 0.275] | MonST3R | NYU | abs_rel | metric (no-align) | CROSSPROTO |
| 0.67 | ~0.0405 | [0.011, 0.149] | MonST3R | iBims-1 | abs_rel | moge-eval (affine) | NEW |
| 0.67 | ~0.1188 | [0.032, 0.438] | DUSt3R | KITTI | abs_rel | metric (no-align) | CROSSPROTO |
| 0.66 | ~0.1799 | [0.049, 0.660] | DUSt3R | ETH3D | abs_rel | metric (no-align) | NEW |
