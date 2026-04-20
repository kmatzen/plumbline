# Paper-reference audit — 2026-04-20

Independent audit of `paper_reference.value` and `paper_reference.citation` entries
against the canonical arXiv sources. Performed read-only; no YAMLs were modified.

## Summary

- **YAMLs with a pinned, paper-comparable value**: 23
- **VERIFIED (value + table + column + row all match paper)**: 9
- **WRONG_TABLE (value matches but table number in citation is wrong)**: 5
- **WRONG_VALUE (right table/column/row, number doesn't match paper cell)**: 2
- **WRONG_ROW (citation references a different model variant than the YAML uses)**: 2
- **NOT_FOUND (no such number in the cited paper)**: 2
- **N/A (informational-only, placeholder value, or no paper target)**: 11

**Biggest concerns:**
1. `depth_pro_nyuv2.yaml` — paper does NOT evaluate on NYUv2; the 0.961 δ₁ has no
   source cell. The Depth Pro paper's Table 1 evaluates Booster, ETH3D, Middlebury,
   NuScenes, Sintel, Sun-RGBD only. Classified NOT_FOUND.
2. `depth_anything_v2_sintel.yaml` — DA-V2 Table 2 reports ViT-L Sintel
   AbsRel = 0.487, not 0.075. Value appears fabricated for a smoke test.
3. Systemic MoGe citation-table error: every YAML that cites MoGe Table 2 for
   depth-map estimation is wrong — MoGe depth results are in **Table 3**; Table 2
   is point-map estimation. The values themselves mostly match, but the cited
   table number is off.
4. `moge2_vitl_kitti.yaml` — cites Table B.4 (ViT-Base ablation) for a
   ViT-Large model run. The 3.92 Reld value comes from the ViT-Base ablation
   block, not the ViT-Large result (ViT-L is averaged-only in Table 1).

## Per-YAML audit table

| YAML | Model | Dataset | Metric | Claimed | Paper cell (verified source) | Verified? | Notes |
|---|---|---|---|---|---|---|---|
| da_v2_small_nyuv2 | DA-V2 ViT-S | NYU Eigen | abs_rel | 0.053 | DA-V2 Table 2 (zero-shot relative), NYU-D AbsRel col, ViT-S row = 0.053 | VERIFIED | YAML already self-flagged as "UNVERIFIED"; this audit confirms the paper cell. |
| da_v2_small_kitti | DA-V2 ViT-S | KITTI Eigen | abs_rel | 0.078 | DA-V2 Table 2, KITTI AbsRel col, ViT-S row = 0.078 | VERIFIED | |
| da_v2_base_nyuv2 | DA-V2 ViT-B | NYU Eigen | abs_rel | 0.049 | DA-V2 Table 2, NYU-D AbsRel col, ViT-B row = 0.049 | VERIFIED | YAML self-flagged "UNVERIFIED"; confirmed here. |
| da_v2_base_kitti | DA-V2 ViT-B | KITTI Eigen | abs_rel | 0.078 | DA-V2 Table 2, KITTI AbsRel col, ViT-B row = 0.078 | VERIFIED | |
| da_v2_large_kitti | DA-V2 ViT-L | KITTI Eigen | abs_rel | 0.074 | DA-V2 Table 2, KITTI AbsRel col, ViT-L row = 0.074 | VERIFIED | |
| da_v2_large_nyuv2 | DA-V2 ViT-L | NYU Eigen | abs_rel | 0.0420 | MoGe Table 3 (NOT Table 2), affine-invariant disparity, NYUv2 col, DA V2 row = 4.20 → 0.0420 | WRONG_TABLE | Value matches, but citation says "MoGe paper Table 2, NYU column, affine-invariant disparity row, DA-V2 ViT-L baseline." MoGe depth results are in Table 3; Table 2 is point-map estimation. Also note the DA-V2 paper's OWN Table 2 reports DA-V2 ViT-L NYU AbsRel = 0.045 (a different evaluation protocol); the 0.0420 is specifically the MoGe-paper re-evaluation under the affine-invariant disparity protocol. |
| da_v2_small_diode_indoor | DA-V2 ViT-**S** | DIODE val indoor | abs_rel | 0.0533 | MoGe Table 3, affine-invariant disparity, DIODE col, DA V2 row = 5.33 → 0.0533 | WRONG_ROW | YAML model variant is ViT-S, but MoGe Table 3 reports DA-V2 only in ViT-L ("all methods utilize ViT-Large as backbone"). The 0.0533 number is the ViT-L DIODE result; there is no published ViT-S DIODE number under aff-inv-disparity alignment in either the DA-V2 paper or the MoGe paper. (DA-V2's own Table 2 reports ViT-S DIODE AbsRel=0.073 under the DA-V2 protocol.) Additionally the citation says "Table 2" but MoGe depth table is Table 3. |
| moge_vitl_nyuv2 | MoGe-1 ViT-L | NYU Eigen | abs_rel | 0.0341 | MoGe Table 3, affine-invariant disparity, NYUv2 col, MoGe row = 3.41 → 0.0341 | WRONG_TABLE | Value matches. Citation says "Table 2 (Depth Map Est.)"; canonical MoGe paper has Table 2 = point map estimation, Table 3 = depth map estimation. Everything else (column, row, variant) is correct. |
| moge_vitl_kitti | MoGe-1 ViT-L | KITTI Eigen | abs_rel | 0.0405 | MoGe Table 3, affine-invariant disparity, KITTI col, MoGe row = **4.08** → 0.0408 | WRONG_VALUE | Citation says Table 2; should be Table 3. Value off by ~0.7% (0.0405 claimed vs 0.0408 in paper). YAML comment reads "Reld 4.05" but paper cell reads 4.08 in both arXiv v1 and ar5iv canonical renderings. |
| moge_vitl_diode_indoor | MoGe-1 ViT-L | DIODE val indoor | abs_rel | 0.0400 | MoGe Table 3, affine-invariant disparity, DIODE col, MoGe row = 4.00 → 0.0400 | WRONG_TABLE | Value matches paper cell. Citation says Table 2; should be Table 3. Note: paper reports combined val (325 indoor + 446 outdoor); YAML runs indoor-only — the YAML comments flag this clearly. |
| moge_vitl_diode_both | MoGe-1 ViT-L | DIODE val combined | abs_rel | 0.0400 | MoGe Table 3, affine-invariant disparity, DIODE col, MoGe row = 4.00 → 0.0400 | WRONG_TABLE | Same as above but on the apples-to-apples combined slice. Value matches; citation says Table 2, should be Table 3. |
| moge_vitl_gso | MoGe-1 ViT-L | GSO | abs_rel | 0.0 (placeholder) | — | N/A | YAML explicitly marks value as a placeholder; tolerance 1.0. Informational only. |
| moge2_vitl_nyuv2 | MoGe-2 ViT-L | NYU Eigen | abs_rel | null | — | N/A | No `value` pinned; YAML notes "UNVERIFIED — needs direct paper read". Confirmed: MoGe-2 paper has no per-dataset ViT-L NYU number in any table (Table 1 is averaged; Table B.4 is ViT-Base ablation). |
| moge2_vitl_nyuv2_metric | MoGe-2 ViT-L (metric) | NYU Eigen | abs_rel | null | — | N/A | Informational metric run, no paper target. |
| moge2_vitl_kitti | MoGe-2 ViT-**L** | KITTI Eigen | abs_rel | 0.0392 | MoGe-2 Table B.4 (appendix), affine-invariant disparity, KITTI col — but this block is ViT-**Base** ablation, not ViT-Large | WRONG_ROW | The 3.92 Reld value exists in the paper but only in the **ViT-Base ablation block** of Table B.4. ViT-Large per-dataset KITTI is NOT reported in the paper; main-paper Table 1 only publishes averages across 10 datasets (no per-dataset breakdown for the full ViT-L model). The YAML uses `variant: 2-vitl` (ViT-Large), so this is a ViT-Base number applied to a ViT-Large run. |
| metric3d_v2_nyuv2 | Metric3D-v2 ViT-L | NYU Eigen | abs_rel | 0.063 | Metric3D-v2 Table I, NYU zero-shot, ViT-L CSTM_label = 0.063 | VERIFIED | |
| metric3d_v2_kitti | Metric3D-v2 ViT-L | KITTI Eigen | abs_rel | 0.052 | Metric3D-v2 Table I, KITTI zero-shot, ViT-L CSTM_label = 0.052 | VERIFIED | |
| metric3d_v2_giant_nyuv2 | Metric3D-v2 ViT-g | NYU Eigen | abs_rel | 0.067 | Metric3D-v2 Table I, NYU zero-shot, ViT-g CSTM_label = 0.067 | VERIFIED | |
| metric3d_v2_giant_kitti | Metric3D-v2 ViT-g | KITTI Eigen | abs_rel | 0.051 | Metric3D-v2 Table I, KITTI zero-shot, ViT-g CSTM_label = 0.051 | VERIFIED | |
| marigold_v1_1_nyuv2 | Marigold v1-1 | NYU Eigen | abs_rel | 0.055 | Marigold Table 1 (NOT Table 2), NYUv2 AbsRel col, "Ours (w/ ensemble)" row = 5.5 → 0.055 | WRONG_TABLE | Value matches. Citation says "Table 2. NYUv2 Eigen test, v1-0 (affine-invariant)." Canonical paper: Table 1 = quantitative zero-shot comparison; Table 2 = ablation on training noise types. The depth comparison is in Table 1. |
| marigold_v1_1_kitti | Marigold v1-1 | KITTI Eigen | abs_rel | 0.099 | Marigold Table 1, KITTI AbsRel col, "Ours (w/ ensemble)" = 9.9 → 0.099 | VERIFIED | Citation correctly says Table 1. |
| depth_pro_nyuv2 | Depth Pro | NYU Eigen | delta_1 | 0.961 | — Paper does NOT evaluate on NYUv2 | NOT_FOUND | Depth Pro paper (Bochkovskii et al. 2024, arXiv:2410.02073) Table 1 evaluates only Booster, ETH3D, Middlebury, NuScenes, Sintel, Sun-RGBD. The paper explicitly states these datasets were chosen because "to our knowledge, they were never used in training any of the evaluated systems." Depth Pro δ₁ values in Table 1: Sun-RGBD 89.0, Booster 46.6, Middlebury 60.5, ETH3D 41.5, Sintel 40.0, NuScenes 49.1. No 0.961 / 96.1 appears anywhere for Depth Pro. The YAML comment itself hedged ("approximate... pin exact on first run") but the paper has no row to pin from. |
| depth_pro_kitti | Depth Pro | KITTI Eigen | abs_rel | null | — | N/A | YAML explicitly notes "No paper target". |
| da3_nyuv2 | DA3 Large-1.1 | NYU Eigen | delta_1 | 0.974 | DA3 Table 4 (monocular depth comparisons), NYU δ₁ col, DA3 row = 97.4 → 0.974 | VERIFIED | |
| da3_gso | DA3 Large-1.1 | GSO | abs_rel | 0.0 (placeholder) | — | N/A | Explicit placeholder; tolerance 1.0. |
| da3_eth3d_courtyard_chamfer | DA3 Large-1.1 | ETH3D courtyard | chamfer | null | — | N/A | Informational; no paper target. |
| da_v2_large_gso | DA-V2 ViT-L | GSO | abs_rel | 0.0 (placeholder) | — | N/A | Explicit placeholder; tolerance 1.0. |
| da_v2_metric_indoor_large_nyuv2 | DA-V2 metric-indoor-L | NYU Eigen | abs_rel | null | — | N/A | Informational; no paper target (different checkpoint from paper's NYU-finetuned ViT-L). |
| da_v2_metric_outdoor_large_kitti | DA-V2 metric-outdoor-L | KITTI Eigen | abs_rel | null | — | N/A | Informational; no paper target. |
| depth_anything_v2_sintel | DA-V2 ViT-L | Sintel | abs_rel | 0.075 | DA-V2 Table 2, Sintel AbsRel col, ViT-L row = **0.487** | WRONG_VALUE | The YAML comment already hedges ("rough sanity value; verify against upstream before declaring it the reference"). The claimed 0.075 does NOT appear in DA-V2 Table 2 — ViT-L Sintel is 0.487; no row gives 0.075 for Sintel. (0.073 is the ViT-S DIODE value; 0.075 is MiDaS v3.1 DIODE. Neither is Sintel.) The smoke-test YAML's value is fabricated/guessed. |
| vggt_paper_dtu_mvs | VGGT | DTU MVS | chamfer | 0.382 | VGGT Table 2 (Dense MVS on DTU), Overall col, VGGT row = 0.382 | VERIFIED | Acc 0.389, Comp 0.374, Overall 0.382 all match. |
| vggt_paper_scannet_depth | VGGT | ScanNet | abs_rel | null | — | N/A | Explicitly informational; paper does not evaluate ScanNet depth. |
| vggt_eth3d_multiscene_chamfer | VGGT | ETH3D (3-scene subset) | overall | 0.709 | VGGT Table 3 (Point Map on ETH3D), Overall col, "Ours (Point)" row = 0.709 | VERIFIED | Value matches paper cell for full cross-scene average. YAML correctly flags that it runs a 3-scene subset, not full split (tolerance 1.0). |
| vggt_eth3d_courtyard_chamfer | VGGT | ETH3D courtyard | chamfer | null | — | N/A | Informational; single-scene, not paper-comparable. |

## Source verification log

Sources consulted for the paper cells above:

- **Depth Anything V2** (Yang et al. 2024, arXiv:2406.09414) — arxiv.org/html/2406.09414,
  Table 2 cross-checked row-by-row for all three variants (ViT-S/B/L) across NYU-D,
  KITTI, Sintel, ETH3D, DIODE AbsRel columns.
- **MoGe** (Wang et al. 2024, arXiv:2410.19115) — ar5iv.labs.arxiv.org/html/2410.19115.
  Confirmed table-caption ordering: Table 1 = training sources; Table 2 = point map
  estimation; Table 3 = depth map estimation; Table 4 = FoV. All MoGe YAMLs citing
  "Table 2" for depth results are referencing the wrong table number. Table 3
  "affine-invariant disparity" section values confirmed: MoGe NYUv2 3.41, KITTI 4.08,
  DIODE 4.00; DA-V2 ViT-L NYUv2 4.20, KITTI 5.61, DIODE 5.33.
- **MoGe-2** (Wang et al. 2025, arXiv:2507.02546) — ar5iv.labs.arxiv.org/html/2507.02546.
  Table 1 reports ViT-Large results as averages across 10 datasets only; no per-dataset
  breakdown for the full ViT-L model. Table B.4 reports per-dataset results but only
  for the ViT-Base ablation variant. The 3.92 Reld KITTI aff-inv disparity cell comes
  from the ViT-Base ablation block in Table B.4, not a ViT-L cell.
- **Metric3D v2** (Hu/Yin et al. 2024, arXiv:2404.15506) — arxiv.org/html/2404.15506.
  Table I zero-shot NYU/KITTI values for ViT-L CSTM_label and ViT-g CSTM_label
  confirmed row-by-row.
- **Marigold** (Ke et al. 2024, arXiv:2312.02145) — arxiv.org/html/2312.02145.
  Table 1 = quantitative zero-shot comparison (AbsRel / δ₁ on 5 benchmarks);
  Table 2 = training-noise ablation. Depth numbers are in Table 1, not Table 2.
- **Depth Pro** (Bochkovskii et al. 2024, arXiv:2410.02073) — arxiv.org/html/2410.02073.
  Table 1 reports Booster / ETH3D / Middlebury / NuScenes / Sintel / Sun-RGBD only;
  NYUv2 is NOT in the evaluation set. The paper explicitly states these datasets were
  picked because "they were never used in training any of the evaluated systems."
- **Depth Anything 3** (Bytedance Seed 2025, arXiv:2511.10647) — arxiv.org/html/2511.10647v1.
  Table 4 "Monocular depth comparisons" columns KITTI/NYU/SINTEL/ETH3D/DIODE δ₁;
  DA3 row NYU δ₁ = 97.4 confirmed.
- **VGGT** (Wang et al. 2025, arXiv:2503.11651) — arxiv.org/html/2503.11651.
  Table 2 (Dense MVS on DTU) VGGT row: Acc 0.389, Comp 0.374, Overall 0.382.
  Table 3 (Point Map on ETH3D) VGGT row: Acc 0.901, Comp 0.518, Overall 0.709.
