# Paper-reference audit

Independent audit of `paper_reference.value` and `paper_reference.citation` entries
against the canonical arXiv sources. Performed read-only; no YAMLs were modified.

> **Current state (2026-05-28):** all 28 `verified_pdf` YAMLs that pin a
> non-null value are audited against the source PDF — see the
> "Verified-coverage status" section at the bottom. No remaining
> `verified_pdf` value is unaudited, fabricated, or inferred. A 2026-05-28
> re-verification of the 22 ✅ targets confirmed 21/22 against the papers
> and corrected two transcription slips read straight from the PDF:
> `dust3r-co3dv2-pose` mAA(30) 0.774→0.772 and `dust3r-kitti` companion
> δ₁ 0.8600→0.8660 (both cells remain ✅ within ±5 %).
> The sections below are a chronological log; the 2026-04-20 summary
> counts are historical (superseded by the bottom of the file).

## 2026-04-20 pass

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

---

## 2026-05-03 follow-up — 6 post-audit `verified_pdf` YAMLs

Repo state: 44 reproduction YAMLs, 26 with `source_confidence: verified_pdf`.
The 2026-04-20 audit covered 23 of those; six landed since and need to be
audited against the source PDFs the same way.

| YAML | Model | Dataset | Metric | Claimed | Paper cell | Verified? | Notes |
|---|---|---|---|---|---|---|---|
| da_v2_large_diode | DA-V2 ViT-L | DIODE val (combined) | abs_rel | 0.0533 | MoGe Table 3, affine-invariant disparity, DIODE col, **DA-V2 row = 5.33 → 0.0533** | VERIFIED | Same paper cell as above audit's `moge_vitl_diode_*` lineage; citation correctly says Table 3 (not the recurring Table-2 mistake). YAML observed 0.0529 (1.7 % off). |
| da_v2_large_kitti_moge | DA-V2 ViT-L | KITTI (MoGe-eval bundle) | abs_rel | 0.0561 | MoGe Table 3, affine-invariant disparity, KITTI col, **DA-V2 row = 5.61 → 0.0561** | VERIFIED | Cross-checked against the prior audit's MoGe source-verification entry ("DA-V2 ViT-L NYUv2 4.20, KITTI 5.61, DIODE 5.33"). YAML observed 0.0569 (1.4 % off). |
| moge_vitl_kitti | MoGe-1 ViT-L | KITTI (MoGe-eval bundle) | abs_rel | 0.0408 | MoGe Table 3, affine-invariant disparity, KITTI col, **MoGe row = 4.08 → 0.0408** | VERIFIED | Closes the prior audit's `WRONG_VALUE` entry — earlier YAML pinned 0.0405; current YAML pins 0.0408 with citation correctly pointing at Table 3 (not Table 2). YAML observed 0.0404 (D8 close, 0.9 % off). |
| vggt_dtu_fp32_probe | VGGT (fp32) | DTU MVS | overall | 0.382 | Same target cell as `vggt_paper_dtu_mvs` (VGGT Table 2 Overall 0.382, prior-audit VERIFIED) | VERIFIED (probe inherits) | Diagnostic-only YAML for D3 dtype rule-out; reuses the existing `dtu_vggt_table2` protocol with `dtype: float32`. Result 0.750 mm Overall (within 1 % of bf16 baseline) → fp32 ruled out as a D3 lever. |
| vggt_co3dv2_pose | VGGT | CO3Dv2 (multi-view pose) | pairwise_pose_auc@30 | 0.882 | **VGGT Table 1**, "Camera Pose Estimation on RealEstate10K and CO3Dv2 with 10 random frames", CO3Dv2 AUC@30 col, **"Ours (Feed-Forward)" row = 88.2 → 0.882** (also: Ours-with-BA = 91.8) | VERIFIED 2026-05-03 | First Table 1 cell to enter the audit. WebFetch of `arxiv.org/html/2503.11651` confirmed exact value. Pending GPU run. |
| mast3r_co3dv2_pose | MASt3R | CO3Dv2 (multi-view pose) | pairwise_pose_auc@30 | 0.818 | MASt3R Table 3 (Multi-view pose regression on CO3Dv2 / RealEstate10K, 10 random frames), CO3Dv2 col block, row (b) pairwise, MASt3R: **RRA@15 94.6 / RTA@15 91.9 / mAA(30) 81.8** → AUC@30 0.818 (companions 0.946 / 0.919) | **VERIFIED (direct PDF, 2026-05-23)** | Resolves the prior WEBFETCH_INCOMPLETE. The arXiv HTML render only ever served the appendix; downloaded `arxiv.org/pdf/2406.09756` and read Table 3 (PDF page 10) directly. All three cells match the YAML exactly. Protocol text (§4.3) also confirmed: 41 CO3Dv2 categories, 10 frames/seq, all 45 pairs, no GT focals. Paper cell now confirmed; **GPU run still pending** before the row counts as a reproduction MATCH. |

### Findings out of this pass

1. ~~**`mast3r_co3dv2_pose` is structurally `verified_pdf` in the YAML but
   not WebFetch-confirmable from this session.**~~ **RESOLVED 2026-05-23.**
   The HTML renders only ever served the appendix. Downloaded
   `arxiv.org/pdf/2406.09756` and read Table 3 directly (PDF page 10):
   CO3Dv2 row (b) MASt3R = RRA@15 **94.6** / RTA@15 **91.9** / mAA(30)
   **81.8**. All three match the YAML (0.946 / 0.919 / 0.818) exactly.
   §4.3 protocol text also confirmed (41 categories, 10 frames, 45
   pairs, no GT focals). The cell value is now genuinely `verified_pdf`;
   the GPU run is the only thing left before it counts as a MATCH.

2. **`seven_scenes.py` docstring claim "MASt3R paper has Tables 1-4
   only" is inaccurate.** The same WebFetch passes show Tables 7-8 in
   the appendix, so total tables ≥ 8 (and Table 5/6 likely exist
   between main-body Table 4 and appendix Table 7). The substantive
   claim (MASt3R does not evaluate 7-Scenes for pairwise pose) is
   unaffected — `7-Scenes` / `7Scenes` returned zero hits across all
   WebFetch attempts. The docstring should be corrected to "MASt3R's
   main-body tables go through Table 4; the paper does not evaluate
   7-Scenes for pairwise pose under any table number".

3. **No new systematic errors.** All four mono-depth/MVS YAMLs from
   this batch cite tables correctly (no recurring `WRONG_TABLE` —
   the earlier MoGe Table-2-vs-Table-3 confusion is gone).

### Updated counts (2026-05-03; MASt3R row re-verified 2026-05-23)

- YAMLs with a pinned, paper-comparable value: **29** (was 23).
- VERIFIED: **15** (was 9; +5 from the 2026-05-03 pass, +1 from the
  2026-05-23 direct-PDF read of `mast3r_co3dv2_pose`).
- WEBFETCH_INCOMPLETE: **0** (the lone case, `mast3r_co3dv2_pose`, was
  resolved by direct PDF read 2026-05-23).
- WRONG_TABLE / WRONG_VALUE / WRONG_ROW counts unchanged from 2026-04-20.

---

## 2026-05-23 follow-up — complete `verified_pdf` coverage (direct PDF reads)

Cross-checked the audit against the live repo: enumerated all YAMLs with
`source_confidence: verified_pdf` **and** a non-null pinned value (25 of
them) and confirmed each is backed by a row in this file. Two — both
GeoWizard cells — had never been audited despite being marked
`verified_pdf`. Closed that gap by direct PDF read (the arXiv HTML
render is not needed; `pdftotext`/`pypdf` over the canonical PDF is the
ground truth).

| YAML | Model | Dataset | Metric | Claimed | Paper cell (verified source) | Verified? | Notes |
|---|---|---|---|---|---|---|---|
| geowizard_nyuv2 | GeoWizard | NYU Eigen | abs_rel | 0.052 | GeoWizard Table 1 (Fu et al. 2024, arXiv:2403.12013) "6 zero-shot affine-invariant depth benchmarks", NYUv2 AbsRel col, **"GeoWizard (Ours)" row = 5.2 → 0.052** (δ1 = 96.6) | VERIFIED (direct PDF, 2026-05-23) | First audit of this YAML. PDF page 10. Cell is the **paper target**; the observed value is upstream-blocked (D17), but the target itself is confirmed accurate. |
| geowizard_kitti | GeoWizard | KITTI Eigen | abs_rel | 0.097 | GeoWizard Table 1, KITTI AbsRel col, **"GeoWizard (Ours)" row = 9.7 → 0.097** (δ1 = 92.1) | VERIFIED (direct PDF, 2026-05-23) | First audit of this YAML. Same table/PDF page as above. Paper target confirmed; observed value upstream-blocked (D18/D22). Note Table 1's DIODE column is "DIODE-Full" (29.7), not the indoor/outdoor split used by the MoGe-lineage YAMLs. |
| mast3r_co3dv2_pose | MASt3R | CO3Dv2 pose | pairwise_pose_auc@30 | 0.818 | MASt3R Table 3, CO3Dv2 row (b) MASt3R, mAA(30) = 81.8 → 0.818 (companions RRA@15 94.6, RTA@15 91.9) | VERIFIED (direct PDF, 2026-05-23) | See the 2026-05-03 follow-up table (row updated in place). Resolved D23. |
| cut3r_nyuv2 | CUT3R | NYU-v2 (single-frame) | abs_rel | 0.086 | CUT3R Table 1 (Single-frame Depth Evaluation, Wang et al. 2025, arXiv:2501.12387), NYU-v2 Abs Rel col, **"Ours" row = 0.086** (companion δ<1.25 = 90.9) | VERIFIED (direct PDF, 2026-05-23) | PDF page 5. Protocol: per-frame median scaling per DUSt3R. Value PDF-verified. **Eval-protocol diff now resolved (D24, 2026-05-25): documented PROTOCOL DELTA, not a paper-match** — plumbline's strict raw+crop protocol scores 0.0522, better than paper 0.086, vs the DUSt3R-lineage filled+no-crop re-score 0.0777. Model correct; `paper_match: no` is expected. |
| cut3r_kitti | CUT3R | KITTI (single-frame) | abs_rel | 0.092 | CUT3R Table 1, KITTI Abs Rel col, **"Ours" row = 0.092** (companion δ<1.25 = 91.3) | VERIFIED (direct PDF, 2026-05-23) | PDF page 5, same table/row as cut3r_nyuv2. Value PDF-verified. Eval-protocol diff resolved (D24): documented PROTOCOL DELTA — plumbline's strict Eigen-652+Garg-crop scores 0.0858 vs the lineage val_selection_cropped 0.092. Not a paper-match; model correct. |
| cut3r_bonn | CUT3R | Bonn (VIDEO, per-sequence scale) | abs_rel | 0.078 | CUT3R **Table 2** (Video Depth Evaluation), Per-sequence scale, BONN Abs Rel col, **"Ours" row = 0.078** (companion δ<1.25 = 93.7) | VERIFIED (direct PDF, 2026-05-23) | PDF page 6. First video reproduction (one sample = one sequence). Value PDF-verified. Eval-protocol diff resolved (D24): documented PROTOCOL/SELECTION DELTA — different sequence/frame set than CUT3R's 5-seq × 110-frame Table 2 set, so observed 0.0536 vs paper 0.078 is expected. Not a paper-match; model correct. |

### Verified-coverage status (2026-05-23)

All **28** `verified_pdf` YAMLs that pin a non-null value are now audited
against the source PDF:

- **VERIFIED outright (value + table + col + row match):** 20 —
  da3_nyuv2, da_v2_{small,base,large}_{nyuv2,kitti} (the canonical
  DA-V2 Table 2 cells), da_v2_large_diode, da_v2_large_kitti_moge,
  metric3d_v2_{kitti,nyuv2,giant_kitti,giant_nyuv2}, marigold_v1_1_kitti,
  vggt_paper_dtu_mvs, vggt_dtu_fp32_probe, vggt_eth3d_multiscene_chamfer,
  vggt_co3dv2_pose, mast3r_co3dv2_pose, geowizard_nyuv2, geowizard_kitti,
  cut3r_nyuv2, cut3r_kitti, cut3r_bonn (all three: value verified;
  eval-protocol diff resolved 2026-05-25 as a documented protocol delta —
  D24, not paper-matches).
- **Value VERIFIED, citation table-number corrected in-YAML since the
  2026-04-20 audit** (the old `WRONG_TABLE` rows): da_v2_large_nyuv2,
  marigold_v1_1_nyuv2, moge_vitl_nyuv2, moge_vitl_diode_both,
  moge_vitl_kitti. Each YAML now cites the correct table (MoGe Table 3 /
  Marigold Table 1); the values were already confirmed in 2026-04-20.

No remaining `verified_pdf` value is unaudited, fabricated, or inferred.
Every fabricated/guessed target found in 2026-04-20 (Depth Pro NYU 0.961,
DA-V2 Sintel 0.075, MoGe-2 KITTI 0.0392) is now `value: null` /
informational in its YAML.

## Observed-value provenance (2026-05-28)

Two verification passes on the 22 ✅ cells:

**Target re-verification (paper PDFs):** 21/22 targets confirmed at their
cited table/row/column. **2 one-digit transcription slips corrected**
(both cells remain ✅ within ±5 %):
- `dust3r-co3dv2-pose` mAA(30) **0.774 → 0.772** (MASt3R ECCV Table 3
  DUSt3R row = 94.3 / 88.4 / **77.2**; companions were already correct).
- `dust3r-kitti` companion δ₁ **0.8600 → 0.8660** (DUSt3R Table 2 KITTI;
  ar5iv re-read).

**Observed-vs-retained-JSON provenance:**
- **Byte-exact vs a retained result JSON (5):** vggt/mast3r/dust3r
  CO3Dv2 pose, dust3r-kitti, monst3r-sintel-pose — JSONs on
  `s3://plumbline-bench/runs/` (3 also in `docs/runs/`).
- **Archive-confirmed (~10):** the early mono-depth ✅ cells match
  `docs/runs/archive/20260421.md` within rounding.
- **No retained JSON yet (5):** monst3r-nyuv2, da-v2-large-diode,
  da-v2-large-kitti-moge, moge-vitl-kitti, moge-vitl-diode-both — run in
  earlier sessions; **queued for a JSON-capturing re-run**
  (`gpu_queue.yaml`, priority 3). After those, every ✅ is byte-verified.

**Habit going forward:** every reproduction run syncs its result JSON to
`s3://plumbline-bench/runs/` so coverage and provenance grow together.
