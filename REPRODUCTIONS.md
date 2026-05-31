# Reproductions

A **reproduction** is a pinned config that re-produces a specific number from
a specific paper, within a documented relative tolerance. Running it is the
harness's acceptance test.

> **Note (2026-05-03):** status matrix below reflects all GPU runs
> through 2026-04-27. Open discrepancies and next-session priorities
> are in `docs/DISCREPANCIES.md`. Per-YAML paper-citation audit (now
> 32 paper-pinned YAMLs, 32 verified as of 2026-05-30; was 15 at the
> 2026-05-23 MASt3R direct-PDF read) is in
> [`reproductions/AUDIT.md`](./reproductions/AUDIT.md).

## Status matrix (2026-05-03)

Model × dataset cell statuses:

**Legend:** ✅ MATCH within tolerance against a **verified_pdf**
paper target · ⚠️ observed, off paper · 🎯 observed, paper target
unconfirmed · ⌛ infra ready, awaiting data/compute · 🚧 planned
(loader/adapter not yet wired) · ℹ️ informational (no paper target)
· — not a canonical paper combo

Only cells where `paper_reference.source_confidence == verified_pdf`
count as ✅. The 2026-04-20 audit
([reproductions/AUDIT.md](./reproductions/AUDIT.md)) removed four
claims that couldn't be verified against the source PDFs; affected
cells are now ℹ️ instead of ✅.

| Model → Dataset | NYUv2 | KITTI | DIODE | ETH3D | DTU | Co3Dv2 | GSO |
|---|---|---|---|---|---|---|---|
| **DA-V2 Small** | ✅ **0.0510** vs 0.053 | ✅ **0.0770** vs 0.078 | ℹ️ **0.0722** _(no ViT-S paper cell under this protocol)_ | — | — | — | ⌛ |
| **DA-V2 Base** | ℹ️ **0.0456** vs 0.049 _(6.9% off — reproduction delta, exceeds ±5%; see yaml note)_ | ✅ **0.0756** vs 0.078 | — | — | — | — | — |
| **DA-V2 Large** | ✅ **0.0428** vs 0.0420 | ✅ **0.0710** vs 0.074 | ✅ **0.0529** vs 0.0533 _(landed 2026-04-27 via DIODE FoV-warp loader)_ | ✅ **0.0473** vs 0.0473 _(MoGe-eval mono-depth, scale_shift_clamped; 2026-05-30)_ | — | — | ✅ **0.0125** vs 0.0125 (δ₁ 0.9999) _(MoGe Table 3 GSO; 2026-05-30)_ · ✅ **0.0348** vs 0.0348 _(MoGe Table 3 iBims-1; 2026-05-30)_ |
| **DA-V2 Metric-Outdoor-L** | — | ℹ️ **0.0877** _(VKITTI-finetuned; no direct paper)_ | — | — | — | — | — |
| **Metric3D-v2 L** | ✅ **0.0660** vs 0.063 | ✅ **0.0495** vs 0.052 | — | — | — | — | — |
| **Metric3D-v2 Giant** | ✅ **0.0702** vs 0.067 | ✅ **0.0503** vs 0.051 | — | — | — | — | — |
| **DA3** | ✅ δ₁ **0.9684** vs 0.974 | — | — | ⚠️ chamfer 7.14 (protocol gap) | — | ⌛ informational only (paper does not evaluate CO3Dv2 pose; A/B vs VGGT/MASt3R) | 🎯 **0.0150** (δ₁ 0.9994) _(no paper target)_ |
| **MoGe-1 ViT-L** | ✅ **0.0342** vs 0.0341 | ⚠️ **0.0447** vs 0.0408 _(9.4% off; D8 structural protocol)_ | ✅ **0.0407** vs 0.0400 _(1.7% off; FoV-warp port 2026-04-26)_ | ✅ **0.0311** vs 0.0317 _(MoGe-eval mono-depth, scale_shift_clamped; 2026-05-30)_ | — | — | ✅ **0.00958** vs 0.00944 (δ₁ 0.9999) _(MoGe Table 3 GSO; 2026-05-30)_ |
| **MoGe-2 ViT-L** | ✅ **0.0305** (scale+shift) | ℹ️ _(paper publishes ViT-L only as 10-dataset avg)_ | ⌛ | — | — | — | ⌛ |
| **MoGe-2 metric** | ⌛ 0.0899 informational | — | — | — | — | — | — |
| **Marigold v1-1** | ✅ **0.0577** vs 0.055 | ℹ️ **0.1090** vs 0.099 _(v1-1 / 1-step is the newer distilled checkpoint; paper cell is v1-0 / 50-step — D9 RESOLVED 2026-05-25: 0.0992 reproduces end-to-end on Marigold's own pipeline. Documented checkpoint-generation delta, not a paper-match cell.)_ | — | — | — | — | — |
| **GeoWizard** | ℹ️ **0.0574** vs 0.052 _(10.5% off — D17 RESOLVED 2026-05-26: paper number is best-of-N seeds, not single-seed; plumbline's 0.0574 matches `fuxiao0719/GeoWizard#36` reproducer @0.0576; paper-author-confirmed cherry-pick eval recipe)_ | ℹ️ **0.131** vs 0.097 _(35.2% off — D18 RESOLVED 2026-05-26 by same root cause)_ | — | — | — | — | — |
| **Depth Pro** | ℹ️ δ₁ **0.9347** _(paper does not evaluate NYU — earlier 0.961 pin was fabricated)_ | ℹ️ Sintel δ₁ **0.2409** / **0.400** _(blocked, Table 16 protocol)_ | ✅ Booster δ₁ **0.4878** vs **0.466** _(Table 1, 2026-05-31)_ | — | — | — | ℹ️ iBims δ₁ **0.8458** _(informational, 2026-05-31)_ |
| **MASt3R** (N-view post-2026-04-27) | — | — | — | 2-view pose sweep | — | ✅ AUC@30 **0.7960** vs 0.818 _(2.7 % off, verified 2026-05-26 on RTX 3090; companion RRA@15 = 0.9708; dust3r PointCloudOptimizer N=10, init=mst, niter=300, curope CUDA ext built)_ | — |
| **VGGT** | — | — | — | ⚠️ 0.875 m vs 0.709 _(D10 13-scene investigated 2026-05-27, +23.5 % over paper; one outlier `terrains` Comp 10.18 m drives the aggregate — without it 12-scene mean is 0.515, 27 % tighter than paper. See docs/DISCREPANCIES.md D10.)_ | ⚠️ 0.756 m vs 0.382 mm _(D3 upstream-blocked: PatchmatchNet filter + fp32 verified no-op, residual ~2× is in public VGGT-1B output)_ | ✅ AUC@30 **0.8964** vs 0.882 _(1.6 % over, verified 2026-05-26 on RTX 3090; CO3Dv2 staged via scripts/co3dv2_prefetch.py at ~3 GB)_ | — |
| **CUT3R** _(video + unordered)_ | ℹ️ **0.0522** vs 0.086 _(better — D24 protocol delta: strict raw+crop vs lineage filled+no-crop; model correct, not a paper-match)_ | ℹ️ **0.0858** vs 0.092 _(better — D24 protocol delta: Eigen-652+Garg vs lineage val_selection_cropped)_ | — | — | — | ℹ️ recurrent/online — handles ordered video & unordered sets | — |
| **MonST3R** _(dynamic / video, base path)_ | ✅ **0.0896** vs 0.091 _(1.5% off, Table 3 single-frame, `nyu_dust3r_lineage` protocol; verified 2026-05-26, adapter v1.1)_ | ℹ️ **0.0959** vs 0.101 _(5.05% off — just over ±5% tol, Table 3 single-frame, `kitti_dust3r_lineage` protocol — 1269-frame gathered set; verified 2026-05-26, adapter v1.1)_ | — | — | — | — | — |
| **DUSt3R** _(origin paper, single-frame F(I,I))_ | ℹ️ **0.0777** vs 0.0650 _(D28 — GT-processing delta, NOT alignment: re-scoring the same preds, paper 0.065 is **bracketed** by Eigen-crop 0.0489 ↔ lineage 0.0777; alignment sweep ruled out (median already best); 2026-05-28)_ | ✅ **0.1049** vs 0.1074 _(2.3% off; Table 2 KITTI, view-duplicate adapter v1.1, `kitti_dust3r_lineage` protocol; full 1269-frame / 13-drive gathered set; companion δ₁ 0.8661 vs 0.8660 (≈exact; paper δ₁ corrected 0.8600→0.8660 per PDF re-verify 2026-05-28); 2026-05-28)_ | — | — | — | ✅ AUC@30 **0.7893** vs 0.772 _(2.2 % over; MASt3R Table 3 DUSt3R row; target corrected 0.774→0.772 per PDF re-verify 2026-05-28)_ | — |

Sintel + Bonn lineage cells (also MonST3R Table 3 single-frame) land as **ℹ️ informational** — structurally faithful but off-paper >5 %:
- **MonST3R-Sintel** (`sintel_dust3r_lineage`, 14 dynamic-scene clips, max_depth=70 sky-mask, per-frame median): ℹ️ AbsRel **0.3726** vs paper 0.345 _(8.0 % off, worse; companion δ₁ 0.567 vs 0.565 within 0.4 %)_. Two compounding causes (per D27): (a) `temple_2` outlier (mean 0.93, max 7.87) — view-duplicate single-frame fragility on textureless / heavily-occluded synthetic scenes (without `temple_2` aggregate is ~0.32, within tolerance); (b) upstream `depth_metric.ipynb` Sintel cell uses **per-sequence scale+shift LAD2 + `post_clip_max=70` + valid-pixel-weighted mean across 14 seqs** — outlier frames hurt plumbline's equal-frame mean more than the paper's pixel-weighted-per-seq mean. Both deferred (model-side fragility + protocol delta).
- **MonST3R-Bonn** (`bonn_dust3r_lineage_single`, 5 sequences × all RGB frames, per-frame median, max_depth=70): ℹ️ AbsRel **0.0654** vs paper 0.076 _(14.0 % off, better; companion δ₁ 0.960 vs 0.939, also off, better)_ — **D27 RESOLVED 2026-05-26**: single-record diff against upstream `depth_metric.ipynb` shows the paper's 0.076 is produced by **per-sequence scale+shift LAD2** alignment (`depth_evaluation(..., max_depth=70, align_with_lad2=True)`, weighted-mean across 5 seqs), not the paper §4.2 text's claimed "per-frame median scaling". plumbline is paper-text-faithful; paper number reflects upstream code recipe. Same shape as D9 / D17 / D24 (paper-private eval recipes). Re-scoring on MonST3R's exact `rgb_110/depth_110` subset already ruled out frame-subset (0.0635). See `docs/DISCREPANCIES.md` D27.
- **DUSt3R-Bonn** (`bonn_dust3r_lineage_single`, same protocol as above): ℹ️ AbsRel **0.1337** vs paper **0.0808** _(65.4 % off worse; per-seq split: balloon2 0.0785, person_tracking2 0.0455 — both at/under paper; crowd2 0.184, crowd3 0.152, synchronous 0.182 — the three dynamic-content seqs drive the aggregate up). **D28 finding 2026-05-28**: DUSt3R is not a dynamic-scene model (that's the premise of MonST3R), and per-frame median-scale-only cannot compensate for systematic dynamic-region depth errors the way a per-seq scale+shift LAD2 recipe (D27 pattern, what MonST3R's notebook uses) does. Same model + recipe family as the (matching) MonST3R-Bonn cell; the residual delta is the recipe choice, not the model. The paper's 0.0808 is paper-private — §4.3 doesn't specify the indoor scoring recipe. See `docs/DISCREPANCIES.md` D28.

**DA-V2 Table 2 native benchmarks (parked 2026-05-30, D31/D32/D33)** — harness runs complete;
**do not promote to ✅** (reads much better than paper; likely upstream eval recipe, not adapter bug).
MoGe-bundle Table-3 cells on the same datasets **still ✅** (different preprocessing).

| Reproduction | Paper AbsRel | Observed | Δ vs paper | Handoff |
|---|---|---|---|---|
| `da-v2-small-eth3d-native` | 0.142 | **0.1012** | −29 % | [ETH3D](docs/ETH3D_DAV2_TABLE2_HANDOFF.md) |
| `da-v2-base-eth3d-native` | 0.137 | **0.0936** | −32 % | [ETH3D](docs/ETH3D_DAV2_TABLE2_HANDOFF.md) |
| `da-v2-large-eth3d-native` | 0.131 | **0.0888** (all `dslr_scan_eval`: **0.0782**) | −32 % / −40 % | [ETH3D](docs/ETH3D_DAV2_TABLE2_HANDOFF.md) |
| `depth-anything-v2-sintel` | 0.487 | **0.2321** (`clean` pass **0.2224**) | −52 % | [Sintel](docs/SINTEL_DAV2_TABLE2_HANDOFF.md) |
| `da-v2-small-diode-native` | 0.073 | **0.2196** | +201 % | [D29](docs/D29_DIODE_TABLE2_HANDOFF.md) — native outdoor |
| `da-v2-base-diode-native` | 0.068 | **0.2182** | +221 % | D29 |
| `da-v2-large-diode-native` | 0.066 | **0.2142** | +225 % | D29 |
| `da-v2-*-diode-moge-bundle` (exp.) | 0.073 / 0.068 / 0.066 | **0.062 / 0.059 / 0.054** | −13–18 % | D29 — MoGe GT/mask, still under paper |

**Informational Table-2 crosswalk on MoGe bundles** (2026-05-31, ViT-L, `scale_shift_clamped` except DIODE bundle uses Table-2 `scale_shift` per `diode_dav2_moge_bundle`):

| Reproduction | Paper T2 | Observed | Notes |
|---|---|---|---|
| `da-v2-large-eth3d-moge-table2` | 0.131 | **0.0473** | plain `scale_shift` → AbsRel ~1200 (invalid) |
| `da-v2-large-sintel-moge-table2` | 0.487 | **0.2139** | same as Table-3 MoGe cell |
| `da-v2-large-diode-moge-bundle` | 0.066 | **0.0543** | `scale_shift` on bundle |

Compare: `da-v2-large-eth3d-moge` **0.0473** ✅ · `da-v2-large-sintel-moge` **0.2139** ✅ ·
`da-v2-large-diode` **0.0529** ✅ (MoGe Table 3, `scale_shift_clamped`).

**MoGe upstream harness** (2026-05-30, ViT-L `rel`): DIODE **0.0529**, ETH3D **0.0471**,
Sintel **0.2138** — matches plumbline MoGe-bundle ✅ cells; still ≠ Table 2 paper.
See [`docs/DA_V2_TABLE2_UPSTREAM_EVAL.md`](docs/DA_V2_TABLE2_UPSTREAM_EVAL.md),
`scripts/run-moge-upstream-dav2.sh`. Informational crosswalk repros (Table 2 targets on
MoGe bundles + `scale_shift`): `da-v2-large-{eth3d,sintel}-moge-table2`,
`da-v2-large-diode-moge-bundle`.

**Adapter v1.0 → v1.1 (2026-05-26, eval-mono-depth-avg null-result):** the suspected single-frame fix — averaging the two symmetric pair predictions (`pred1.pts3d.mean(dim=0)`, MonST3R `eval_mono_depth` shape) instead of routing through the MASt3R-shared PairViewer — was implemented and re-run across all four cells. **Result: all four cells moved by <0.005 AbsRel**, ruling out avg-pred as the cause of the Sintel/Bonn deltas. The v1.1 path is still preserved (it matches MonST3R's upstream eval code verbatim, making the adapter strictly more faithful). The Bonn delta itself was closed shortly after by **D27 (2026-05-26)** via a single-record code-level diff against upstream `depth_metric.ipynb`: paper §4.2 text says "per-frame median scaling" but the actual notebook scores via per-sequence scale+shift LAD2 (`align_with_lad2=True`, valid-pixel-weighted across 5 seqs) — paper-text-vs-code mismatch, not a plumbline bug. Same finding also explains the Sintel direction (per-seq pixel-weighted aggregation dilutes the `temple_2` outlier the equal-frame plumbline mean amplifies). See `docs/DISCREPANCIES.md` D27.

**Video benchmark (new 2026-05-23):** the **Bonn RGB-D Dynamic** loader
(`bonn`, one-sample-per-sequence) closes the runnable-video gap.
`cut3r-bonn` targets CUT3R Table 2 video-depth (per-sequence scale)
Bonn AbsRel **0.078** (verified_pdf); observed **0.0536** — a documented
protocol/selection delta (D24, resolved 2026-05-25): a different
sequence/frame set than CUT3R's 5-seq × 110-frame Table 2 set, so it is
not a paper-match. MonST3R also reports Bonn (0.067 w/ flow GA); a
faithful MonST3R-video cell awaits the flow-path follow-up.

### Paper-match count

**28 ✅ mono-depth cells + 4 ✅ pose cells = 32 total** with `source_confidence: verified_pdf`:

- NYU (8): DA-V2 S/L, Metric3D-v2 L/Giant, MoGe-1 ViT-L, Marigold, DA3, **MonST3R** (lineage protocol, 2026-05-26)
- KITTI Eigen+Garg (5): DA-V2 S/B/L, Metric3D-v2 L/Giant
- KITTI MoGe-eval (2): MoGe-1 ViT-L (D8 close), DA-V2 ViT-L (2026-04-27)
- KITTI dust3r-lineage (1): **DUSt3R** (0.1049/0.1074, −2.3%) — was present in the grid + site but omitted from this breakdown
- DIODE (2): MoGe-1 ViT-L, DA-V2 ViT-L (FoV-warp loader, 2026-04-26/27)
- **GSO (2, NEW 2026-05-30): MoGe-1 ViT-L (0.00958 vs 0.00944), DA-V2 ViT-L (0.01247 vs 0.0125)** — MoGe Table 3 GSO column
- **iBims-1 (2, NEW 2026-05-30): MoGe-1 ViT-L (0.0316 vs 0.0320), DA-V2-L (0.0348 vs 0.0348)** — MoGe Table 3; DA-V2 re-eval uses `scale_shift_clamped` (D30; prior 0.0391 under plain `scale_shift`)
- **ETH3D MoGe-eval mono-depth (2, NEW 2026-05-30): MoGe-1 ViT-L (0.0311 vs 0.0317), DA-V2 ViT-L (0.0473 vs 0.0473)** — `eth3d-moge-eval` loader + `scale_shift_clamped` (D30)
- **Tier A MoGe Table 3 (4, NEW 2026-05-30): DDAD** — MoGe-1 (0.0902 vs 0.0891), DA-V2-L (0.1310 vs 0.1300); **Sintel MoGe bundle** — MoGe-1 (0.1863 vs 0.1840), DA-V2-L (0.2139 vs 0.2140)
- _(2026-05-28 confidence audit: DA-V2 Base NYU (6.9% off) and MonST3R KITTI (5.05% off) downgraded ✅→ℹ️. Counts match site/README as of 2026-05-30.)_
- **CO3Dv2 pose (3): VGGT (AUC@30 0.8964 vs 0.882, +1.6 %) + MASt3R (mAA(30) 0.7960 vs 0.818, −2.7 %) + DUSt3R (mAA(30) 0.7893 vs 0.772, +2.2 %), all on MASt3R Table 3 protocol** — v0.1 acceptance criterion #2 met (VGGT) and twice-seconded (MASt3R / DUSt3R).
- **Sintel trajectory pose (1): MonST3R Table 4 — ATE 0.1134 vs 0.108 (+5.0 %), plumbline-computed 2026-05-27** via the new adapter v1.2 video-pose path + `metrics/pose.py` ATE/RPE family.

Each cell is verified against the source PDF (table + col + row) per
`reproductions/AUDIT.md`.

**Multi-view ✅ cells**: 0 (paper-match). Two structurally-correct
reproductions on `main`:
- VGGT-DTU (D3) — protocol port complete; ~2 × residual gap declared
  upstream-blocked 2026-04-27 (PatchmatchNet filter + fp32 + Jensen
  toolkit + 49-view all verified ~no-ops).
- VGGT-ETH3D (D4 / D10) — full 13-scene run lands Overall 0.875 m
  vs paper 0.709 (+23.5 %, MISMATCH). The aggregate is dominated by
  one outlier scene (`terrains` Comp 10.18 m); excluding it, the
  other 12 scenes mean 0.515 m (27 % tighter than paper). D10 closed
  as ⚠️ investigated with a localised follow-up. The 3-scene subset
  variant is preserved at `vggt_eth3d_subset_chamfer.yaml` for
  regression detection.

**Pose ✅ cells (plumbline matrix)**: **4** — every number below was computed by plumbline's runner, loader, adapter, and metric code.

- ✅ **VGGT CO3Dv2** (Table 1, AUC@30): observed **0.8964** vs paper **0.882** (+1.6 % over). Companion **pairwise_RRA@15 = 0.9819**. Run wall ~28 min; feed-forward, no global alignment.
- ✅ **MASt3R CO3Dv2** (Table 3, mAA(30)): observed **0.7960** vs paper **0.818** (−2.7 % under). Companion **pairwise_RRA@15 = 0.9708**. Run wall ~3.2 h cumulative (interrupted once by an SSH-daemon drop on the vast.ai box; resumed via plumbline's prediction cache, 143 of 410 samples already computed). Paper cell PDF-verified 2026-05-23 (D23 resolved).
- ✅ **DUSt3R CO3Dv2** (MASt3R Table 3 DUSt3R row, mAA(30)): observed **0.7893** vs paper **0.772** (+2.2 % over; target corrected 0.774→0.772 per PDF re-verify 2026-05-28). Companion **RRA@15 = 0.9722** vs paper 0.943 (+3.1 %), **RTA@15 = 0.8801** vs paper 0.884 (−0.4 %). All three within ±5 %. Run wall ~2.9 h (399 of 399 samples; PCO ~25 s/sample on a 3090). Adapter v1.0 (new this PR) wraps `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` via the shared `_run_mast3r` helper (added `scene_graph` kwarg, hardcoded `"complete"` previously). Same MASt3R Table 3 row as the MASt3R cell above — paper-PDF-verified 2026-05-27 with the same direct-read confidence as the MASt3R row.
- ✅ **MonST3R Sintel trajectory pose** (Table 4): observed **ATE 0.1134** vs paper **0.108** (+5.0 % over). Companion **RPE-trans = 0.0446** vs paper 0.042 (+6.3 %), **RPE-rot = 0.792°** vs paper 0.732° (+8.2 %). All three within ±10 %. Run wall ~42 min (14 dynamic Sintel-final clips). Adapter v1.2 video-pose path: builds the `swinstride-5-noncyclic` pair graph, drives MonST3R's extended `global_aligner` with `flow_loss_weight=0.01` (RAFT-Sintel checkpoint), `motion_mask_thre=0.35`, `temporal_smoothing_weight=0.01`, `use_self_mask=True`, then `compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)` — i.e., the same apparatus as MonST3R's `launch.py --mode=eval_pose --eval_dataset=sintel`. ATE/RPE math via `metrics/pose.py:trajectory_{ate,rpe}_rmse_sim3`, which wrap `evo.main_ape.ape` + `main_rpe.rpe` with `align=True, correct_scale=True` — bit-identical to MonST3R's `vo_eval.py:eval_metrics`. End-to-end agreement: plumbline 0.11340 vs MonST3R-own-pipeline 0.1134 (paper-trust signal from PR #14) — same to 4 decimals.

The first three run the same `Co3Dv2VGGTPoseEvalLoader` recipe (41 SEEN cats × 10 seq × 10 frame, seed=0), `vggt_co3d_histogram` AUC mode, `pose_translation_antipodal: true`. CO3Dv2 staged once via `scripts/co3dv2_prefetch.py` (~3 GB selective fetch, ~30 min), then jobs ran back-to-back on the same disk. MASt3R / DUSt3R inference goes through dust3r's `PointCloudOptimizer` (N≥3, init=mst, niter=300, scene_graph=complete); curope CUDA ext built per-MASt3R-dust3r-fork in-session (32 % speedup vs the pytorch RoPE2D fallback the adapter ships with). DA3 has an informational companion (`da3_co3dv2_pose.yaml`) with no paper target.

**Paper-trust verifications via the model's own pipeline (NOT plumbline-computed)**: these confirm the paper number is reproducible end-to-end on the author's released code, separate from whether plumbline's own stack reaches the same number. Same shape as D9 / D24. Each entry is a paper-trust signal, not a plumbline ✓ matrix cell.

- **Marigold NYU + KITTI** (D9, 2026-05-25): NYU 0.0577 / paper 0.055; KITTI 0.0992 / paper 0.099 with v1-0 / 50-step / ens-10 on Marigold's prepared eval set + `script/depth/eval.py`. See `docs/DISCREPANCIES.md` D9.
- **CUT3R NYU + KITTI + Bonn** (D24, 2026-05-25): NYU 0.08595 / paper 0.086; KITTI 0.09219 / 0.092 (Table 1); Bonn 0.07661 / 0.078 (Table 2, per-seq scale). Via CUT3R's `eval/monodepth` + `eval/video_depth --align scale`. See `docs/DISCREPANCIES.md` D24.
- **MonST3R Sintel pose, own-pipeline cross-check** (2026-05-27, via `launch.py --mode=eval_pose --eval_dataset=sintel`, 14 dynamic clips): ATE 0.1134 / paper 0.108 (+5.0 %); RPE-trans 0.0446 / 0.042 (+6.3 %); RPE-rot 0.7921 / 0.732 (+8.2 %). Plumbline's adapter v1.2 video-pose path now lands **0.11340** end-to-end on the same Sintel set (above) — same to 4 decimals, confirming the two pipelines drive identical compute and only differ in trajectory I/O. This entry is preserved as historical context for the adapter port; the matrix cell is the plumbline ✓ entry, not this one.

**Plumbline-computed trajectory pose (new 2026-05-27)**: ATE / RPE-trans / RPE-rot now land in `src/plumbline/metrics/pose.py` (`trajectory_ate_rmse_sim3`, `trajectory_rpe_rmse_sim3`) by wrapping `evo`'s `main_ape.ape` + `main_rpe.rpe` — bit-identical to MonST3R's own `vo_eval.py:eval_metrics`. Runner enables them via `pose_trajectory_metrics: true` on the YAML; Sintel loader gained a `full_seq` mode (one Sample per scene). MonST3R adapter v1.2 added the `video_pose=True` path that wires the full Table 4 apparatus (`_run_monst3r_video_pose` in `src/plumbline/models/monst3r.py`) — `swinstride-5-noncyclic` pair graph + flow / motion / temporal terms in the global alignment.

**CO3Dv2 disk gate cleared 2026-05-26 (selective fetch):** the raw
CO3Dv2 distribution is ~4.3 TB (276 zips × ~18 GB avg) — well past the
200 GB vast.ai box's budget and the historical block on these jobs. The
VGGT/MASt3R eval protocol only needs 4 100 JPEGs (41 cats × 10 seq × 10
frame) plus per-category metadata, so `scripts/co3dv2_prefetch.py` does
a surgical HTTP-Range fetch: download each `{cat}_000.zip` metadata
chunk (~30-90 MB), replicate the loader's `seed=0` selection algorithm
to enumerate the exact JPEGs needed, then `zipfile.ZipFile` over a
custom `RangeHTTPFile` reads each individual JPEG's local file header +
compressed bytes out of the per-category big chunks. End-to-end verified
on `apple` (100 JPEGs, 76 MB, loader produces correct 10×10 Samples).
Full-set estimate: **~3-4 GB on disk, ~60 min one-time fetch**. The
three Tier-1 jobs (`vggt-co3dv2-pose`, `mast3r-co3dv2-pose`,
`da3-co3dv2-pose`) all share the staged set and now have
`data_footprint_gb: 4` in `gpu_queue.yaml`.

**Off-paper / upstream-blocked cells** (each root-caused in
`docs/DISCREPANCIES.md`):

- VGGT-DTU (D3), GeoWizard NYU (D17), GeoWizard KITTI (D18) —
  **upstream-blocked**. The adapter and protocol audits are
  exhausted; residual gap is in the public checkpoint or a
  paper-private eval config. Cells stay as ⚠️ in the matrix; the
  YAMLs ship on `main` because the protocol shape is correct.
- Marigold-KITTI (D9 / D22 Marigold portion) — ✅ **RESOLVED 2026-05-25**
  by end-to-end native-pipeline reproduction. Paper cell 0.099 is
  reproducible with v1-0 / 50-step / ens-10 on Marigold's exact
  prepared `kitti_eigen_split_test.tar` (0.0992, 0.2 % off).
  Plumbline's `marigold_v1_1_kitti.yaml` lands ~0.11 because it
  mirrors the current upstream eval script which defaults to the
  newer **v1-1 / 1-step** distilled checkpoint — a documented
  checkpoint-generation delta (v1-1 still matches paper on NYU).

**Dropped from the ✅ count (2026-04-20 audit):**

- Depth Pro NYU (δ₁ 0.961) — paper has no NYU row; target was fabricated.
- MoGe-2 KITTI — paper has no per-dataset ViT-L row.
- DA-V2-Small DIODE-indoor — cited cell is for ViT-L, not ViT-S.
- DA-V2 Sintel — cited target (0.075) does not appear in the paper.
- (2026-04-27) "MASt3R Table 5 on 7-Scenes" claim in `seven_scenes.py`
  docstring — MASt3R does not evaluate 7-Scenes for pairwise pose.

### Source-fidelity audit (2026-05-23)

`docs/SOURCE_AUDIT.md` audits every adapter against its released upstream
source. Most are faithful; fixes landed for DA-3 (extrinsics shape +
relative-depth flag) and π³ (confidence shape). One **deferred** item
touches verified cells: the DA-V2 *paper* path passes
`image_interpolation_method=3` (`cv2.INTER_AREA`) where upstream uses
`cv2.INTER_CUBIC` (=2). The 8 ✅ DA-V2 cells were validated with INTER_AREA,
so switching to the faithful INTER_CUBIC needs a GPU re-validation of those
cells before it lands (behavior left unchanged for now).

### Biggest open gaps (in order of per-cell leverage)

1. ~~**CO3Dv2 GPU run**~~ — ✅ closed 2026-05-26 (VGGT 0.8964, MASt3R
   0.7960) + extended 2026-05-27 with DUSt3R (0.7893). Three CO3Dv2
   pose ✅ cells live on the same `vggt_co3d_histogram` AUC mode.
2. ~~**D10 · VGGT-ETH3D 13-scene full split**~~ — 🔬 investigated
   end-to-end 2026-05-27. Full 13-scene run lands Overall 0.875 vs
   paper 0.709 (+23.5 %, MISMATCH). One scene (`terrains`) drives
   the entire aggregate gap: Comp 10.18 m vs ~0.5 m on the other 12
   scenes. Excluding terrains, 12-scene mean Overall 0.515 vs paper
   0.709 (plumbline 27 % tighter). Stays ⚠️ off-paper with a now-
   localised follow-up (ICP-per-window diagnostic on terrains). See
   `docs/DISCREPANCIES.md` D10.
3. ~~**D23 · MASt3R Table 3 PDF re-verification**~~ — ✅ done
   2026-05-23. Direct PDF read of `arxiv.org/pdf/2406.09756` Table 3
   confirmed CO3Dv2 row (b) MASt3R = 94.6 / 91.9 / 81.8, matching
   `mast3r_co3dv2_pose.yaml` exactly. Only the GPU run remains.

### Closed-blocked (do not retry without an upstream change)

D3 (VGGT-DTU), D17 / D18 (GeoWizard NYU + KITTI). All
three hit the same wall: adapter + protocol audits exhausted, residual
gap is in the public release. Re-enter the queue if/when upstream
releases an updated checkpoint or eval script.

### Deprioritized (2026-04-19 pivot)

Loaders exist and are unit-tested but data remains auth-gated:

- **Sintel depth** → substituted by **GSO** / **iBims-1** (synthetic clean-GT slot).
- **ScanNet-1500 pose** → substituted by **Co3Dv2** / **7Scenes** (pose paper rows).

---

All matches were produced on a single RTX 3090 Ti inside ~3 hours of
cumulative wall-clock time (first-run weight downloads dominated).
See the per-reproduction status table below for citations, observed
values, and notes.

## Running

Set the appropriate dataset-root env var first; YAML files deliberately
don't hardcode machine-specific paths:

```bash
export NYUV2_ROOT=~/data/nyuv2      # for any da-v2-*-nyuv2 reproduction
export SCANNET_ROOT=~/data/scannet  # for vggt-paper-scannet-depth
export SINTEL_ROOT=~/data/sintel    # for depth-anything-v2-sintel
export KITTI_ROOT=~/data/kitti      # for any *-kitti reproduction
export DIODE_ROOT=~/data/diode      # for any *-diode-* reproduction
export DTU_ROOT=~/data/dtu          # for vggt-paper-dtu-mvs

plumbline reproduce <name>
```

This loads `reproductions/<name>.yaml`, runs the model on the dataset,
computes metrics, and compares the primary metric against the published
value.

## Per-YAML detail

Per-YAML observed values, paper citations, and audit status live in
[`reproductions/AUDIT.md`](./reproductions/AUDIT.md). Each YAML's
own `notes:` field carries run-specific detail (RTX-3090 wall-clock,
δ₁/RMSE companions, alignment mode).

### Note on the NYUv2 Eigen 2014 protocol

Paper matches required three loader/runner details that weren't obvious
from reading the paper itself:

1. **Depth field: `rawDepths`, not `depths`.** NYU's .mat ships both the
   sparse Kinect measurements (`rawDepths`, ~24% holes) and Silberman's
   colorization-filled version (`depths`, dense). Every modern mono-depth
   paper that cites "NYU Eigen" evaluates against `rawDepths`;
   `NYUv2Dataset(depth_field="raw")` is the default.
2. **`depth_clip: [0.001, 10.0]` post-alignment.** Scale+shift alignment
   occasionally produces extreme per-sample predictions (on DA-V2 Large
   sample 88, an aligned value hit 1e8 m). Paper eval clips the aligned
   prediction to the same range as the valid GT mask. Reproduction
   YAMLs set this explicitly.
3. **`gt ∈ [1e-3, 10]m` valid mask.** Standard NYU convention; plumbline
   already applies this via the loader's Eigen crop + positivity mask.

A 2026-04-18 diagnostic confirmed the author's own `run.py` on the
HuggingFace ViT-S checkpoint produces AbsRel=0.0621 against the *filled*
`depths` field — within 0.3% of what plumbline produced before the raw-
default landed. Switching to rawDepths drops that to 0.0510 (vs paper
0.053), and the same switch takes ViT-L from 0.0554 to 0.0428 (vs paper
0.045). Without the clip, ViT-L averaged 77.9 because of sample 88 alone.

### Note on the KITTI Eigen protocol

Three details tend to separate "just ran the HF model on KITTI" numbers
from the paper targets:

1. **Annotated-depth GT, not raw LiDAR projections.** The original Eigen
   2014 protocol reprojects Velodyne points into camera frame, yielding
   sparse and noisy GT. Modern papers (DA-V2, Metric3Dv2, DA3, MoGe,
   Depth Pro) evaluate against the KITTI Depth-Prediction Benchmark's
   *annotated* dense depth maps (Uhrig et al. 2017, ~14 GB public
   archive). plumbline's `KITTIDataset` loads the annotated maps.
2. **Garg crop on evaluation.** Pixels outside
   `row ∈ [0.408 H, 0.992 H) × col ∈ [0.036 W, 0.964 W)` are excluded.
   Pass `apply_garg_crop: true` in the dataset kwargs; the loader
   populates `Sample.depth_valid` with the crop AND-ed with `depth > 0`.
   Without the crop, hood-of-car pixels and image borders dominate the
   metric.
3. **`depth_clip: [1e-3, 80.0]` post-alignment.** Standard KITTI cap
   (80 m). Apply it the same way NYU's `[1e-3, 10.0]` clip is applied.

KITTI sample-list variants (697 raw / 652 with-GT / 500 improved)
differ by paper. Plumbline bundles the **652-frame with-GT list**
(Monodepth2's `splits/eigen_benchmark/test_files.txt`) at
`reproductions/kitti_eigen_benchmark_652.txt`. Every KITTI
reproduction in this repo resolves `sample_list` from that in-repo
file, so two hosts with different copies of KITTI on disk evaluate
the exact same 652 frames. The loader falls back to
`$KITTI_ROOT/<name>` if the sample_list filename isn't in-repo, which
preserves the pre-2026-04 behavior.

**Disk footprint — the 652-frame list spans 28 raw drives with
12–25 frames per drive** (mode 23–24). Each full
`2011_XX_XX_drive_XXXX_sync` archive contains thousands of frames
but the benchmark only evaluates the listed ~24 per drive, so the
raw archives are aggressively prunable if `$KITTI_ROOT` runs tight
on disk: keep only the per-drive frames listed in the bundled sample
list (plus the matching `velodyne_points` / `oxts` for poses if
needed) and drop the rest. Full raw drives total ~65 GB at
`~/data/kitti/raw`; the pruned footprint is an order of magnitude
smaller.

## Adding a new reproduction

1. Read the target paper's evaluation section carefully. Note:
   - Exact dataset + split + sample list.
   - View count / resolution / crop policy.
   - Scale alignment (metric? median? scale-and-shift?).
   - The metric name and the exact numerical value.
2. Write `reproductions/<short-name>.yaml`:
   - `model.name` + `kwargs` to match the paper's model variant + settings.
   - `dataset.name` + `kwargs` to match the paper's sample selection.
   - `tasks`, `scale_alignment`, `max_views` to match the protocol.
   - `paper_reference.primary_metric`, `.value`, `.tolerance_relative`.
   - `paper_reference.citation` — point a reader at the exact table/line.
3. For sample-level reproducibility, commit a `<short-name>.samples.txt`
   listing sample IDs in evaluation order and reference it from the YAML.
4. On the first successful run, pin the observed value in the YAML's
   `paper_reference.value` (if not already known from the paper) and the
   final `tolerance_relative`.

## Why tolerances

Bitwise reproducibility on CUDA is not possible for most current foundation
models — mixed-precision and cuDNN autotune introduce run-to-run noise. We
therefore express agreement as a **relative** tolerance on the primary
metric (default ±5%). If a run falls outside tolerance, investigate:

- Coordinate-system drift (the `conventions.py` assertions should catch most).
- Resolution / resize interpolation differences.
- Depth vs disparity vs inverse-depth confusion in the adapter.
- Scale alignment mode mismatch.

These failure modes are tracked as known traps in `plan.md § 9`.
