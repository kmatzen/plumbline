# Confidence audit: where does each gap actually live?

Every plumbline reproduction is a stack of four independent layers. When an
observed number disagrees with a paper, the discrepancy lives in **exactly one
of them** — and the whole point of this document is to say *which*, for each
off-paper or open cell, and to separate **what we have confirmed** from **what
is still unknown**.

This is the answer to "what, precisely, is suspicious?" — not a list of failures.

## The four layers

| Layer | What it is | "Suspicious" means… | How we clear it |
|---|---|---|---|
| **L1 · Adapter / method** | Our model wiring: convention conversion, resolution, normalization, the inference call, the alignment mode we pass | The model is being driven wrong (a flip, a unit, a crop, a bad alignment) | A ✅ paper-match for the **same model on another dataset**, or a sanity cell on the **same dataset with another model**, isolates the model from the dataset |
| **L2 · Dataset / GT parsing** | Our loader: depth decode, units, valid-pixel mask, intrinsics, RGB↔GT geometric alignment, frame subset | The GT (or its pairing with RGB) is wrong before any model runs | Single-record diff of decoded GT against a reference; per-scene/per-frame breakdown to find a localized blowup |
| **L3 · Paper eval recipe** | The paper's *own* scoring: alignment recipe (per-frame median vs per-seq scale+shift LAD2), best-of-N seeds, frame subset, depth clip, aggregation, checkpoint version | The paper scored in a way its text doesn't fully state (often *paper-private*, or the released **code disagrees with the paper text**) | Read the upstream eval code/issue tracker; reproduce the paper number **on the paper's own pipeline** |
| **L4 · Released artifact** | The public checkpoint itself produces different output than the paper's internal one | The weights we can download are not the weights the paper measured | Exhaust the documented inference knobs; show the gap survives in raw model output |

**The headline finding:** **no currently-open discrepancy is localized to L1.**
Every off-paper cell either reproduces the model elsewhere or has a same-dataset
sanity check that clears the adapter. The gaps are in L2 (one confirmed bug,
two suspected), L3 (the dominant cause), or L4.

---

## Master table

Direction is relative to the paper: **better** = plumbline beats the paper
number, **worse** = misses it. "Better" gaps are themselves a signal — they
almost never come from an adapter bug.

| Cell | D# | Dir. | Gap lives in | Confidence | Root cause known? |
|---|---|---|---|---|---|
| GeoWizard NYU / KITTI | D17/D18 | worse | **L3** recipe | High | **Yes** — best-of-N seeds (author-confirmed) |
| MonST3R Bonn (Table 3) | D27 | better | **L3** recipe | High | **Yes** — per-seq scale+shift LAD2; code ≠ text |
| MonST3R Sintel (Table 3) | D27 | worse | **L3** + L1-fragility | Med | Partly — recipe + `temple_2` outlier |
| CUT3R NYU / KITTI | D24 | better | **L3** recipe | High | **Yes** — lineage filled/no-crop pipeline |
| Marigold KITTI | D9 | worse | **L4** checkpoint | High | **Yes** — paper v1-0/50-step; we pin v1-1/1-step |
| DUSt3R NYU | D28 | worse | **L3** recipe | Med | No — indoor recipe paper-private (KITTI ✅) |
| DUSt3R Bonn | D28 | worse | **L3** + L1-domain | Med | Partly — recipe + DUSt3R is not a dynamic model |
| DA-V2 native ETH3D | D31/D33 | better | **L2 (fixed)** → **L3** | Med | Partly — RGB/GT bug fixed; residual recipe open |
| DA-V2 native Sintel | D32 | better | **L3** recipe | Med | No — sky-mask / aggregation recipe |
| DA-V2 native DIODE | D29 | worse | **L2/L3** preprocessing | Med | Partly — MoGe-bundle preprocessing explains gap |
| Depth Pro Sintel | T1 | worse | **L3** recipe | Med | No — aggregation / clip unknown |
| Depth Pro Middlebury | T1 | better | **REMOVED** (was L3) | — | Loader removed pre-release — no verified anchor |
| Depth Pro NuScenes | T1 | better | **REMOVED** (was L3) | — | Loader removed pre-release — no verified anchor |
| Depth Pro Sun-RGBD | T1 | worse | **REMOVED** (was L2) | — | Loader removed pre-release — likely GT/pairing bug |
| VGGT ETH3D 13-scene | D10 | worse | **L2** scene-specific | High | Localized to `terrains` (12/13 beat paper) |
| VGGT DTU chamfer | D3 | worse | **L4** checkpoint | High | **Yes** — public VGGT-1B output ~2× off |
| UniK3D ETH3D (Table 21) | T21 | better | **L3** recipe | Med | Partly — native 454-frame set vs UniK3D's HDF5 fixed-res eval |
| UniK3D DIODE (Table 22) | T22 | better | **L3** recipe (small) | Med | Partly — ~6% on the *exact* 325-img set; minor clip/align within [0.01, 25] m |
| Depth Pro ETH3D (Table 1) | T1 | worse | **L3** recipe | Med | No — far-range metric-scale saturates (indoor ✅, far outdoor →0) |

---

## L1 · Adapter / method — broadly CLEARED

This is the layer a user most needs reassurance about, and it is the one we are
most confident in. We never have an open cell whose gap we attribute to a plain
adapter bug, because every model is cross-checked:

- **Same model, other dataset matches.** DUSt3R misses NYU/Bonn but **matches
  KITTI** (0.1049 vs 0.1074) and **CO3Dv2 pose** (mAA 0.7893 vs 0.772) — the
  DUSt3R wiring is correct, so its indoor-depth gap is not L1. Same logic clears
  DA-V2 (33-way verified), MoGe, Metric3Dv2, VGGT (CO3Dv2 ✅), MonST3R
  (Sintel-pose ✅, NYU ✅), Depth Pro (Booster ✅).
- **Same dataset, sanity cell matches.** Depth Pro on **iBims-1 indoor GT reads
  δ₁ 0.8458** with the same weights and metric path — so the adapter handles
  indoor metric GT correctly, and the Sun-RGBD/Sintel misses are not a broken
  adapter.

The one adapter that **did** look like an L1 bug — **π³** scoring 6–20× worse
than VGGT under identical alignment while the paper reports π³ ≈ VGGT — was
**removed** pre-release rather than shipped unverified (see
[`blocked/PI3_RECONSTRUCTION.md`](blocked/PI3_RECONSTRUCTION.md)). So the
statement above holds for what we ship: no *open* cell is an L1 bug.

> The one place L1 contributes is as *fragility*, not a bug: single-frame
> view-duplicate models (MonST3R/DUSt3R) are genuinely unstable on a few
> textureless synthetic frames (`temple_2`), which inflates an **equal-frame**
> mean more than the paper's pixel-weighted mean. That is a real model
> property, not a wiring error — and it compounds an L3 recipe delta rather
> than standing alone.

**Confident:** no off-paper cell is caused by a convention/unit/resolution bug
in an adapter.
**Unknown:** none at this layer.

---

## L2 · Dataset / GT parsing — one confirmed bug, three loaders removed, one suspect

This is where a real, fixable plumbline bug *can* hide, so it gets the most
scrutiny. When a loader had **no verified anchor** and a suspicious result, we
removed it rather than ship it (the three Depth Pro Table-1 loaders below).

### Confirmed and fixed — DA-V2 native ETH3D RGB/GT misalignment (D31)
The clearest L2 case we have found. Per-view GT was rendered at the DA-V2
inference cap (≈345×518) while RGB stayed at native DSLR resolution
(~4135×6205) padded into the same canvas — so each GT pixel (full-FOV at render
scale) was scored against a pred pixel covering only the top-left patch of the
FOV. **Fix:** area-resample RGB to the GT render size before inference
(`resize_images_to_pv_render`). A single-sample check dropped AbsRel 0.224 →
0.024. This is the proof that the parsing layer *is* audited and that bugs here
are findable and fixable.
**Confident:** the geometric RGB↔GT alignment is now correct.
**Unknown:** a residual gap remains *after* the fix — that part is L3 (below).

### Suspected → REMOVED — Depth Pro Sun-RGBD / Middlebury / NuScenes (Table 1)
δ₁ **0.451 vs paper 0.890** on Sun-RGBD — the only Depth Pro Table-1 cell that
reads *dramatically worse*. The adapter is cleared by the iBims indoor sanity
(0.8458), so a 2× miss on another indoor set points at **GT decode or RGB↔depth
pairing** in the Sun-RGBD loader, not the model. Middlebury (0.759 vs 0.605) and
NuScenes (0.594 vs 0.491) read *better* than paper against an unverifiable
15-sample set / random-881 subset.

**Resolution (2026-05-31): the three loaders were removed before public
release** — none had a verified result proving the *loader* parsed GT correctly,
so their numbers were unverifiable and (for Sun-RGBD) looked like a bug. We do
not ship code tied to a suspiciously-unverifiable result. The attempts are
documented in `docs/blocked/DEPTH_PRO_{SUN_RGBD,MIDDLEBURY,NUSCENES}_TABLE1.md`.
The Depth Pro **adapter** stays (Booster ✅); only the single-purpose loaders
went.

### Suspected (scene-specific) — VGGT ETH3D 13-scene `terrains` (D10)
Aggregate Completeness is dragged from ~0.56 (median of 12 scenes) to the
13-scene mean by **`terrains` alone: Completeness 10.18 m, 13× any other
scene**. Excluding it, the 12-scene mean (0.515) is **27% tighter than paper**.
So 12/13 scenes confirm the method; the suspicion is fully localized to one
scene.
**Confident:** method + protocol correct on 12/13 scenes.
**Unknown:** whether `terrains` is a genuine VGGT failure (sparse/low-texture
geometry) or an L2 scale/GT artifact specific to that scene's reconstruction.
Resolvable by inspecting the `terrains` predicted vs GT point cloud directly.

---

## L3 · Paper eval recipe — the dominant cause

Most off-paper cells are here: the paper scored in a way its text doesn't fully
specify. Sometimes we have **pinned the recipe**; sometimes it remains unknown.

### Recipe KNOWN (root cause nailed)
- **GeoWizard NYU/KITTI (D17/D18).** Paper number is **best-of-N seeds**, not a
  fixed-seed metric — author-confirmed on `fuxiao0719/GeoWizard#36`. Three
  independent reproductions (incl. ours, 0.0574) converge on the single-seed
  value; the paper's 0.052 is the cherry-picked seed. Adapter ✓, dataset ✓.
- **MonST3R Bonn (D27).** Paper §4.2 text says "per-frame median," but the
  released `depth_metric.ipynb` actually scores **per-sequence scale+shift LAD2,
  valid-pixel-weighted across 5 sequences**. The code disagrees with the text;
  plumbline is text-faithful (and reads *better*). Frame-subset ruled out
  (0.0635 on the exact subset). This is the canonical "released code ≠ paper
  text" finding and the template for D9/D17/D24/D28.
- **CUT3R NYU/KITTI (D24).** Paper uses the DUSt3R-lineage **filled + no-crop**
  GT pipeline; plumbline uses strict raw+crop. Off-paper *better*. The lineage
  fill recipe is understood but deliberately not ported (it loosens the GT).
- **Marigold KITTI (D9).** Strictly L4-adjacent: the paper cell is **v1-0 /
  50-step**, plumbline pins the newer distilled **v1-1 / 1-step** default. The
  paper 0.0992 **reproduces end-to-end on Marigold's own v1-0 pipeline** — so
  adapter ✓, dataset ✓; the delta is a documented checkpoint/step choice.

### Recipe UNKNOWN (localized to L3, not yet pinned)
- **DUSt3R NYU/Bonn (D28).** DUSt3R §4.3 does not specify the indoor scoring
  recipe. KITTI (outdoor) ✅ clears the adapter; the indoor GT-processing recipe
  is paper-private. The paper's 0.065 NYU number is **bracketed** by our
  Eigen-crop (0.0489) and lineage (0.0777) re-scorings of the *same predictions*
  — i.e. the whole gap is a GT-processing choice, not the model. Bonn adds an
  L1-domain caveat: DUSt3R is not a dynamic-scene model (that is MonST3R's
  premise), so per-frame median can't compensate for systematic dynamic-region
  error the way per-seq LAD2 (D27) does.
  **Unknown:** the exact indoor alignment/clip the DUSt3R authors used.
- **DA-V2 Table 2 native, ETH3D / Sintel / DIODE (D29/D31/D32/D33).** After the
  D31 RGB/GT fix, all three variants still read ~30% *off* on full ETH3D, and
  Sintel/DIODE are off too — **yet the MoGe-bundle Table-3 cells on the very
  same datasets are ✅**. Same model + same dataset, different preprocessing →
  the gap is the **DA-V2-Table-2-native eval recipe**, almost certainly an
  upstream preprocessing/aggregation we have not reconstructed.
  **Unknown:** the native Table-2 preprocessing (sky-mask handling on Sintel,
  outdoor depth handling on DIODE, GT source + recipe on ETH3D).
- **Depth Pro Sintel (Table 1).** Adapter cleared by iBims; protocol aligned,
  gap is aggregation/clip. Runs on the ✅-anchored `sintel` loader, so it stays
  as a documented ℹ️ finding.
  **Unknown:** the exact aggregation/clip.
  _(The sibling Middlebury / NuScenes / Sun-RGBD loaders were **removed**
  pre-release — no verified anchor; see the L2 section.)_
- **Depth Pro ETH3D (Table 1).** Ran 2026-06-11 (454/454, exact Table 16
  manifest): δ₁ **0.3648** (GT focal) / 0.3339 (self-focal) vs paper **0.415** —
  bimodal by scene. Close indoor scenes match the paper (kicker 0.90, office
  0.92, pipes 0.93); three far-range outdoor scenes collapse to δ₁≈0 because
  Depth Pro under-scales far metric depth (meadow GT median 8.2 m → pred 2.2 m,
  ~3.7× compression). Same paper-private far-depth recipe shape as the other
  Depth Pro Table 1 cells. NOT tuned.
  **Unknown:** the far-depth preprocessing/clip nuance the paper applied.
- **UniK3D ETH3D / DIODE (Tables 21/22).** Both metric, no-alignment cells read
  *better* than the paper (ETH3D AbsRel 0.1544 vs 0.236, 35% tighter; DIODE
  0.1509 vs 0.161, 6.3% under). DIODE scores the **exact** 325-image official
  indoor val set with UniK3D's own [0.01, 25] m clip and lands ~6% better on all
  three headline metrics in mutual agreement — a tight repro that narrowly
  misses the ±5% band (→ ℹ️, not ✅). ETH3D's larger margin is a
  frame-set/resolution delta: `eth3d-native-depth` scores the 454
  native-resolution DSLR frames over the 13 high-res train scenes, whereas
  UniK3D's own eval reads an HDF5-packed ETH3D (`test_split=train.txt`) at a
  fixed smaller `image_shape` — a different, likely-harder sample/resolution
  mix. NOT tuned.
  **Unknown:** UniK3D's exact ETH3D HDF5 frame list + eval resolution.

---

## L4 · Released artifact — the public checkpoint differs

- **VGGT DTU chamfer (D3).** The PatchmatchNet confidence filter and fp32
  inference were both verified to be **no-ops** on the gap, and the protocol is
  structurally correct (per-view-masked chamfer). The residual ~2× survives in
  the **raw public VGGT-1B point output** — i.e. the downloadable checkpoint
  does not reproduce the paper's internal DTU number.
  **Confident:** adapter + protocol exhausted.
  **Unknown:** nothing actionable on our side; this is an upstream-weights gap.

---

## Bottom line

**What we are confident in**
- No open discrepancy is an adapter (L1) bug. Every model is cross-validated by
  a same-model or same-dataset ✅/sanity cell.
- The parsing layer (L2) is audited: the one real bug we found (D31 ETH3D
  RGB/GT) was localized by single-record diff and fixed; the three loaders we
  could *not* verify (Depth Pro Middlebury / NuScenes / Sun-RGBD) were removed
  rather than shipped.
- For the L3-resolved cells, we have *named the exact recipe* the paper used
  (best-of-N, per-seq LAD2, lineage fill, checkpoint version) — usually by
  reading upstream eval code or the issue tracker, not by guessing.
- "Better-than-paper" cells are understood as recipe/preprocessing differences,
  not as wins we are claiming.

**What is still unknown (the honest open list)**
1. DA-V2 Table-2 **native** preprocessing on ETH3D / Sintel / DIODE (we have ✅
   on these datasets via the MoGe-bundle protocol; the native recipe is open).
2. DUSt3R **indoor** depth recipe (NYU/Bonn) — paper-private.
3. Depth Pro Table-1 **Sintel** aggregation/clip (the Middlebury / NuScenes /
   Sun-RGBD loaders were removed pre-release — see L2).
4. Whether VGGT-ETH3D **`terrains`** is a model failure or a scene-specific GT
   artifact (everything else on ETH3D is fine).

Each unknown is a single, scoped question — not a vague "doesn't reproduce."
Cross-references: [`DISCREPANCIES.md`](DISCREPANCIES.md) (outstanding work),
[`BLOCKED.md`](BLOCKED.md), and the per-cell handoff docs.

---

## Per-paper trust

How much to trust each paper's published cells when reading the matrix. (Moved
here from DISCREPANCIES.md — this is documented understanding, not outstanding
work.)

| Paper | Verified ✅ | Trust | Note |
|---|---|---|---|
| Depth Anything V2 (2406.09414) | 8 | High | All cells reproduce; one fabricated Sintel pin was caught and demoted. |
| Metric3D-v2 (2404.15506) | 4 | High | NYU + KITTI L/Giant, all within ±10 %. |
| MoGe-1 (2410.19115) | 5+ | High | After fixing a Table-2-vs-3 citation error (2026-04-20 audit). |
| Marigold (2312.02145) | 2 | High | Both cells reproduce end-to-end on Marigold's own pipeline; KITTI gap is a v1-0/v1-1 checkpoint delta (D9, L4). |
| GeoWizard (Fu 2024) | 0 match / 2 explained | Explained | Paper number is best-of-N seeds (author-confirmed, D17/D18); adapter structurally correct. |
| Depth Pro (2410.02073) | 1 (Booster) | Partial | Booster ✅; Sintel off-paper; Middlebury/NuScenes/Sun-RGBD loaders removed (no anchor). |
| Depth Anything 3 (2511.10647) | 1 (NYU δ₁) | Moderate | Paper Table 4 reports only δ₁; NYU is the only paper-comparable cell. |
| MoGe-2 (2507.02546) | 0 | No path | Paper publishes only 10-dataset averages — no per-dataset ViT-L cell exists. |
| VGGT (2503.11651) | 1 (CO3Dv2 pose) | Mixed | CO3Dv2 pose ✅; DTU ~2× (D3, upstream checkpoint); ETH3D under/over (D4/D10). |
| MASt3R (2406.09756) | 1 (CO3Dv2 pose) | High | Table 3 CO3Dv2 ✅; cell PDF-verified (D23). |
| DUSt3R (2312.14132) | 2 (KITTI, CO3Dv2) | High | Outdoor depth + pose ✅; indoor depth recipe paper-private (D28, L3). |
| CUT3R (2501.12387) | 0 plumbline-match | High | All 3 paper cells reproduce end-to-end on CUT3R's own pipeline; plumbline cells are stricter-protocol deltas (D24, L3). |
| MonST3R (2410.03825) | 2 (NYU, Sintel pose) | High | NYU depth ✅ + Sintel trajectory pose ✅; Bonn/Sintel depth are recipe deltas (D27, L3). |
