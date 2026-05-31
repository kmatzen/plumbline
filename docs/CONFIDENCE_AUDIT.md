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
| Depth Pro Middlebury | T1 | better | **L3** recipe | Low | No — 15-sample set, no public eval |
| Depth Pro NuScenes | T1 | better | **L3** subset | Low | No — random-881 subset seed unknown |
| Depth Pro Sun-RGBD | T1 | worse | **L2** GT/pairing | Med | No — large miss points at GT/pairing |
| VGGT ETH3D 13-scene | D10 | worse | **L2** scene-specific | High | Localized to `terrains` (12/13 beat paper) |
| VGGT DTU chamfer | D3 | worse | **L4** checkpoint | High | **Yes** — public VGGT-1B output ~2× off |

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

## L2 · Dataset / GT parsing — one confirmed bug, two suspects

This is where a real, fixable plumbline bug *can* hide, so it gets the most
scrutiny.

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

### Suspected — Depth Pro Sun-RGBD (Table 1)
δ₁ **0.451 vs paper 0.890** — the only Depth Pro Table-1 cell that reads
*dramatically worse*. The adapter is cleared by the iBims indoor sanity
(0.8458), so a 2× miss on another indoor set points at **GT decode or RGB↔depth
pairing** in our Sun-RGBD loader, not the model.
**Confident:** not the adapter.
**Unknown:** whether it's the depth unit/scale decode, the sensor-vs-clean GT
choice, or frame pairing. Needs a single-record GT diff against a known-good
Sun-RGBD reader.

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
- **Depth Pro Sintel / Middlebury / NuScenes (Table 1).** Adapter cleared by
  iBims. Sintel: protocol aligned, gap is aggregation/clip. Middlebury (reads
  *better*, 15 samples) and NuScenes (reads *better*, random-881 subset) hinge
  on a **subset seed / aggregation we don't have**.
  **Unknown:** the exact per-dataset subset + aggregation; these are the least
  documented Table-1 cells.

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
  RGB/GT) was localized by single-record diff and fixed.
- For the L3-resolved cells, we have *named the exact recipe* the paper used
  (best-of-N, per-seq LAD2, lineage fill, checkpoint version) — usually by
  reading upstream eval code or the issue tracker, not by guessing.
- "Better-than-paper" cells are understood as recipe/preprocessing differences,
  not as wins we are claiming.

**What is still unknown (the honest open list)**
1. DA-V2 Table-2 **native** preprocessing on ETH3D / Sintel / DIODE (we have ✅
   on these datasets via the MoGe-bundle protocol; the native recipe is open).
2. DUSt3R **indoor** depth recipe (NYU/Bonn) — paper-private.
3. Depth Pro Table-1 **subset + aggregation** for Sintel / Middlebury /
   NuScenes / Sun-RGBD (Sun-RGBD additionally suspected L2 GT/pairing).
4. Whether VGGT-ETH3D **`terrains`** is a model failure or a scene-specific GT
   artifact (everything else on ETH3D is fine).

Each unknown is a single, scoped question — not a vague "doesn't reproduce."
Cross-references: [`DISCREPANCIES.md`](DISCREPANCIES.md) (D-numbers),
[`BLOCKED.md`](BLOCKED.md), and the per-cell handoff docs.
