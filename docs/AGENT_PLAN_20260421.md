# Plan — fix the 2026-04-21 agent run's 6 not-green rows

Context: the GPU-rental agent run (see `docs/AGENT_RUN_20260421.md`) left 4
rows ⚠️ OFF-PAPER and 2 ❌ SKIPPED (geowizard). This plan organizes the
fixes by effort + risk, cheap-first. Two of the four OFF-PAPER rows likely
clear with a 15-minute change; one is a multi-hour dataset-loader rewrite.

## Priorities

Ordered to maximize "fixes per hour":

1. **[15 min]** Cross-check DA-V2 paper Table 2 for Base-variant NYU
   (`da-v2-base-nyuv2`).
2. **[15 min]** Inspect `scale_shift_robust` fit space (depth vs disparity)
   (`moge-vitl-nyuv2`).
3. **[15 min]** Decide the `moge-vitl-diode-indoor` structural question
   (demote vs re-cite).
4. **[2–4 h]** Patch or shim upstream GeoWizard so it imports against
   current diffusers / transformers (2 rows fixed at once).
5. **[4–8 h]** Rewrite DIODE loader to match MoGe's eval pipeline
   (`moge-vitl-diode-both`, probably also tightens `-indoor`).
6. **[2 h]** `vggt-eth3d-multiscene-chamfer` re-run on the 3-scene subset
   with a corrected citation (the 0.709 paper value is for the full 13-scene
   split — either match it by running the full split or re-cite).

The first three are all 15-minute laptop-side edits, no rental box needed.
Do those in one sitting, commit, then pick up the adapter work for a
follow-up session.

---

## 1. `da-v2-base-nyuv2` (citation error)

**Observed** 0.0456 vs **cited** 0.0490. YAML tagged as `UNVERIFIED`:
the 0.049 came from an LLM fetch that was never cross-read against the
paper. Small (0.051 vs 0.053) and Large (0.043 vs 0.042) both matched.

**Plan**

1. WebFetch `arxiv.org/pdf/2406.09414` (DA-V2 paper), read Table 2.
2. Locate the Base-variant NYU AbsRel cell in the zero-shot relative
   evaluation block. The paper publishes three variants (S / B / L); Small
   and Large are cross-checked.
3. If the paper says ~0.046 (matches our observed): update
   `reproductions/da_v2_base_nyuv2.yaml`:
   - `value: 0.046` (or whatever is printed)
   - strip the `UNVERIFIED` marker from `citation` and `notes`
   - keep `tolerance_relative: 0.05`
4. If the paper actually says 0.049: the run is genuinely 7 % off and a
   deeper dive is warranted (Eigen crop? depth-field default? HF checkpoint
   version?). Compare against `da-v2-large-nyuv2` (1.9 % off) and
   `da-v2-small-nyuv2` (3.8 % off) to narrow which variant-specific config
   diverges.

**Verification:** no GPU run needed if the paper says ~0.046 (observed is
frozen); re-run if we change protocol.

---

## 2. `moge-vitl-nyuv2` (protocol / solver)

**Observed** 0.0305 vs **cited** 0.0341. The MoGe paper Table 3 has two
NYU ViT-L rows, only one of which matches our alignment protocol:

| Paper row | AbsRel | plumbline? |
|---|---:|---|
| Affine-invariant **depth** | 0.0297 | — |
| Affine-invariant **disparity** | 0.0341 | ← this is what `scale_shift_robust` claims |

Our 0.0305 is one point above the depth row. Two hypotheses:

**H1 (bug):** `scale_shift_robust` fits in depth space, not disparity.
**H2 (real):** plumbline's ROE is stronger than MoGe's own ROE on the
same predictions; we genuinely beat the paper's disparity-row number.

**Plan**

1. Read `src/plumbline/scale_alignment.py::scale_shift_robust` —
   specifically where it builds the linear system. If it regresses
   `pred → gt` directly it's depth-space (H1); if it regresses
   `1/pred → 1/gt` or the equivalent with a fit, it's disparity-space (H2).
2. Also inspect the dispatch in `runner.py` — verify that `scale_alignment:
   scale_shift_robust` in the YAML routes to the same code path (not
   some other alias).
3. If H1: fix the solver, re-run `moge-vitl-nyuv2` + all other
   `scale_shift_robust` reproductions (`moge-vitl-diode-indoor`,
   `moge-vitl-diode-both`). Expect the NYU number to move toward 0.034.
4. If H2: add a paragraph to the YAML notes explaining "we beat paper's
   disparity ROE by ~10 %, attributable to <specific IRLS detail>" and
   update the `value` to 0.0297 (or widen the citation to either row).

**Verification:** one `plumbline reproduce moge-vitl-nyuv2` re-run under the
fixed solver (2 min).

---

## 3. `moge-vitl-diode-indoor` (structural mismatch)

**Observed** 0.0424 vs **cited** 0.0400. The cited 0.0400 is MoGe paper's
**combined** DIODE val (indoor + outdoor averaged); this YAML runs
**indoor-only**. Structurally different populations.

**Plan**

1. Re-read MoGe Table 3 DIODE column — does it break out per-domain? If
   yes, cite the indoor-only number. If no:
2. Demote this YAML: `source_confidence: approximate` (not
   `verified_pdf`), or null out `paper_reference.value`. Keep the
   harness running it as an informational smoke test with an "is this
   roughly sane?" threshold (e.g. `< 0.08`) in a separate mechanism.
3. Keep `moge-vitl-diode-both` as the one paper-match DIODE row (pending
   the loader fix in § 5).

**Verification:** YAML edit only, no re-run.

---

## 4. GeoWizard environmental failure (2 rows fixed at once)

Upstream `GeoWizard` (cloned to `/workspace/deps/geowizard/`) does
`from diffusers.models.embeddings import PositionNet` — that class was
renamed in diffusers > ~0.26. Separately, older diffusers versions tried
to `from transformers.utils import FLAX_WEIGHTS_NAME`, which was dropped
in transformers 5.x.

Both break on any modern diffusers / transformers combo, but we need
modern diffusers for Marigold. The right fix is NOT "pin an old diffusers
for everyone."

**Plan**

Option A (preferred, low blast radius): **shim the upstream** from
`src/plumbline/models/geowizard.py::_load()`. Before the upstream pipeline
module is imported, monkey-patch:

```python
import diffusers.models.embeddings
if not hasattr(diffusers.models.embeddings, "PositionNet"):
    from diffusers.models.embeddings import GLIGENTextBoundingboxProjection as PositionNet
    diffusers.models.embeddings.PositionNet = PositionNet
```

(Verify the actual replacement class; it may be `PixArtAlphaCombinedTimestepSizeEmbeddings`
or similar depending on what GeoWizard actually uses `PositionNet` for.)

Option B: fork GeoWizard upstream, commit the renames, vendor the fork
path. More invasive; skip unless A is too brittle.

Option C: drop GeoWizard from the verified_pdf queue. Two rows
(`geowizard-nyuv2`, `geowizard-kitti`) get demoted to smoke tests.

**Verification:** `plumbline reproduce geowizard-nyuv2` — if MATCH (~0.048
vs cited 0.052), do `-kitti` next.

---

## 5. DIODE outdoor loader — `moge-vitl-diode-both` (188.6 % gap)

The big one. The MoGe paper's DIODE eval diverges from plumbline's in
three ways (all flagged in the YAML's own notes):

| Concern | Plumbline today | MoGe paper |
|---|---|---|
| Depth field | raw `.npy` | uint16 `depth.png` from `Ruicheng/monocular-geometry-evaluation` |
| Sky mask | loader's `depth_mask` | per-sample `segmentation.png` from same HF dataset |
| Depth clip | `[1e-3, 50]` m | probably 80 m (KITTI) or 30 m (NYU-style); needs confirmation |

**Plan**

1. Pull MoGe's eval code (their `evaluate.py` or `eval_depth.py` in
   `microsoft/MoGe`) and nail down exactly which HF dataset fields they
   use and their preprocessing.
2. Stage `Ruicheng/monocular-geometry-evaluation` DIODE assets into the
   plumbline S3 cache (part of it is already there for iBims-1 and GSO).
3. Write a second DIODE loader in `src/plumbline/datasets/diode.py` —
   something like `DIODEMogeEvalLoader` — that reads the preprocessed
   depth + segmentation assets. Default `diode` keeps the raw-.npy path
   for backwards compat; opt into the MoGe variant via a
   `depth_source: "moge_hf"` kwarg on the `dataset:` block in
   `reproductions/moge_vitl_diode_*.yaml`.
4. Tighten `depth_clip` based on what MoGe actually does.
5. Re-run both DIODE reproductions.

**Verification:** indoor run should land near cited 0.040; combined run
the same. If combined is still > 0.05, inspect sky-mask application.

**Effort:** 4–8 hr, new code + tests + two re-runs.

---

## 6. `vggt-eth3d-multiscene-chamfer` (subset / full-split mismatch)

**Cited** 0.709 is VGGT paper Table 3, **Overall** column, cross-scene
average across the **full** ETH3D 13-scene eval. Plumbline's YAML runs
a **3-scene subset** (courtyard + delivery_area + facade). The earlier
local run landed 0.818 on the subset, which is plausible for that slice
but not comparable to the full-split number.

The partial run from this rental (with early-sample skips before vggt
install landed) is unusable for this cell anyway; a clean re-run is
needed.

**Plan**

1. Re-run `vggt-eth3d-multiscene-chamfer` cleanly (vggt now installed).
2. If observed is near 0.818 (our earlier local number), decide:
   - **A:** stage the remaining 10 ETH3D scenes and run the full 13-scene
     split, which should land at 0.709. Data footprint is significant —
     the current 3-scene subset is 3.2 GB, so full-split is ~14 GB.
   - **B:** find a per-scene breakdown in VGGT's supplementary or
     reproduce per-scene Chamfer and compute a subset-specific paper
     target.
   - **C:** demote to informational.
3. `vggt-paper-dtu-mvs` is a separate slow run (tolerance already 5 %);
   run it cleanly once the queue is idle.

**Verification:** one slow re-run per scenario.

---

## Sequencing

**Today (laptop, no GPU):** items 1, 2, 3 — three 15-minute YAML/code
edits.

**Next GPU-rental session:** items 4 (GeoWizard shim + repros), 5 (DIODE
loader + repros), 6 (VGGT-ETH3D clean re-run + decision).

**Defer:** anything beyond the 4 off-paper + 2 skipped. The smoke-test
(§6 of the runbook) targets — pi3 smokes, additional ETH3D scenes —
should land after the paper-match matrix is fully green.

## Rollback plan

Every change is on `main` already; if any edit regresses a
currently-green row, `git revert <sha>` returns to the pre-change state
and the agent report regenerates from the preserved `/tmp/results/` +
S3 mirror.
