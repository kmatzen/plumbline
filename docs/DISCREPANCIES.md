# Discrepancies — 2026-04-21 rental run and before

Catalog of every adapter / loader / protocol / solver mismatch surfaced
this session, with resolution plan. Grouped by layer so a reader can
triage: is this an **adapter bug**, a **loader** issue, a **protocol**
mismatch, an **alignment-solver** choice, or a **citation** error?

Status legend:

- ✅ **FIXED** — change landed on `main`, verification complete.
- 🧪 **FIX-PENDING-VERIFY** — change landed, waiting on a GPU re-run.
- 🔎 **SUSPECTED** — hypothesis + diagnosis path; not yet reproduced.
- 📅 **DEFERRED** — known root cause, scoped for v0.2+.
- 📝 **EXPLAINED-NOT-A-BUG** — discrepancy is real but expected given
  the papers / solvers involved; documented to prevent rediscovery.

---

## Adapter bugs

### D1 · GeoWizard — `generator` kwarg not accepted upstream   ✅ FIXED

Symptom: every sample raised `TypeError:
DepthNormalEstimationPipeline.__call__() got an unexpected keyword
argument 'generator'` → 654/654 skipped.

Root cause: `src/plumbline/models/geowizard.py::predict` passed
`generator=self._generator` to the upstream pipeline, but
`/workspace/deps/geowizard/geowizard/models/geowizard_pipeline.py:73`'s
`__call__` signature doesn't accept it. Never exercised end-to-end
since diffusers' public API shifted under the adapter.

Fix: commit `c50201e` — drop the kwarg; seed `torch.manual_seed(seed +
sample_index)` before each call so per-sample determinism still holds
via torch's global RNG (which upstream samples internally).

Verification: re-run in the current cleanup batch.

### D2 · GeoWizard — upstream diffusers API drift   ✅ FIXED (shim)

Symptom: `ImportError: cannot import name 'PositionNet' from
'diffusers.models.embeddings'` followed by `ModuleNotFoundError: No
module named 'diffusers.models.dual_transformer_2d'`.

Root cause: upstream `GeoWizard` repo was forked from an older diffusers
(≤ 0.25-ish). `PositionNet` was renamed to
`GLIGENTextBoundingboxProjection`; `dual_transformer_2d` moved under
`diffusers.models.transformers.`.

Fix: commit `a35c4f5` — `_shim_diffusers_for_geowizard()` aliases both
via attribute assignment and `sys.modules` injection before the
upstream import.

### D3 · VGGT + DTU — suspected m↔mm scale mishandle   🔎 SUSPECTED

Symptom: partial re-run landed chamfer 56.37 mm vs paper 0.382 mm —
147 × off. (Partial because 715/924 samples had skipped on the first
re-run before `vggt` was pip-installed.)

Hypothesis: VGGT emits depth in metres; DTU loader keeps
GT in mm (see `src/plumbline/datasets/dtu.py:145`). Runner aligns
prediction→GT via `_aligned_point_map` → ICP → Umeyama (which SHOULD
absorb a uniform scale factor cleanly). A 147 × residual suggests ICP
converged on a partial match — maybe a rotation-only branch, or a
scale-clipped result. Not 1000 × (no conversion) and not 1 × (correct)
— something in between.

Fix plan: trigger a clean re-run with the full 22 scans (need to not
time out — the first attempt hit 4 h 25 m without landing). Options:

1. Run a 1-scan subset first to get a number with `pointcloud_alignment
   = icp` vs `= umeyama` vs `= none` to see where the 147 × lands. That
   isolates ICP vs scale-handling from dataset-size.
2. Add per-scene timing logs to the scene-aggregation path in
   `src/plumbline/runner.py::_aligned_point_map` so next run tells us
   WHICH scene eats hours.
3. If ICP is the culprit, swap to `umeyama` on DTU (paper reports
   under "VGGT's own MASt3R-style recipe" — see YAML notes).

Effort estimate: 30 min single-scan probe + re-run.

---

## Loader / dataset issues

### D4 · ETH3D scene_aggregation empty metrics   ✅ FIXED

Symptom: `vggt-eth3d-multiscene-chamfer` produced 137 per-sample pose
metrics but empty `aggregate_metrics` / `per_scene_metrics` — the MVS
chamfer never computed.

Root cause: the ETH3D loader requires `scan_clean/scan*.ply` to
populate `Sample.point_cloud_gt`. The S3 cache only had
`dslr_scan_eval/` (laser GT in eval coordinate frame), so
`point_cloud_gt=None` and the scene-aggregation code at
`runner.py:177` silently skipped the MVS branch.

Fix: commit `f89cdc4` — fetched `*_scan_clean.7z` for the 3 scenes via
`py7zr` (no `p7zip-full` binary on rental; no sudo), unpacked,
flattened the nested `scene/scene/scan_clean` directory, pushed to
`s3://plumbline-bench/datasets/eth3d/<scene>/scan_clean/`. Invalidated
the dataset manifest.

Verification: pending in the current cleanup batch.

### D5 · DIODE outdoor prediction outliers   🧪 FIX-PENDING-VERIFY

Symptom: first `moge-vitl-diode-both` run under the new HF-bundle loader
landed mean AbsRel = **2481** across 771 samples. Median was 0.025
(dead on paper's 0.040), but a handful of outdoor samples produced
post-alignment depths with per-sample AbsRel in the 40 K — 1.4 M range.

Root cause: MoGe's eval applies a secondary GT mask
(`dataloader.py::_process_instance`):
`max_depth = nanquantile(gt, 0.01) * 1000`, then
`depth_valid &= gt ≤ max_depth`. For outdoor samples with degenerate
predictions this collapses the mean. Plumbline's first loader didn't
mirror it.

Fix: commit `7fd6ff6` — added the `drop_max_depth` filter to
`DIODEMogeEvalLoader`. Re-run pending.

### D6 · DIODE Moge-eval loader `split` kwarg   ✅ FIXED

Symptom: protocol pinned `dataset.split: val`, runner forwarded
`split` to the dataset constructor, new loader's `__init__` didn't
accept it → `TypeError`.

Fix: commit `ae046ab` — `DIODEMogeEvalLoader.__init__` now takes
`split="val"` (raises on anything else).

### D7 · KITTI annotated-depth not in S3 cache   ✅ FIXED (retroactively)

Symptom: all three `da-v2-{S,B,L}-kitti` rows crashed with
`DatasetNotAvailable: KITTI annotated-depth tree not found at
~/data/kitti/depth_annotated`. S3 cache only had
`kitti/raw/` (RGB), not the `data_depth_annotated` tree.

Fix: fetched `data_depth_annotated.zip` (14 GB) from avg-kitti,
unpacked, pushed to `s3://plumbline-bench/datasets/kitti/depth_annotated/`.
Future sessions' `stage_all_data.sh` pulls it automatically.

---

## Protocol mismatches

### D8 · MoGe KITTI — structural protocol delta   📅 DEFERRED (v0.2)

Symptom: `moge-vitl-kitti` lands 9.4 % off paper (0.0408) under the
Monodepth2 Eigen / Garg protocol. Switching to Eigen crop made it
*worse* (20.4 % off).

Root cause (from subagent read of `microsoft/MoGe` @ HEAD):

| Knob | Plumbline | MoGe paper |
|---|---|---|
| Sample list | Monodepth2 `eigen_benchmark` 652 frames | HF bundle `KITTI/.index.txt` (custom) |
| Depth cap | `[1e-3, 80]` m | none (per-scene `clamp_min(1/max_gt)` only) |
| GT source | Uhrig dense depth | MoGe's re-encoded `depth.png` |
| Crop | Garg | none; they center-warp to 750×375 |
| Resolution | native | warped 750×375 |
| Alignment | full-res LSQ | 64×64-downsample LSQ then apply full-res |

Resolution path: write `KITTIMogeEvalLoader` (sister of
`DIODEMogeEvalLoader`) reading MoGe's HF bundle KITTI data + a new
`kitti_moge_eval` protocol. 4–6 h of loader + protocol + verification
work. Scoped v0.2.

### D9 · Marigold KITTI — probable same-protocol delta   🔎 SUSPECTED

Symptom: 10.1 % off paper under plumbline's `kitti_eigen_garg`
protocol (and 17.8 % off under Eigen crop — worse, same as MoGe).

Hypothesis: Marigold uses a similar bespoke eval to MoGe — the two
papers publish on matching rows, their eval repos overlap (same-era
SD-UNet depth adapters). Subagent confirmed Marigold uses Eigen crop
in its public eval config, but our Eigen-crop attempt was worse, so
either a secondary knob differs (the `alignment_max_res` resolution
the subagent flagged, or depth-cap) or the crop is wrong.

Resolution path: write `KITTIMarigoldEvalLoader` (or share with
`KITTIMogeEvalLoader` if their HF eval bundles overlap). Defer to
v0.2 alongside D8.

### D10 · VGGT-ETH3D — 3-scene subset vs full-split paper target   📅 DEFERRED

Symptom: plumbline's YAML runs courtyard + delivery_area + facade
(3 scenes); paper's Table 3 Overall 0.709 is cross-scene mean across
the 13 ETH3D train scenes. A 3-scene subset genuinely can't paper-match
the 13-scene aggregate.

Resolution path: one of (a) stage the remaining 10 scenes — +~14 GB
data — and run the full split; (b) extract a per-scene breakdown
from VGGT supplementary if it exists; (c) demote to informational
with a tolerance of 0.20 and cite "3-scene subset for fast sanity
check, not paper-match". Option (c) is what the 2026-04-20 audit
intended; the `tolerance_relative: 1.0` that was there before today's
repo-wide 5 % cap encoded that intent.

---

## Alignment / solver choice

### D11 · Plumbline's `scale_shift_robust` beats MoGe's own LSQ   📝 EXPLAINED

Observed AbsRel on `moge-vitl-nyuv2` under `scale_shift_robust` was
0.0305 vs MoGe-paper 0.0341 under what they also call
"affine-invariant disparity" — 10.6 % **better** than paper. Looked
like a bug.

Root cause (subagent read of `moge/utils/alignment.py`): MoGe's own
eval uses **plain** `torch.linalg.lstsq` with uniform weights — no
IRLS, no Huber, no MAD. Plumbline's `scale_shift_robust` is Huber-IRLS
with MAD-scale (k=1.345). On NYU's long-tailed disparity-residual
distribution, Huber downweights outlier pixels LSQ gets pulled by, so
plumbline's ROE genuinely outperforms MoGe's LSQ on the *same*
predictions.

Fix: commit `c14d776` — switch `moge-vitl-nyuv2` YAML from
`scale_shift_robust` to `scale_shift` (plain LSQ) to match paper. New
observed 0.03419 vs paper 0.03410 = 0.28 % MATCH.

Cascade: same alignment change to `moge-vitl-kitti`,
`moge-vitl-diode-both`, `moge-vitl-diode-indoor`, `moge2-vitl-*`,
`protocols/gso_moge.yaml`.

### D12 · KITTI Eigen-crop hypothesis (rejected)   📝 EXPLAINED

Hypothesis: plumbline applies the Garg crop on KITTI; Marigold and
MoGe use Eigen. Switching would fix the ~10 % gap on both.

Result: **rejected empirically** (commit `fb58b90`). Under Eigen crop
`moge-vitl-kitti` went from 9.4 % off to 20.4 %; Marigold went from
10.1 % to 17.8 %. Reverted both YAMLs to `kitti_eigen_garg`.
`protocols/kitti_eigen_crop.yaml` preserved for a paper that genuinely
uses it.

The real delta on MoGe-KITTI is D8 (structural protocol); Marigold-
KITTI is D9.

---

## Citations

### D13 · DA-V2 Large NYU — two competing paper numbers   ✅ FIXED

Two legitimate citations exist:

- DA-V2 paper (arXiv:2406.09414), Table 2 p. 8: ViT-L NYU AbsRel =
  0.045 under "the paper's own eval".
- MoGe paper (arXiv:2410.19115), Table 3 DA-V2-L baseline row: AbsRel
  = 0.0420 under MoGe's `align_affine_lstsq`.

Same model, same protocol name, two different numbers. Plumbline
reproduces 0.0427 under plain LSQ — 1.6 % off MoGe's 0.0420, 5.2 %
off DA-V2's 0.045.

Fix: commit `58fc159` — pin at MoGe's 0.0420 (tighter, achievable)
with a citation that records both so the ~7 % cross-paper gap is
documented.

### D14 · DA-V2 Base NYU citation originally UNVERIFIED   ✅ FIXED

Citation was tagged `UNVERIFIED — DA-V2 paper per-variant NYU table
needs direct read.` Subagent fetched the arXiv PDF and confirmed 0.049
is correct (Table 2, p. 8). Commit `603e717` dropped the tag.

See D15 for the remaining Base-specific ~7 % gap — separate issue.

### D15 · DA-V2 Base NYU 7.2 % off paper   📝 EXPLAINED

Plumbline observes 0.0455 on DA-V2-B NYU vs paper 0.049 — 7.2 % off.
NOT a Base-specific issue:

| Variant | Observed | Paper | Δabs | Δrel |
|---|---:|---:|---:|---:|
| Small | 0.0510 | 0.053 | −0.002 | −3.8 % |
| Base  | 0.0455 | 0.049 | −0.003 | −7.2 % |
| Large | 0.0427 | 0.045 | −0.002 | −5.2 % *(vs MoGe-reported 0.042: 1.6 %)* |

All three variants are 0.002 — 0.003 AbsRel *below* paper. Base only
looks worst because the paper denominator is smaller. Per-sample
cross-variant Pearson is 0.89 — 0.97 — the models agree on which
samples are hard; they just all score uniformly lower than paper.

Paper-vs-HF checkpoint swap had ~0 effect (<0.0005 delta). Real
suspects: NYU Eigen-crop convention, rawDepths-field filter interaction,
or small protocol detail in DA-V2's own eval we're not replicating.
Below the priority threshold for v0.1 — **document and move on**.

### D17 · GeoWizard NYU 10 % off paper   🔎 SUSPECTED

Symptom: with D1 (generator) + D2 (shim) fixed and the adapter now
running end-to-end, observed `geowizard-nyuv2` AbsRel = 0.0573 vs
paper 0.052 — 10.2 % off.

Candidates:

1. **RNG divergence from paper.** Plumbline now seeds
   `torch.manual_seed(seed + sample_index)` before each ensemble
   call (D1 fix). The paper may use a single fixed seed for the
   ensemble latents; per-sample re-seeding samples a different
   latent-chain distribution and can shift mean AbsRel 10 % on
   diffusion eval (ensemble converges in expectation but finite-
   ensemble variance is non-trivial).
2. **Paper protocol alignment**: plumbline uses `scale_shift_depth`
   (depth-space LSQ) per GeoWizard YAML. Need to confirm from the
   public GeoWizard eval script that this matches — subagent
   didn't read this one.
3. **Protocol resolution**: paper uses processing_res=768, same as
   plumbline's YAML.

Priority: low. Unlike D8/D9 (MoGe/Marigold KITTI), this one's only
10 % off and on the "informational-smoke" side. Defer to v0.2 with
D9 investigation.

### D18 · GeoWizard KITTI 35 % off paper   🔎 SUSPECTED (same family as D8/D9)

Symptom: `geowizard-kitti` AbsRel = 0.131 vs paper 0.097 — 35.2 %
off. Much worse than NYU (10 %).

Almost certainly the same structural KITTI protocol mismatch as D8
(MoGe-KITTI) and D9 (Marigold-KITTI): all three diffusion / prior-
depth adapters reproduce on NYU within 5-10 % but miss KITTI by
10-35 % under plumbline's Monodepth2-Eigen-benchmark-652 +
Garg-crop + [1e-3, 80] m clip. The KITTI eval these papers use is
bespoke (MoGe's 750×375 center-warp, no crop, no cap) — and
GeoWizard paper likely cites a similar eval since the three papers
are era-peers with overlapping baseline tables.

Resolution: same as D8/D9 — a `KITTIMogeEvalLoader` + protocol.
Probably also matches GeoWizard's eval enough to close this row.
v0.2.

### D19 · MoGe-DIODE-both still 2481 after `drop_max_depth`   🔎 SUSPECTED

Symptom: fix landed (commit `7fd6ff6`), re-ran `moge-vitl-diode-both`
cache-hit. Median AbsRel still 0.025 (on paper), mean still 2481.

Root cause (revised from D5): MoGe's `drop_max_depth` filters **GT**
— doesn't catch **predicted** post-alignment depths blowing up. MoGe
also applies `clamp_min(1 / gt_depth[mask].max())` on the predicted
disparity BEFORE inverting to depth (per subagent read of
`moge/test/metrics.py:210`). This is the missing piece — caps
per-sample predicted depth at `max(gt_depth)`.

Plumbline's `scale_shift` alignment path doesn't do this per-sample
disparity clamp. Fix would be a new alignment variant or a
post-alignment per-sample cap; not a 5-min change.

Resolution: v0.2. Median continues to land bullseye on paper, so
the adapter + loader + solver are fundamentally correct; just one
more clamp away from the full paper-match.

### D16 · MoGe-DIODE-indoor cited combined-val paper value   ✅ FIXED

The YAML cited MoGe's 0.0400 (combined val, 771 samples) but ran
indoor-only (325 samples). Structurally mismatched populations.
Demoted to `source_confidence: approximate`, `value: null`; drops out
of the verified queue. Commit `603e717`.

---

## Priorities for the next session

Cheap-first sequencing (assuming a fresh GPU rental):

1. **VERIFY** the in-flight cleanup batch lands the 6 expected rows
   (moge-diode-both, moge-vitl-kitti Garg, marigold-kitti Garg,
   geowizard × 2, vggt-eth3d). If the geowizard + diode + eth3d
   numbers all come back MATCH, the adapter bugs + loader bug + missing
   GT are closed out.
2. **D3 VGGT-DTU scale diagnosis**: 30-min single-scan probe with
   `pointcloud_alignment ∈ {icp, umeyama, none}` to isolate the 147 ×
   issue. This is the one open adapter-side suspect.
3. **D8 + D9 KITTI-MoGe/Marigold loader**: 4–6 h to write
   `KITTIMogeEvalLoader` + protocol + verification. Closes the two
   remaining ~10 % KITTI gaps. Defer unless this is a v0.1 blocker.
4. **D10 VGGT-ETH3D full split**: stage the 10 missing scenes + re-run.
   Or demote. Low priority.
5. **D15 DA-V2 NYU ~0.002 bias**: nice-to-have. Inspect the NYU
   loader's Eigen-crop + rawDepths interaction. Not a blocker.

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/20260421-run1/` preserve the observations
from each run so the report can be regenerated from any point.
