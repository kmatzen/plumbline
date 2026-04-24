# Discrepancies

Catalog of every adapter / loader / protocol / solver / citation mismatch
surfaced to date. Open entries keep full diagnosis context. Closed entries
(FIXED + verified, or EXPLAINED-NOT-A-BUG) live at the bottom as one-liners
with commit SHAs; full history is in git.

Status legend:

- 🧪 **FIX-PENDING-VERIFY** — change landed, waiting on a GPU re-run.
- 🔎 **SUSPECTED** — hypothesis + diagnosis path; not yet reproduced.
- 📅 **DEFERRED** — known root cause, scoped for v0.2+.

## Open issues at a glance

| ID | One-liner | Status |
|---|---|---|
| D3 | VGGT-DTU chamfer 134× off paper (OOM fixed; correctness open) | 🔎 correctness |
| D4 | VGGT-ETH3D multiscene Overall 147 % off; completeness regressed 3–4× vs prior run on scan_clean GT swap | 🔎 correctness |
| D9 | Marigold-KITTI — OFF-PAPER under both candidate protocols (closest 13 % under kitti_moge_eval) | 🔎 secondary-delta |
| D10 | VGGT-ETH3D full 13-scene vs 3-scene subset | 📅 deferred |
| D17 | GeoWizard NYU 10 % off — RNG or secondary protocol detail | 🔎 suspected |
| D18 | GeoWizard-KITTI — OFF-PAPER under both protocols (closest 14 % under kitti_moge_eval, 45 % under marigold_kitti_eval) | 🔎 secondary-delta |
| D22 | Marigold/GeoWizard KITTI paper cells do not reproduce under either Marigold's own eval code or MoGe's bundle — paper likely uses a private eval config | 🔎 new 2026-04-24 |

---

## Open issues

### D3 · VGGT-DTU chamfer — correctness   🔎 OPEN

OOM issue fixed 2026-04-24 (`1fc0f9c`, scene_voxel_size unit-mixup:
DTU is mm, protocol had 0.01 copy-pasted from ETH3D's meters → 10 μm
cells, making per-chunk voxel_downsample a no-op). First clean GPU
run: Overall = **51.34 mm vs paper 0.382 mm (134× off)**, n=924,
per-scan Acc 6-19 mm, Comp 44-90 mm. Completeness is 10-15× Acc —
strongly suggests pred cloud covers a narrower region than GT scan;
per-sample camera_centers Umeyama + scene-level ICP may be mis-scaling
the scene.

YAML also has a metric-key mismatch: `primary_metric: chamfer`, but
the runner emits `accuracy` / `completeness` / `overall`. Match check
reports NaN (no chamfer key). Either rename `overall` → `chamfer` in
the runner or fix the YAML.

### D4 · VGGT-ETH3D multiscene — regression on scan_clean GT   🔎 OPEN

Overall 1.7512 m vs paper 0.709 (147 % off). YAML's prior-run note
had 0.8178 m (15 % off) under the `dslr_scan_eval` GT — today's
result on the new `scan_clean` GT is 2× worse. Per-scene
completeness degraded 3-4× (0.46 → 1.43 m mean). `scan_clean` GT
probably has broader point coverage than `dslr_scan_eval`, inflating
GT→pred NN distance. A/B the two GT sources on one scene to confirm.

### D9 · Marigold-KITTI — OFF-PAPER under both candidate protocols   🔎 OPEN

Tested under three protocols; paper value 0.099 doesn't reproduce under any:

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.1146 | +15.8 % |
| `kitti_moge_eval` | 0.0865 | −12.7 % |
| `marigold_kitti_eval` | 0.1179 | +19.1 % |

`marigold_kitti_eval` implements Marigold's own paper code (`kitti_bm_crop`
+ `valid_mask_crop: eigen` + `scale_shift_depth`, per
`prs-eth/Marigold/src/dataset/kitti_dataset.py`). That it's *further*
from paper than `kitti_moge_eval` means the paper cell didn't come
from Marigold's public eval pipeline — probably a private config or
different checkpoint.

YAML stays on `marigold_kitti_eval` (the literal paper-code pipeline)
per "never modify YAMLs to fit a number". Closing this requires finding
the paper's actual eval config — upstream issue, not a plumbline bug.

### D18 · GeoWizard-KITTI — same pattern as D9   🔎 OPEN

| Protocol | AbsRel | vs paper |
|---|---|---|
| `kitti_eigen_garg` (pre-session) | 0.131 | +35 % |
| `kitti_moge_eval` | 0.1103 | +13.7 % |
| `marigold_kitti_eval` | 0.1406 | +45 % |

Same as D9: `marigold_kitti_eval` is worse than `kitti_moge_eval`.
GeoWizard shares the diffusion-depth lineage with Marigold; D22
(paper-private-eval hypothesis) most likely applies to both.

### D22 · Marigold / GeoWizard KITTI paper cells don't reproduce   🔎 NEW 2026-04-24

Neither the literal paper-code pipeline (`marigold_kitti_eval`) nor
the MoGe bundle pipeline (`kitti_moge_eval`) reproduces
Marigold 0.099 or GeoWizard 0.097 on KITTI. Both are consistently
off by 13-45 % in various directions. Under `marigold_kitti_eval`
(the paper-code pipeline), the harness is *further* from paper than
under `kitti_moge_eval` — which rules out "we just need to use the
paper's code". Suggests the paper-reported cells come from a private
eval config (unreleased resolution setting, checkpoint version, or
pre-processing step).

Not paper-match-blocking in the sense that we can't close the gap —
it's a finding about the ground truth being recorded. Document and
move on until Marigold / GeoWizard authors clarify.

### D21 · Prediction cache doesn't invalidate on loader preprocessing change   🔎 NEW 2026-04-24

Cache key in `src/plumbline/runner.py` `_predict_with_cache` is
`(model.name, model.config_hash(), dataset_name, sample.sample_id)`.
It ignores the actual bytes / shape of the input tensor the loader
produces. Observed 2026-04-24: after porting MoGe's homographic warp
into `KITTIMogeEvalLoader`, a re-run of `moge-vitl-kitti`
cache-hit on the previous shard (1242×375 predictions) against the
new 750×375 GT, silently producing nonsense metrics (AbsRel 0.1895,
4 × the pre-fix value). Worked around by `rm -rf` of the stale
shard; a proper fix hashes the first-sample tensor shape + a small
byte sample into the cache key, or invalidates on `dataset.__class__`
fingerprint changes.

### D10 · VGGT-ETH3D 3-scene vs 13-scene split   📅 DEFERRED

Plumbline's YAML runs courtyard + delivery_area + facade (3 scenes);
paper's Table 3 Overall 0.709 is the 13-scene cross-scene mean. A
3-scene subset genuinely can't match the 13-scene aggregate.

Resolution: (a) stage remaining 10 scenes (+~14 GB data) and run full
split; (b) extract per-scene paper numbers from VGGT supplementary;
or (c) demote to informational with larger tolerance. Earlier audit
intended (c) — `tolerance_relative: 1.0` encoded that before the
repo-wide 5 % cap landed.

### D17 · GeoWizard NYU 10 % off   🔎 SUSPECTED

Observed `geowizard-nyuv2` AbsRel = 0.0573 vs paper 0.052 — 10.2 %
off, after D1 + D2 fixes. Candidates:

1. RNG divergence — plumbline seeds `torch.manual_seed(seed + idx)`
   per-sample; paper may use a single fixed seed.
2. Alignment mode — plumbline uses `scale_shift_depth`; GeoWizard's
   public eval script may differ.
3. Processing resolution — 768 matches paper; unlikely the source.

Priority: low. Defer to v0.2 with D9.

(D20 closed 2026-04-24, see bottom table.)

---

## Priorities for the next session

**Completed 2026-04-24:**
- D8 — MATCH (`KITTIMogeEvalLoader` delegates to MoGe's `_process_instance`).
- D20 — scene-agg memory bug fixed: eager per-chunk voxel_downsample (`8827a87`) + unit-mixup fix (`1fc0f9c`, DTU mm vs ETH3D m). D3 now completes without OOM.
- D21 — prediction cache key now mixes input-tensor fingerprint (`8827a87`).
- Marigold-style protocol + YAML repoint landed (`ce4183e`, `6d24c73`). Verify surfaced D22.

**Open correctness investigations:**
1. **D3 — VGGT-DTU chamfer Overall 134× off** (51 mm vs paper 0.382 mm). No OOM. Per-sample Acc 6-19 mm, Comp 44-90 mm — suggests scale / alignment issue, not just voxel density. Also fix YAML metric-key mismatch (`chamfer` vs emitted `overall`).
2. **D4 — VGGT-ETH3D regression vs prior run** (1.75 m vs prior 0.82 m, 2× worse). A/B `scan_clean` vs `dslr_scan_eval` GT to isolate.
3. **D22 — Marigold + GeoWizard KITTI paper cells** don't reproduce under either candidate protocol. Needs upstream clarification on paper's actual eval config.
4. **D17** — GeoWizard NYU 10 % off. Possibly same family as D22.

**Landed 2026-04-24:**
- D8 — MoGe-KITTI AbsRel 0.0404 vs paper 0.0408 (0.9 % off) ✅
  `KITTIMogeEvalLoader` now delegates to MoGe's own
  `EvalDataLoaderPipeline._process_instance` for the homographic
  FoV-crop. D9 + D18 share the same loader — D9 pending GPU verify,
  D18 pending xformers install + verify.
- D21 — prediction cache stale-hit bug exposed en route to D8;
  workaround is `rm -rf` of the shard before re-run. Proper fix
  deferred.

**Nice-to-have (v0.2):**
- D10 — VGGT-ETH3D 13-scene full split, or demote to informational.
- D17 — GeoWizard NYU 10 % off (RNG or alignment).
- D15 — DA-V2 NYU ~0.002 bias (Eigen-crop + rawDepths).

---

## Closed issues

One-line reference; full diagnosis in the linked commit message.

| ID | One-liner | Closed by |
|---|---|---|
| D1 | GeoWizard — `generator` kwarg not accepted upstream | ✅ `c50201e` |
| D2 | GeoWizard — upstream diffusers API drift (shim) | ✅ `a35c4f5` |
| D5 | DIODE outdoor prediction outliers (`drop_max_depth`) | ✅ `7fd6ff6` (residual → D19) |
| D6 | DIODE MoGe-eval loader `split` kwarg | ✅ `ae046ab` |
| D7 | KITTI annotated-depth not in S3 cache | ✅ (staged to S3) |
| D11 | `scale_shift_robust` overfits NYU vs MoGe's plain LSQ | 📝 `c14d776` |
| D12 | KITTI Eigen-crop hypothesis (rejected empirically) | 📝 `fb58b90` |
| D13 | DA-V2 Large NYU — pinned to MoGe's 0.0420 | ✅ `58fc159` |
| D14 | DA-V2 Base NYU citation verified 0.049 | ✅ `603e717` |
| D15 | DA-V2 NYU ~0.002 AbsRel systematic downshift (S/B/L) | 📝 below-threshold |
| D16 | MoGe-DIODE-indoor combined-val citation demoted | ✅ `603e717` |
| D19 | MoGe-DIODE-both `scale_shift_clamped` alignment | ✅ 2026-04-23 verify: 0.0406 vs paper 0.0400 (1.5 % off) |
| D8 | MoGe-KITTI — port MoGe's homographic FoV-crop to `KITTIMogeEvalLoader` | ✅ 2026-04-24 verify: 0.0404 vs paper 0.0408 (0.9 % off) |
| D20 | Scene-aggregation memory bloat — eager per-chunk voxel_downsample + DTU voxel_size unit fix | ✅ `8827a87` + `1fc0f9c`: D3 completes without OOM (51 mm, 8 GB peak RSS vs 28 GB prior) |
| D21 | Prediction cache key → stale hits on loader preprocessing change — fingerprint input tensor | ✅ `8827a87`, regression test `test_input_fingerprint_invalidates_on_change` |

---

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/<ts>/` preserve observations from each run
so a report can be regenerated from any point.
