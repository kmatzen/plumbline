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
| D3 | VGGT-DTU scene-aggregation OOMs (exit 137) on 22-scan chamfer | 🧪 loader fix landed; mem bug exposed 2026-04-23 |
| D4 | VGGT-ETH3D multiscene verify (needs D20 + scan_clean GT) | 🧪 pending GPU |
| D8 | MoGe-KITTI — structural protocol fix landed, observed 16 % off paper | 🧪 secondary-delta open |
| D9 | Marigold-KITTI — same protocol fix, observed 16 % off paper | 🧪 secondary-delta open |
| D10 | VGGT-ETH3D full 13-scene vs 3-scene subset | 📅 deferred |
| D17 | GeoWizard NYU 10 % off — RNG or secondary protocol detail | 🔎 suspected |
| D18 | GeoWizard KITTI — under new protocol, awaiting GPU verify | 🧪 pending GPU |
| D20 | Scene-aggregation chamfer OOM (mem bug, not perf) | 🧪 re-opened 2026-04-23 |

---

## Open issues

### D3 · VGGT-DTU scene-aggregation OOMs   🧪 MEM-BUG

Per-scene fix (aggregation=scene, 1 cm voxel) landed in `ad924e9` +
`2c140b3` and closed the protocol/alignment question (2026-04-22
scan1 probe: ICP 43 mm, Umeyama 81 mm; paper 0.382 mm). Blocks on a
new issue found 2026-04-23: scene-aggregation loads all 924 cached
predictions + full GT clouds into memory at once, OOM-killed (exit 137)
at ~28 GB RSS on a 31 GB box. Two runs, same symptom. Inference
completes; failure is in the aggregation / chamfer step.

Fix path: stream predictions per scene in the scene-merge loop instead
of holding all 22 scans' worth of predictions in memory. See
`src/plumbline/runner.py` scene-merge loop. Regression should include
a peak-RSS assertion.

### D4 · VGGT-ETH3D multiscene chamfer   🧪 PENDING GPU

`scan_clean/scan*.ply` GT fetched + pushed to S3 (`f89cdc4`). Loader
reads from it. GPU verification blocked on D20 (same OOM pattern
likely — ETH3D has fewer scenes so may fit; unverified).

### D8 · MoGe-KITTI — secondary protocol delta   🧪 OPEN

Structural fix landed: new `KITTIMogeEvalLoader` + `kitti_moge_eval`
protocol mirroring MoGe's own eval (HF-bundle 652 samples, no crop,
no 80 m clip, log-PNG depth, disparity-space LSQ + `1/gt.max()` floor).
`moge-vitl-kitti` opts in with `scale_shift_clamped`.

GPU verify 2026-04-23: observed AbsRel 0.0475 vs paper 0.0408 —
**16.4 % off, worse than the pre-fix 9.4 %.** Secondary delta remains:
either the disparity-clamp interaction is different for KITTI than
DIODE, or there's a protocol detail in the MoGe eval script we're
still missing. Candidates to investigate next:

- MoGe-specific resolution / padding quirk on the 750×375 bundle
- Whether the clamp should apply to the median or the max
- `drop_max_depth` applied to KITTI (MoGe's eval code is dataset-gated)

### D9 · Marigold-KITTI — same protocol delta   🧪 OPEN

Same loader + protocol as D8, but `scale_shift_depth` alignment
(Marigold fits in depth space per `eval.py`).

GPU verify 2026-04-23: observed AbsRel 0.1146 vs paper 0.099 —
**15.8 % off, similar to the pre-fix 10.1 %.** Parallels D8: the
new loader didn't close the gap. Likely a secondary knob specific to
Marigold's `eval.py` (e.g., `alignment_max_res`, ensemble-alignment
strategy).

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

### D18 · GeoWizard KITTI   🧪 PENDING GPU

Pre-fix: AbsRel 0.131 vs paper 0.097 (35.2 % off). `geowizard-kitti`
now points at the shared `kitti_moge_eval` protocol + loader with
`scale_shift_depth` alignment. Given D8/D9 both landed 16 % off under
the same protocol, expect similar secondary-delta behaviour.

### D20 · Scene-aggregation memory-bloat   🧪 RE-OPENED

Perf fix landed (`693f70c`) — once-per-scene ICP on fused+voxel-
downsampled prediction cloud instead of per-sample ICP. But the
2026-04-23 D3 re-run exposed a separate bug: the aggregation path
holds all per-sample predictions in memory simultaneously. DTU's
924 × 28 MB predictions ≈ 26 GB of point maps alone, plus 22 GT
clouds. Box ran out of RAM twice (exit 137, 22 GB RSS, 31 GB total).

Regression test in `tests/test_mvs_pipeline.py` asserts ICP-call
count but not peak RSS; it doesn't catch this.

Fix path: stream predictions per scene (load + merge + discard)
rather than building a full in-memory dict keyed by sample.

---

## Priorities for the next session

**Laptop-side prep (blocks GPU work):**
- D20 — stream predictions per scene in aggregation path. D3 and D4
  will OOM again without this.

**GPU-side verification (after D20 memory fix):**
1. D3 — VGGT-DTU chamfer under `aggregation: scene` on all 22 scans.
2. D4 — VGGT-ETH3D multiscene chamfer under the same path.

**Open investigations (not blocking paper-match queue):**
- D8 / D9 — root-cause the 16 % secondary delta on KITTI MoGe-eval
  protocol. Start with `drop_max_depth` applied to KITTI (likely
  dataset-gated in MoGe's eval code).
- D18 — GPU verify GeoWizard-KITTI once D8/D9 are understood.

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

---

## Rollback

Every fix in this doc is a single commit on `main`. `git revert <sha>`
cleanly reverts any individual change. `/tmp/results/` and
`s3://plumbline-bench/runs/<ts>/` preserve observations from each run
so a report can be regenerated from any point.
