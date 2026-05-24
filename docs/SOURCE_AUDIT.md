# Source-code fidelity audit (2026-05-23)

Audit of every plumbline model adapter against the **publicly released
upstream source code** for that method — checkpoint loading, image
preprocessing (resize/interpolation/normalization/crop/dtype), the forward
call, output extraction, and the convention/units conversion
(depth-vs-disparity, `world_from_camera` vs `camera_from_world`, intrinsics
space). Upstream code was read verbatim (raw GitHub / downloaded files);
every finding cites the upstream symbol and the local `file:line`.

This is a **source-fidelity** audit, separate from paper-cell reproduction
(`reproductions/AUDIT.md`) and from the upstream-blocked-numbers question
(`docs/DISCREPANCIES.md`). A faithful adapter can still fail a paper cell if
the released *checkpoint* doesn't match the paper (e.g. GeoWizard).

## Summary

| Adapter | Upstream | Net assessment | Key finding |
|---|---|---|---|
| depth-anything-v2 | DepthAnything/Depth-Anything-V2 | minor-divergence | paper-path interp constant `3`=INTER_AREA, upstream uses INTER_CUBIC (=2); HF path faithful |
| depth-anything-3 | ByteDance-Seed/Depth-Anything-3 | **needs-fix → fixed** | extrinsics were assumed (N,3,4) but upstream returns (N,4,4) (latent crash); DA3-LARGE is relative not metric |
| metric3d-v2 | YvanYin/Metric3D | faithful | canonical-camera de-scale uses *scaled* fx correctly; only sub-pixel resize-rounding deviation |
| moge | microsoft/MoGe | faithful | correct affine/metric split; only a 0.5px principal-point offset vs utils3d |
| vggt | facebookresearch/vggt | **faithful** | extrinsics correctly inverted camera_from_world→world_from_camera; autocast keeps fp32 weights |
| mast3r | naver/mast3r (+dust3r) | faithful (dust3r-GA) | `get_im_poses` is cam→world (correct); uses dust3r PCO not MASt3R sparse-GA (disclosed) |
| marigold | prs-eth/marigold (diffusers) | faithful | forward + depth-space alignment correct; defaults differ from paper protocol (exposed as kwargs) |
| geowizard | fuxiao0719/GeoWizard | faithful | seed_all/xformers/no-generator/domain all match; `--half_precision` is a dead upstream flag |
| depth-pro | apple/ml-depth-pro | faithful | metric depth + focal handling correct; never passes GT `f_px` (intended no-intrinsics mode) |
| pi3 | yyfz/Pi3 | minor-divergence → fixed | `conf` trailing-dim kept (fixed); no resize-to-14 + bf16-weights vs autocast (documented) |
| cut3r | CUT3R/CUT3R | faithful | transcribed from demo.py while building; depth/pose/pointmap conversions match |
| monst3r | Junyi42/monst3r | faithful (base only) | base dust3r alignment only; flow refinement intentionally out of scope (documented) |

## Fixes applied in this audit (2026-05-23)

- **depth-anything-3 — extrinsics shape (latent crash).** Upstream
  `output_processor.py` returns `extrinsics` as `(N,4,4)` in the default
  no-input-extrinsics path (`extrinsics.squeeze(0).cpu().numpy() # (N,4,4)`);
  the adapter assumed `(N,3,4)` and did `cam_from_world[:,:3,:] = extr_np`,
  which would `ValueError` on a 4x4. Now shape-tolerant (accepts 4x4 or 3x4).
- **depth-anything-3 — metric claim.** DA3-LARGE is **relative/affine-
  invariant** (README model table: "Rel. Depth" only; `output_processor`
  applies no metric conversion, `is_metric=0`). Changed `is_metric=True →
  False`, `alignment_hint "none" → "scale_shift"`, and the docstring. (The
  matched `da3-nyuv2` cell is unaffected — its YAML already sets
  `scale_alignment: scale_shift`.)
- **pi3 — confidence shape.** Upstream `conf` is `(N,H,W,1)` (Pi3
  `example.py`: `sigmoid(res['conf'][..., 0])`); the adapter kept the
  trailing dim, yielding `(N,H,W,1)`. Now drops it → `(N,H,W)`.
- **depth-anything-v2 — corrected the false `# cv2.INTER_CUBIC` comment**
  (the value `3` is `cv2.INTER_AREA`). Behavior intentionally unchanged —
  see deferred item below.
- **mast3r — retracted the unsubstantiated "(paper §4.3 default)" comment**
  on `schedule='linear'` (dust3r's documented default is `'cosine'`; both
  are valid). Clarified that N>=3 uses dust3r's PointCloudOptimizer, not
  MASt3R's own sparse-GA.

## Deferred (needs GPU re-validation or is a config choice, not a code bug)

- **depth-anything-v2 interpolation (affects 8 verified cells).** Upstream
  `image2tensor` uses `cv2.INTER_CUBIC` (=2); the adapter passes `3`
  (`cv2.INTER_AREA`). The 8 ✅ DA-V2 cells were validated with INTER_AREA.
  Switching to the faithful INTER_CUBIC is a sub-pixel resampling change
  that **requires GPU re-validation of the 8 DA-V2 cells** before landing —
  not changed blindly. (cv2 shrink quality actually favors INTER_AREA, so
  the cells likely still match, but that must be confirmed, not assumed.)
- **metric3d-v2 resize rounding.** Upstream truncates (`int(w*scale)`); the
  adapter rounds (`int(round(...))`). ≤1px difference; metric scale (the
  de-canonical factor) is unaffected. Fix optional, GPU-revalidate if changed.
- **moge principal-point.** Adapter scales `cx·W, cy·H`; utils3d's
  `denormalize_intrinsics('integer-center')` also subtracts 0.5. 0.5px
  offset, no metric impact.
- **pi3 preprocessing + dtype.** Adapter does no resize-to-multiple-of-14
  (upstream `load_images_as_tensor` LANCZOS-resizes under a ~255k-px budget)
  and casts weights to bf16 rather than fp32+autocast. pi3 has no validated
  cell yet; fold into the GPU bring-up. Use `dtype="float32"` for exact runs.
- **marigold / geowizard defaults.** Both default `dtype="float16"`; the
  released eval runs fp32 (GeoWizard's `--half_precision` is a dead flag).
  Both expose `dtype="float32"` and the reproduction YAMLs pin the paper
  protocol — config, not a code defect.
- **mast3r / monst3r N>=3 use dust3r's PointCloudOptimizer**, not the
  models' own sparse-GA / flow-refined alignment. Disclosed in both
  docstrings; a faithful sparse-GA / flow path is a v0.3 follow-up.

---

## depth-anything-v2

Upstream: `depth_anything_v2/dpt.py` (`image2tensor`, `infer_image`, `forward`),
`depth_anything_v2/util/transform.py`, `run.py`; HF checkpoint
`depth-anything/Depth-Anything-V2-*-hf` (`preprocessor_config.json`).

**Path A — `source="paper"` (paper `.pth` + repo DPT):** checkpoint filename,
model configs (features/out_channels per vits/vitb/vitl), resize
(`width=height=input_size=518, keep_aspect_ratio, ensure_multiple_of=14,
resize_method='lower_bound'`), normalization (`mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]`), `PrepareForNet`, forward, output resize-back, and
disparity→depth (`1/max(disp,EPS)`, `alignment_hint="scale_shift"`) all
**VERIFIED** against `dpt.py`/`run.py`. One **MISMATCH**:
`image_interpolation_method=3` (`depth_anything_v2.py:268`) is
`cv2.INTER_AREA`; upstream uses `cv2.INTER_CUBIC` (=2). The channel order is a
verified no-op (RGB→BGR→RGB round-trip cancels).

**Path B — `source="hf"` (transformers):** fully **VERIFIED** — delegates to
`AutoImageProcessor` (DPTImageProcessor: size 518, `keep_aspect_ratio`,
`ensure_multiple_of=14`, `resample:3`=PIL BICUBIC, ImageNet mean/std) and
`post_process_depth_estimation`. Relative variants → `1/disp` +
`scale_shift`; metric variants → meters + `none`. Both correct.

**Net:** minor-divergence — HF path faithful; paper path has the interpolation
constant divergence (deferred, see above).

## depth-anything-3

Upstream: `src/depth_anything_3/api.py`, `specs.py`,
`utils/io/output_processor.py`, README model table.

VERIFIED: checkpoint (`depth-anything/DA3-LARGE`), `DepthAnything3.from_pretrained`,
numpy-image input, default `process_res=504/upper_bound_resize`, side-effect-free
`inference(export_dir=None, export_format="mini_npz")`, field names
(`depth/intrinsics/extrinsics/conf`), intrinsics in processed pixel space, and
the w2c→`world_from_camera` inversion + view-0 rebase.

Two **MISMATCH** items, both **fixed** this audit:
1. **Extrinsics shape** — `output_processor.py` builds
   `extrinsics = extrinsics.squeeze(0).cpu().numpy() # (N,4,4)` in the default
   path; the adapter assumed `(N,3,4)` (`depth_anything_3.py:188`) → latent
   `ValueError`. Now shape-tolerant.
2. **Metric vs relative** — README marks DA3-LARGE "Rel. Depth" (no "Met.
   Depth"); `output_processor` does no metric conversion (`is_metric=0`). The
   adapter declared `is_metric=True`/`alignment_hint="none"`. Corrected to
   `is_metric=False`/`scale_shift`. (Note: DA3METRIC-LARGE / DA3NESTED-GIANT-
   LARGE are metric but are not shipped here.)

**Net:** needs-fix → fixed.

## metric3d-v2

Upstream: `hubconf.py` `__main__` demo (the canonical recipe).

**VERIFIED** end-to-end: torch.hub strings + `pretrain=True` (note the
`pretrain` not `pretrained` kwarg), canonical input size `(616,1064)`,
`scale=min(616/h,1064/w)`, intrinsic scaling, mean-colour padding
`[123.675,116.28,103.53]`, ImageNet-`[0,255]` mean/std (no `/255`),
`model.inference({'input': rgb})`, un-pad → bilinear upsample to native, and —
critically — the de-canonicalization `depth *= scaled_fx / 1000.0` using the
**scaled** fx (the most common place to get this wrong; correct here), then
`clamp(0,300)`. `requires_intrinsics=True`.

One **SUSPECT** (sub-pixel): adapter resizes with `int(round(w*scale))`,
upstream truncates `int(w*scale)` (`hubconf.py:161`). ≤1px; metric scale
unaffected.

**Net:** faithful.

## moge

Upstream: `moge/model/v1.py` (`MoGeModel.infer`/`forward`), utils3d, README.

**VERIFIED**: HF ids (`Ruicheng/moge-vitl`, `moge-2-*`), `[0,1]` CHW input
(DINOv2 normalization is internal to `forward`, adapter correctly does NOT
double-normalize), `.infer()` with upstream default `resolution_level=9`, depth
taken as the shift-corrected `points[...,2]` (not re-projected), point-map +
mask pass-through, and the v1-affine (`is_metric=False`,`scale_shift`) vs
v2-metric (`is_metric=True`,`none`) split.

One low-severity **MISMATCH**: upstream intrinsics are normalized; the adapter
correctly scales `fx·W, fy·H` but omits the `−0.5` half-pixel principal-point
offset that utils3d's `denormalize_intrinsics('integer-center')` applies. 0.5px,
no metric impact.

**Net:** faithful.

## vggt

Upstream: `vggt/utils/load_fn.py` (`load_and_preprocess_images`),
`vggt/utils/pose_enc.py` (`pose_encoding_to_extri_intri`), `vggt/models/vggt.py`.

**VERIFIED (fully faithful):** checkpoint `facebook/VGGT-1B`; preprocessing
(518 / BICUBIC / `ToTensor` ÷255 / center-crop on the tall axis / white-pad
`value=1.0` to max H,W for non-uniform batches); fp32 weights + `autocast`
(matches upstream — weights stay fp32); output keys
(`pose_enc/depth/depth_conf/world_points/world_points_conf`); depth from the
depth head `depth[0,...,0]`; intrinsics in *processed* pixel space (the
adapter passes the processed H,W to `pose_encoding_to_extri_intri`).
**Highest-risk item correct:** upstream `pose_encoding_to_extri_intri` docstring
says extrinsics are "camera from world" (world→cam); the adapter labels them
`camera_from_world`, pads 3x4→4x4, and **inverts** to `world_from_camera`,
with a guarded view-0 rebase.

**Net:** faithful — no fixes required.

## mast3r

Upstream: `naver/dust3r` (`utils/image.py`, `cloud_opt/*`, `image_pairs.py`),
`naver/mast3r`.

**VERIFIED:** checkpoint (`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`);
the in-memory `_images_to_dust3r_dicts` reproduces dust3r `load_images`
*exactly* for the 512 path (ImgNorm `(0.5,0.5,0.5)`, LANCZOS/BICUBIC threshold,
long-edge rounding, the `((2*cx)//16)*16/2` half-crop and `W==H→3:4` rule);
`make_pairs(scene_graph="complete", symmetrize=True)`; global alignment
`init='mst'/niter=300/lr=0.01`. **Highest-risk items correct:** PointCloudOptimizer
and PairViewer `get_im_poses()` both return **camera-to-world** (= plumbline
`world_from_camera`, no inversion); `get_depthmaps()` is camera-frame z-depth
(rebase-invariant); `get_pts3d()` is world-frame; view-0 rebase applied
consistently to poses + point map.

Two disclosed deviations: `schedule='linear'` (dust3r default is `'cosine'`;
both valid — the "paper default" comment was retracted this audit), and N>=3
uses dust3r's PointCloudOptimizer rather than MASt3R's own
`sparse_global_alignment` (v0.3 swap, already noted in the docstring).

**Net:** faithful to dust3r-GA on the MASt3R metric checkpoint.

## marigold

Upstream: `prs-eth/marigold` eval scripts + diffusers `MarigoldDepthPipeline`.

**VERIFIED:** checkpoint `prs-eth/marigold-depth-v1-1`; forward
`pipe(pil, num_inference_steps=4, ensemble_size=10, generator=...)` (matches the
diffusers "paper protocol" note: `ensemble_size=10`, `num_inference_steps=4`);
output is affine-invariant depth in `[0,1]` mapped to input resolution;
`align_depth_least_square` is in **depth space** → adapter's
`alignment_hint="scale_shift"` + `native_space="depth_affine_invariant"` correct.

Two default-value divergences from the prs-eth *paper-cell* protocol, both
exposed as kwargs (config, not bugs): adapter defaults `dtype="float16"` (paper
eval is fp32 via `--half_precision` store_true=False) and `processing_res=None`
(→768; prs-eth NYU/KITTI scripts use `--processing_res 0`/native). Set
`dtype="float32"` + `processing_res=0` for paper rows.

**Net:** faithful adapter.

## geowizard

Upstream: `geowizard/run_infer.py`, `geowizard/models/geowizard_pipeline.py`.

**VERIFIED (faithful to released code):** checkpoint `lemonaddie/Geowizard`;
forward args (`denoising_steps=10, ensemble_size=10, processing_res=768,
match_input_res=True, domain`); domain one-hot set
(`indoor/outdoor/object`); `seed_all` body (`random/np/torch/torch.cuda`, no
cudnn flags) + once-at-startup timing; xformers `try/except` gating; **no
`generator` kwarg** (upstream `__call__` doesn't accept one — adapter correctly
seeds global RNG instead); output is `[0,1]` affine-invariant **depth** (not
disparity) → `scale_shift_depth`.

One divergence: upstream's `--half_precision` is a **dead flag** (dtype never
reaches `from_pretrained`, so released GeoWizard always runs fp32); the adapter
honors `torch_dtype` and defaults to fp16. Set `dtype="float32"` for released-
code numerics (the GeoWizard YAMLs already pin fp32 — the D17/D18 fix).

**Net:** faithful adapter. (Paper cells remain upstream-blocked — D17/D18/D22 —
a checkpoint/eval issue, not an adapter-fidelity issue.)

## depth-pro

Upstream: `src/depth_pro/depth_pro.py` (`DepthPro.infer`, `create_model_and_transforms`),
`cli/run.py`.

**VERIFIED:** config via `dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,
checkpoint_uri=...)` (keeps the FOV head); transform `ToTensor` +
`Normalize(0.5,0.5,0.5)` (the 1536 resize happens *inside* `infer`); fp16
(matches CLI `precision=torch.half`); metric depth `1/clamp(inverse_depth)`
resized to native; intrinsics `fx=fy=focallength_px`, centered principal point,
original-pixel units (`focallength_px` is computed pre-resize on original W);
`is_metric=True`/`alignment_hint="none"`. Adapter never passes `f_px`, so Depth
Pro self-estimates focal — the intended **no-intrinsics metric** mode (matches
`requires_intrinsics=False`).

Minor: an inline comment mis-describes the transform as resizing (it's `infer`
that resizes); a redundant `.half()` after a float32-precision model create.
Both cosmetic.

**Net:** faithful adapter.

## cut3r

Upstream: `CUT3R/CUT3R` `demo.py` (`prepare_input`/`prepare_output`),
`src/dust3r/inference.py`, `src/dust3r/utils/camera.py`. Transcribed verbatim
from `demo.py` while building the adapter (2026-05-23).

**VERIFIED:** `ARCroco3DStereo.from_pretrained(<.pth>)`; the view dicts match
`prepare_input` (img `[-1,1]`, `ray_map` NaN `(1,6,H,W)`, identity
`camera_pose`, `img_mask/ray_mask/update/reset`); preprocessing replicates
dust3r `load_images` 512-branch; `inference(views, model, device)`; depth =
`pts3d_in_self_view[...,2]`; extrinsics = `pose_encoding_to_camera` →
camera-to-world (= `world_from_camera`), rebased to view 0; world point map =
`E[i] @ self_pts[i]` (consistent with depth+pose). Conversion logic covered by
`tests/test_cut3r.py` (mocked backend; caught a real SE(3) bug during build).

**Net:** faithful (GPU validation pending).

## monst3r

Upstream: `Junyi42/monst3r` `demo.py` (DUSt3R-family: `AsymmetricCroCo3DStereo`
+ dust3r `inference` + `global_aligner`).

**VERIFIED (base path):** `AsymmetricCroCo3DStereo.from_pretrained(<HF id>)`;
single-frame via the demo's duplicate trick; reuses MASt3R's audited
`_run_mast3r` (dust3r preprocessing + PointCloudOptimizer + view-0 rebase).
Conversion logic covered by `tests/test_monst3r.py`.

**Intentional scope limit (documented in the adapter):** MonST3R's full video
pipeline adds optical-flow consistency (`flow_loss_weight`), temporal
smoothing, motion masks, and window-wise alignment. This adapter runs the
**base** global alignment only (`flow_loss_weight=0` equivalent) — genuine
MonST3R per-view geometry with plain dust3r alignment, not the flow-refined
trajectory. The flow path is a GPU-validated v0.3 follow-up.

**Net:** faithful base inference; flow refinement out of scope (disclosed).
