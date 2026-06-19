# Third-party notices

plumbline (Apache-2.0) bundles **vendored** source from the projects below under
`src/plumbline/_vendor/`. Each retains its own license (kept verbatim alongside
the code). Only permissively- and NonCommercial-licensed code is vendored; GPL/
copyleft code is **not** — those models stay clone+install. Vendoring code that
is NonCommercial makes the corresponding parts of this repository usable for
**non-commercial** purposes only.

| project | path | source | license |
|---|---|---|---|
| DAGE | `_vendor/dage/` | https://github.com/ngoductuanlhp/DAGE | CC BY-NC 4.0 (`_vendor/dage/LICENSE`) |
| CUT3R | `_vendor/cut3r/` | https://github.com/CUT3R/CUT3R | CC BY-NC-SA 4.0 (`_vendor/cut3r/LICENSE`) |
| DUSt3R | `_vendor/dust3r/` | https://github.com/naver/dust3r | CC BY-NC-SA 4.0 (`_vendor/dust3r/LICENSE`) |
| MASt3R | `_vendor/mast3r/` | https://github.com/naver/mast3r | CC BY-NC-SA 4.0 (`_vendor/mast3r/LICENSE`) |
| MonST3R | `_vendor/monst3r/` | https://github.com/Junyi42/monst3r | CC BY-NC-SA 4.0 (`_vendor/monst3r/LICENSE`) |
| VGGT | `_vendor/vggt/` | https://github.com/facebookresearch/vggt | "VGGT License" (Meta research license — redistribution permitted under the agreement bundled at `_vendor/vggt/LICENSE.txt`; non-commercial research). Pose/depth inference subset; the heavy vggsfm/pycolmap `dependency/` tree is present but lazy (not imported for pose/depth) |
| Depth Anything 3 | `_vendor/depth_anything_3/` | PyPI `depth-anything-3` (ByteDance Seed) | Apache-2.0 (`_vendor/depth_anything_3/LICENSE`) |
| Depth Anything V2 | `_vendor/depth_anything_v2/` | https://github.com/DepthAnything/Depth-Anything-V2 | Apache-2.0 code (`_vendor/depth_anything_v2/LICENSE`); Small weights Apache-2.0, Base/Large weights CC BY-NC 4.0 |
| MoGe (1 & 2) | `_vendor/moge/` | https://github.com/microsoft/MoGe | MIT (`_vendor/moge/LICENSE`); inference subset (model/ + utils + test/dataloader for the MoGe-eval homographic warp). Bundles its pinned `utils3d` @3fab839f at `_vendor/moge/utils3d/` (MIT, `_vendor/moge/UTILS3D_LICENSE`) — the `utils3d.pt` API MoGe needs, distinct from the older `_vendor/utils3d` used by DAGE |
| pipeline *(lib)* | `_vendor/moge/pipeline/` | https://github.com/EasternJournalist/pipeline | MIT (`_vendor/moge/PIPELINE_LICENSE`); pure-stdlib parallel-dataloading lib that MoGe's `test/dataloader` eval pipeline depends on |
| UniK3D | `_vendor/unik3d/` | https://github.com/lpiccinelli-eth/UniK3D | CC BY-NC-SA 4.0 (`_vendor/unik3d/LICENSE`) |
| Video Depth Anything | `_vendor/vda/` | https://github.com/DepthAnything/Video-Depth-Anything | Apache-2.0 code (`_vendor/vda/LICENSE`); Base/Large *weights* CC BY-NC 4.0 |
| π³ (Pi3) | `_vendor/pi3/` | https://github.com/yyfz/Pi3 | BSD-3-Clause (`_vendor/pi3/LICENSE`) |
| StreamVGGT | `_vendor/streamvggt/` | https://github.com/wzzheng/StreamVGGT | CC BY-NC-SA 4.0 (`_vendor/streamvggt/LICENSE.txt`) |
| utils3d *(lib)* | `_vendor/utils3d/` | https://github.com/EasternJournalist/utils3d | MIT (`_vendor/utils3d/LICENSE`) |

These DUSt3R-lineage models each bundle their own `dust3r` + Naver `croco`
subtrees (MASt3R also `mast3r/`; MonST3R also `third_party/RAFT` for flow). All
embed Naver CroCo (CC BY-NC-SA). The `curope` CUDA RoPE extension is vendored as
**source only**: required for **CUT3R** (its pure-torch RoPE fallback
device-asserts — build it via `plumbline install cut3r`, with a one-line torch≥2.5
`tokens.type()`→`tokens.scalar_type()` patch to `kernels.cu`), and an optional
speedup for DUSt3R/MASt3R/MonST3R (pure-torch RoPE works). Build artifacts
(`*.so`, `build/`) are gitignored. MonST3R's `sam2` import is shimmed in the
adapter (import-only), so sam2 is **not** vendored.

**Depth Anything 3** is **Apache-2.0** (permissive, *not* NonCommercial) — it
adds no commercial-use restriction. Only the **mono-depth subset** of the
package is vendored: the `bench/`, `app/` (gradio), `services/` (fastapi) and
`utils/export/` (gsplat/colmap/trimesh export) trees are pruned, so the hostile
deps they carry (`numpy<2`, `xformers`, `gsplat`, `pycolmap`, `moviepy`) are not
pulled in. The slice's three runtime deps (`addict`, `omegaconf`,
`opencv-python`) live in the base install; `xformers`/`gsplat`/`e3nn` are
guarded-optional in the retained code and `evo` is lazy-imported, so none are
required. One local patch: `api.py`'s top-level `utils.export` import is made
lazy (the pruned tree is only reached on an explicit export call).

**utils3d** is a small MIT geometry library (`_vendor/utils3d/`), not a model —
DAGE's bundled MoGe code imports it (`utils3d.torch` / `utils3d.numpy` intrinsics
/ depth-unprojection helpers). It is vendored, frozen at the **0.0.2 / commit
`3913c65`** that DAGE needs (later releases break `get_intrinsics`), because the
exact-commit pin can only be expressed as a PEP 508 git direct reference, which
PyPI rejects in a published wheel — vendoring the 28-file pure-Python tree keeps
`plumbline-bench` publishable. Only the geometry subset is reached; its optional
`moderngl`/`glcontext` rasterization deps are lazy-imported and never loaded on
DAGE's path. The DAGE adapter puts `_vendor/` on `sys.path` so `import utils3d`
resolves the vendored copy; `$UTILS3D_ROOT` overrides for a dev checkout.

Vendored code is upstream source (apart from the noted build-compat patches);
plumbline's adaptations live in the adapters (`src/plumbline/models/`). Runtime
dependencies (torch, einops, …) and model weights are **not** vendored — they
install / download separately.
