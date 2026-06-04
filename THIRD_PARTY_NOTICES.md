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
| Depth Anything 3 | `_vendor/depth_anything_3/` | PyPI `depth-anything-3` (ByteDance Seed) | Apache-2.0 (`_vendor/depth_anything_3/LICENSE`) |
| UniK3D | `_vendor/unik3d/` | https://github.com/lpiccinelli-eth/UniK3D | CC BY-NC-SA 4.0 (`_vendor/unik3d/LICENSE`) |
| Video Depth Anything | `_vendor/vda/` | https://github.com/DepthAnything/Video-Depth-Anything | Apache-2.0 code (`_vendor/vda/LICENSE`); Base/Large *weights* CC BY-NC 4.0 |
| π³ (Pi3) | `_vendor/pi3/` | https://github.com/yyfz/Pi3 | BSD-3-Clause (`_vendor/pi3/LICENSE`) |
| StreamVGGT | `_vendor/streamvggt/` | https://github.com/wzzheng/StreamVGGT | CC BY-NC-SA 4.0 (`_vendor/streamvggt/LICENSE.txt`) |

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

Vendored code is upstream source (apart from the noted build-compat patches);
plumbline's adaptations live in the adapters (`src/plumbline/models/`). Runtime
dependencies (torch, einops, …) and model weights are **not** vendored — they
install / download separately.
