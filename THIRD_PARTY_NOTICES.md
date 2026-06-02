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

These DUSt3R-lineage models each bundle their own `dust3r` + Naver `croco`
subtrees (MASt3R also `mast3r/`; MonST3R also `third_party/RAFT` for flow). All
embed Naver CroCo (CC BY-NC-SA). The `curope` CUDA RoPE extension is vendored as
**source only**: required for **CUT3R** (its pure-torch RoPE fallback
device-asserts — build it via `plumbline install cut3r`, with a one-line torch≥2.5
`tokens.type()`→`tokens.scalar_type()` patch to `kernels.cu`), and an optional
speedup for DUSt3R/MASt3R/MonST3R (pure-torch RoPE works). Build artifacts
(`*.so`, `build/`) are gitignored. MonST3R's `sam2` import is shimmed in the
adapter (import-only), so sam2 is **not** vendored.

Vendored code is upstream source (apart from the noted build-compat patch);
plumbline's adaptations live in the adapters (`src/plumbline/models/`). Runtime
dependencies (torch, einops, …) and model weights are **not** vendored — they
install / download separately.
