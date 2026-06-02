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

CUT3R bundles its vendored `src/dust3r` (a DUSt3R fork) and `src/croco` (Naver
CroCo, CC BY-NC-SA) subtrees. Its `curope` CUDA RoPE extension is vendored as
**source only**; it must be compiled per environment (`plumbline install cut3r`)
— the build artifacts (`*.so`, `build/`) are gitignored. A one-line torch≥2.5
patch (`tokens.type()` → `tokens.scalar_type()`) is applied to `curope/kernels.cu`.

Vendored code is upstream source (apart from the noted build-compat patch);
plumbline's adaptations live in the adapters (`src/plumbline/models/`). Runtime
dependencies (torch, einops, …) and model weights are **not** vendored — they
install / download separately.
