# Third-party notices

plumbline (Apache-2.0) bundles **vendored** source from the projects below under
`src/plumbline/_vendor/`. Each retains its own license (kept verbatim alongside
the code). Only permissively- and NonCommercial-licensed code is vendored; GPL/
copyleft code is **not** — those models stay clone+install. Vendoring code that
is NonCommercial makes the corresponding parts of this repository usable for
**non-commercial** purposes only.

| project | path | source | commit | license |
|---|---|---|---|---|
| DAGE | `_vendor/dage/` | https://github.com/ngoductuanlhp/DAGE | `a2c7901b34d9bca28667418149fbfbc6df9cd1cd` | CC BY-NC 4.0 (`_vendor/dage/LICENSE`) |

Vendored code is **unmodified** upstream source; plumbline's adaptations live in
the adapters (`src/plumbline/models/`). Runtime dependencies (torch, einops,
utils3d, kornia, …) and model weights are **not** vendored — they install / download
separately. Update a vendored copy by re-copying the package at a new pinned commit.
