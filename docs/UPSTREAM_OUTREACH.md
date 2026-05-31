# Upstream outreach — DA-V2 Table 2 / MoGe eval harness (2026-05-30)

Draft text for GitHub issues/comments. Post manually or via `gh` when credentials
allow. Link plumbline handoff: [`DA_V2_TABLE2_UPSTREAM_EVAL.md`](DA_V2_TABLE2_UPSTREAM_EVAL.md).

---

## MoGe — `read_meta` NameError in test dataloader

**Filed:** https://github.com/microsoft/MoGe/issues/153 (2026-05-30)

**Repo:** https://github.com/microsoft/MoGe  
**Suggested title:** `eval_baseline.py` fails: `read_meta` undefined in `moge/test/dataloader.py`

**Body:**

Running `moge/scripts/eval_baseline.py` with `configs/eval/benchmarks/diode.json`
and `baselines/da_v2.py` fails in worker threads:

```
File "moge/test/dataloader.py", line 96, in _load_instance
    meta = read_meta(Path(path, 'meta.json'))
NameError: name 'read_meta' is not defined
```

`moge/train/dataloader.py` line 128 correctly uses `read_json` for the same file.
`moge/utils/io.py` defines `read_json` but not `read_meta`.

**Fix:** replace `read_meta` → `read_json` in `moge/test/dataloader.py`.

**Environment note:** launching from a venv that installs PyPI package `pipeline`
can shadow MoGe's pipeline usage in `moge/test/dataloader.py` (`import pipeline`).
Running from a MoGe-only env avoids that.

---

## Depth Anything V2 — issue #280 (DIODE)

**Posted:** https://github.com/DepthAnything/Depth-Anything-V2/issues/280#issuecomment-4585219654

**URL:** https://github.com/DepthAnything/Depth-Anything-V2/issues/280

**Comment draft:**

We reproduced ~0.21 AbsRel on native DIODE val with devkit GT + `lstsq`
scale-and-shift (771 samples, ViT-L), vs paper Table 2 **0.066**. Indoor-only
split is ~0.072 (matches paper); outdoor native split ~0.33 drives the gap.

MoGe's public eval on the HF `monocular-geometry-evaluation` DIODE bundle
(1024×768 warp, disparity-affine-invariant + clamp) lands ~**0.05** AbsRel —
closer to paper but still ~15–18 % under Table 2 ViT-L 0.066. The DA-V2 repo
ships no zero-shot DIODE eval script.

Details: https://github.com/kmatzen/plumbline/blob/main/docs/DA_V2_TABLE2_UPSTREAM_EVAL.md
and https://github.com/kmatzen/plumbline/blob/main/docs/D29_DIODE_TABLE2_HANDOFF.md

Were Table 2 legacy-benchmark numbers computed on MoGe-style bundles (or DA-2K only)?

---

## Depth Anything V2 — issue #281 (ETH3D)

**Posted:** https://github.com/DepthAnything/Depth-Anything-V2/issues/281#issuecomment-4585219696

**URL:** https://github.com/DepthAnything/Depth-Anything-V2/issues/281

**Comment draft:**

Native ETH3D Table 2 repro (~0.09–0.10 AbsRel, 13 scenes, z-buffer GT @ 518)
is ~30–32 % **under** paper ViT-L **0.131** — same “reads better” pattern as #280.

MoGe Table 3 re-eval on the HF ETH3D bundle (2048×1365, segmentation mask)
reports ~**0.047** for DA-V2-L — a different protocol entirely.

After fixing RGB/GT alignment in our loader, metrics moved toward paper (harder
eval), but still do not match. No ETH3D eval script in the DA-V2 repo.

Handoff: https://github.com/kmatzen/plumbline/blob/main/docs/ETH3D_DAV2_TABLE2_HANDOFF.md

---

## Post commands

```bash
# MoGe new issue
gh issue create -R microsoft/MoGe \
  --title "eval_baseline.py: read_meta undefined in test dataloader" \
  --body-file docs/UPSTREAM_OUTREACH.md  # trim to MoGe section

# DA-V2 comments (requires issue access)
gh issue comment 280 -R DepthAnything/Depth-Anything-V2 --body "…"
gh issue comment 281 -R DepthAnything/Depth-Anything-V2 --body "…"
```
