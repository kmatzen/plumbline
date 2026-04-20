# Dataset footprints

Pre-computed disk-footprint estimates for each dataset × protocol combo
plumbline evaluates on. The point of this table is to make the GPU-box
disk budget **a check you do before rental**, not a surprise when
`tar` fails at 03:00 because KITTI raw blew past 65 GB.

> **Rule:** before rental, sum the footprints of all reproductions
> planned for the session and confirm
> `total + 20% headroom ≤ rental-box disk`. Over budget → cut or
> re-host before booking.

## Dataset-level footprints

Sizes here are **minimum viable** (sample-list-driven) vs **full
release**. The minimum column only counts the files the loader + sample
list actually touch.

| Dataset | Full release | Minimum viable | Sample-list artifact | Notes |
|---|---|---|---|---|
| **NYUv2** | ~3 GB | ~3 GB | `src/plumbline/datasets/_nyuv2_eigen_test.txt` (654 indices) | Single .mat; can't subset without custom repack |
| **KITTI raw + annotated** | ~65 GB raw + ~14 GB annotated depth | **~6 GB** (28 drives × ~24 frames × 3 file types + calib + annotated depth for listed frames) | `reproductions/kitti_eigen_benchmark_652.txt` | Use `scripts/fetch_kitti.py` (A7) to fetch only listed frames per drive |
| **DIODE val** | ~2.8 GB (bundled indoor+outdoor) | ~2.8 GB | (pending A6b) `reproductions/diode_val_{indoor,both}.samples.txt` | Val is bundled — no separate indoor/outdoor archive. Full val is the minimum. |
| **ETH3D high-res** | ~50 GB (all 13 train scenes) | ~8 GB (3 scenes: courtyard + delivery_area + facade; paper-protocol match) | Scene whitelist in YAML | `*_dslr_undistorted.7z` + `*_dslr_scan_eval.7z` per scene. A6c to pin window IDs. |
| **DTU MVS** | ~170 GB (all 128 scans) | ~7 GB (SampleSet.zip — 22 MVSNet test scans) | `DTU_MVS_TEST_SCANS` constant in `dtu.py` | SampleSet.zip from the official MVSNet Google Drive. |
| **Co3Dv2** | ~1.5 TB (19K sequences) | ~10 GB (pose-benchmark subset) | (pending A12 / Tier-2) sequence whitelist | Full set is untenable on a rental box. Must pin sequences before any download attempt. |
| **GSO** | ~2 GB (1030 objects, MoGe bundle) | ~2 GB | Full-set is the protocol | HuggingFace `Ruicheng/monocular-geometry-evaluation`, `GSO.zip`. |
| **7Scenes** | ~12 GB (all 7) | ~12 GB | Default test-split in `SevenScenesDataset` | Microsoft Research; 7 indoor RGB-D sequences. Loader shipped 2026-04-20. |
| **iBims-1** | ~40 MB (MoGe bundle; unzipped ~200 MB) | ~40 MB | MoGe-bundle format; all 100 scenes is the protocol | Loader shipped 2026-04-20 via MoGe's preprocessed HF bundle. Upstream TUM release also works if identically structured. |

Dropped (auth-gated, no longer on the v0.1 critical path):

| Dataset | Status |
|---|---|
| Sintel | Loader works on the public 5.3 GB `MPI-Sintel-complete.zip` (RGB only). Depth + camera archives need registration — email hadn't landed as of 2026-04-19. Substituted by GSO / iBims-1. |
| ScanNet v2 / ScanNet-1500 | ToS-gated. Loaders wired + unit-tested. Substituted by Co3Dv2 / 7Scenes. |

## Per-GPU-session pre-flight

Before booking a rental box:

1. List the reproductions planned for the session.
2. Group by dataset; take the max footprint per dataset (reproductions
   share data).
3. Sum, add 20% headroom for model weights (~5-10 GB aggregate) and
   HF cache.
4. Compare to rental-box disk.

### Example: Tier-1 KITTI + DIODE + DTU + ETH3D session

| Dataset | Footprint | |
|---|---|---|
| KITTI (Eigen-652) | 6 GB | |
| DIODE val (combined) | 2.8 GB | |
| DTU (MVSNet-22) | 7 GB | |
| ETH3D (3 scenes) | 8 GB | |
| NYUv2 | 3 GB | |
| GSO | 2 GB | |
| **Dataset total** | **~29 GB** | |
| Model weights + HF cache | ~15 GB | (DA-V2 variants + Metric3D ViT-g + MoGe + DA3 + Depth Pro + VGGT + Marigold) |
| Prediction cache | ~5 GB | (compressed npz per sample; grows with runs) |
| **Grand total** | **~50 GB** | |
| 20% headroom | ~10 GB | |
| **Required disk** | **~60 GB** | |

A standard RunPod / vast.ai box ships 100-200 GB of NVMe, so this fits.
The failure mode would be adding Co3Dv2 (10+ GB per subset) or trying
to stage the full ETH3D train split (~50 GB) — check the budget before
each session.

## Action items

- **A7** — Implement `scripts/fetch_<dataset>.py` for the datasets
  where "minimum viable" is meaningfully smaller than "full release"
  (KITTI and eventually Co3Dv2).
- **A8** — Implement a disk-budget gate script that reads a list of
  reproduction YAMLs and this footprint table, sums the requirements,
  and fails loud if over budget.
- Re-measure periodically. Minimum-viable sizes can drift as
  reproductions are added (e.g. more ETH3D scenes).
