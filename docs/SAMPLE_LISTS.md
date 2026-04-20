# Sample-list inventory

Every reproduction in `reproductions/*.yaml` selects some subset of a
dataset. The mechanism matters: if the subset isn't pinned in-repo, two
machines can silently disagree on which samples were evaluated, and the
numbers are not comparable.

## Legend

- **IN-REPO** — sample selection is deterministic from files committed
  to this repository (YAML + sample-list file + loader defaults).
- **HOST-ONLY** — selection depends on a file that lives under the
  dataset root on the eval host. Two hosts with different copies of that
  file silently produce different numbers.
- **LOADER-DEFAULT** — selection is hard-coded in a loader constant
  (`DTU_MVS_TEST_SCANS`, etc.). Counts as in-repo.
- **SCENE-FILTER** — YAML lists `scenes: [...]`. Deterministic, but
  sample count per scene depends on the on-disk scene size.
- **UNPINNED** — no explicit sample selection; reproduction iterates
  whatever the loader finds on disk.

## Inventory

| Reproduction | Dataset | Mechanism | Committed artifact | Status |
|---|---|---|---|---|
| `da-v2-{small,base,large}-nyuv2` | NYUv2 | Loader default (Eigen 654 indices) | `src/plumbline/datasets/_nyuv2_eigen_test.txt` | **IN-REPO** |
| `da-v2-metric-indoor-large-nyuv2` | NYUv2 | Loader default | same | **IN-REPO** |
| `metric3d-v2-{L,giant}-nyuv2` | NYUv2 | Loader default | same | **IN-REPO** |
| `moge{,2}-vitl-nyuv2{,-metric}` | NYUv2 | Loader default | same | **IN-REPO** |
| `da3-nyuv2` | NYUv2 | Loader default | same | **IN-REPO** |
| `marigold-v1-1-nyuv2` | NYUv2 | Loader default | same | **IN-REPO** |
| `depth-pro-nyuv2` | NYUv2 | Loader default | same | **IN-REPO** |
| `da-v2-small-kitti` | KITTI | `sample_list: kitti_eigen_benchmark_652.txt` (in-repo) | `reproductions/kitti_eigen_benchmark_652.txt` | **IN-REPO** |
| `da-v2-{base,large}-kitti` | KITTI | same | same | **IN-REPO** |
| `da-v2-metric-outdoor-large-kitti` | KITTI | same | same | **IN-REPO** |
| `metric3d-v2{,-giant}-kitti` | KITTI | same | same | **IN-REPO** |
| `moge{,2}-vitl-kitti` | KITTI | same | same | **IN-REPO** |
| `marigold-v1-1-kitti` | KITTI | same | same | **IN-REPO** |
| `depth-pro-kitti` | KITTI | same | same | **IN-REPO** |
| `da-v2-small-diode-indoor` | DIODE | `sample_ids_file: diode_val_indoor.samples.txt` | `reproductions/diode_val_indoor.samples.txt` (220 IDs) | **IN-REPO** |
| `moge-vitl-diode-indoor` | DIODE | same | same | **IN-REPO** |
| `moge-vitl-diode-both` | DIODE | `sample_ids_file: diode_val_both.samples.txt` | `reproductions/diode_val_both.samples.txt` (612 IDs) | **IN-REPO** |
| `vggt-eth3d-courtyard-chamfer` | ETH3D | `scenes: [courtyard]` | YAML | **SCENE-FILTER** |
| `da3-eth3d-courtyard-chamfer` | ETH3D | same | YAML | **SCENE-FILTER** |
| `vggt-eth3d-multiscene-chamfer` | ETH3D | `scenes: [courtyard, delivery_area, facade]` | YAML | **SCENE-FILTER** |
| `vggt-paper-dtu-mvs` | DTU | Loader default (22 MVSNet test scans) | `DTU_MVS_TEST_SCANS` in `dtu.py` | **LOADER-DEFAULT** |
| `{da-v2,da3,moge}-*-gso` | GSO | Full 1030 objects | — | **UNPINNED** (full set is the protocol) |
| `depth-anything-v2-sintel` | Sintel | `pass_name: final` | YAML | **UNPINNED** (all frames in `final/`) |

## Action items (Phase A)

### 1. Commit `kitti_eigen_benchmark_652.txt` to the repo — **LANDED 2026-04-20**

The canonical 652-frame list (Monodepth2
`splits/eigen_benchmark/test_files.txt`) is now committed at
`reproductions/kitti_eigen_benchmark_652.txt`. All 10 KITTI
reproductions reference it; the KITTI loader resolves relative
`sample_list` against the in-repo `reproductions/` dir first and
falls back to `$KITTI_ROOT` for backward compat. 28 drives, 12–25
frames each (histogram: 11×24, 7×23, 6×25, 2×22, 1×21, 1×12).
Covered by `test_sample_list_relative_path_prefers_repo_reproductions_dir`.

### 2. Pin DIODE sample IDs per reproduction — **LANDED 2026-04-20**

Committed:
- `reproductions/diode_val_indoor.samples.txt` (220 IDs)
- `reproductions/diode_val_both.samples.txt` (612 IDs)

Provenance: `diode-dataset/diode-devkit@21b77612/diode_meta.json`
(2019-08-02). DIODE YAMLs now carry `sample_ids_file:` pointing at
these; the runner resolves the path against `REPRODUCTIONS_DIR` and
hands the IDs to `Dataset.subset_by_ids()` which raises loud if any
listed ID isn't found on the host's archive.

Caveat: the S3 archive (`val.tar.gz`, Last-Modified 2019-08-01) was
last modified one day before the devkit file was generated and
appears to contain ~2% more captures per scan than `diode_meta.json`
enumerates (spot-check: 52 devkit vs 53 archive basenames on one
scan). The committed list favours the devkit's authoritative
enumeration — reproductions against this list are deterministic
across hosts; the handful of "extra" captures on S3 are not
evaluated. Earlier internal notes cited counts of 325 indoor / 771
combined from a historical plumbline scan; those counts were
inflated and are superseded by the 220 / 612 devkit numbers.

### 3. ETH3D window determinism — **AUDITED 2026-04-20, no action needed**

`src/plumbline/datasets/eth3d.py::_scan()` is fully deterministic
given `(scene, views_per_sample)`:

- Scenes are read in sorted order (`sorted(p for p in self.root.iterdir())`).
- Images are sorted by COLMAP `image_id` (ascending). The `image_id` set
  and order are defined by the ETH3D-shipped `images.txt`, not the host
  filesystem.
- Windows are sliding, stride=1:
  `for i in range(0, len(ordered) - views_per_sample + 1): group = ordered[i:i+vps]`
- Sample IDs are `"{scene}/{first_image_id:06d}_v{views_per_sample}"`,
  stable across hosts.

No randomness, no filesystem-order dependence. Two hosts with the same
ETH3D undistorted archive will iterate identical samples in identical
order. No sample list needs to be committed; SCENE-FILTER + fixed
`views_per_sample` is a complete spec.

### 4. No action needed (IN-REPO today)

- All NYUv2 reproductions (the `_nyuv2_eigen_test.txt` file in the
  dataset package is authoritative).
- DTU (`DTU_MVS_TEST_SCANS` is a python constant; any change is a code
  review).
- GSO (full-set is the protocol).

## Principle going forward

New reproduction YAMLs must satisfy **one** of:

1. The loader has a hard-coded default and the YAML accepts it (NYU, DTU).
2. The YAML references a sample-list file committed **inside the repo**
   (not under the data root).
3. The YAML is explicitly marked `status: unpinned` in
   `REPRODUCTIONS.md` and nobody counts its number as a paper-match.

No new reproduction may depend on a file living under `$<DATASET>_ROOT`
for sample selection.
