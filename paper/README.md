# Plumbline paper

This is an incomplete research draft about camera evaluation. The reproduction
ledger is supporting infrastructure. Real tracking comparisons and independent
judgments are still required before submission.

The original 92.9% versus 79.6% comparison is withdrawn: the cross-family defect
order was an arbitrary design choice, not an independently established target.
`controlled_evaluator.json` and `method_validation.png` are retained historical
exploratory outputs and must not support paper claims. The replacement sensitivity
experiment compares amplitudes only within each family and includes standard RPE.
The synthetic search experiment tests the optimizer, not tracking improvements.

## Real-data pilot

### Independent point annotations for identity validation

#### Frozen first-visible LK identity check

`probe_tapvid_identity.py` runs native-resolution, independently queried LK
tracks from each annotation's first visible point. Later annotation positions
and visibility enter only scoring, after the full candidate track is computed.
The baseline does not restart after an LK/out-of-image failure. Four rules are
retained: any available estimate; backward round-trip error at most one native
pixel; starting-patch grayscale NCC at least 0.8; and both checks. NCC uses a
17×17 subpixel patch without rotation/scale alignment; borders and textureless
patches cannot pass. Correctness requires visibility and Euclidean error below
four pixels after coordinate scaling to 256×256. This is not staticness.

On the ten development videos, macro accepted precision is 54.7%, 62.4%, 96.1%,
and 97.3%, respectively. The joint rule's macro visible-correct recall is 24.8%.
All 215 queries and their 11,857 later observations are retained. Raw records and
the exact executed development runner are in `results/tapvid-davis-v1/`.

Before opening held-out arrays, freeze all four rules without threshold changes.
The joint rule is the primary selected check; all three alternatives remain
reported. Primary measurements are per-video accepted precision and
visible-correct recall, averaged across the 20 held-out videos. Also report
acceptance coverage and counts of accepted occluded points. Use 20,000 video-level
bootstrap resamples with seed 677 for descriptive percentile intervals; never
resample individual point frames as independent evidence. A video with zero
accepted points has undefined precision, is reported explicitly, and prevents
an unconditional macro-precision estimate. The script defaults to development;
held-out evaluation requires `--split held_out`. No held-out results have been
examined at protocol-freeze commit `3b57b0e`. This validation does not authorize a camera-DNF
claim and does not assess a VLM or a learned identity verifier.

**Secondary native-pixel / whole-track audit.** After the primary result,
`audit_tapvid_native_support.py` replays exactly the same joint acceptance rule
against annotation coordinates in native pixels. This is a descriptive secondary
analysis, not a new preregistered endpoint or threshold-selection exercise. It
checks visibility and both coordinate residuals against the existing geometry
tolerances, including the boundary (`<=`). Whole-track agreement requires every
accepted later observation to meet that condition; the denominator includes
tracks with at least two accepted later observations. It does not establish that
those observations are widely separated, static, or sufficient for triangulation.

| Native box tolerance | Held-out macro accepted-observation agreement | Held-out macro whole-track agreement |
|---|---:|---:|
| 1 px | 50.7% | 28.7% |
| 2 px | 74.1% | 57.1% |
| 4 px | 88.6% | 75.7% |
| 8 px | 95.6% | 87.1% |

The 435 held-out tracks include 74 with zero accepted later observations, 47
with one, and 314 with at least two. These categories remain explicit; an empty
track is not declared reliable. At four native pixels, 3,771/4,245 accepted
observations and 234/314 eligible tracks agree with the annotations (pooled
descriptive counts, different from the video-macro percentages above). Fifteen
accepted observations are annotated occluded and fail every tolerance. Development
and held-out audit records preserve all accepted coordinates' residuals, visibility,
and frame lag in `results/tapvid-davis-v1/native-support-*-v1.json`.

The earlier 96.2% accepted precision at 256-scale does not certify tight native
boxes, and per-observation agreement does not establish simultaneous whole-track
agreement. Annotation uncertainty also contributes to measured residuals. Do not
use these image checks to assert error-free geometric constraints or to issue
camera-DNF labels. No acceptance threshold, geometric tolerance, or camera verdict
was changed after this audit. The manuscript has not been modified with these
development findings.

The frozen held-out run is now complete. No rule or threshold changed after
development. All 435 reserved queries and their 26,374 later observations are
retained. Exact executed-runner hashes, sequence split membership, initialization
coordinates, and all 38,231 development-plus-held-out scoring records were
independently checked against the staged annotations.

| Frozen rule | Held-out macro accepted precision | Held-out macro visible-correct recall |
|---|---:|---:|
| LK estimate available | 49.3% | 53.0% |
| Forward–backward error ≤1 native px | 53.3% | 50.9% |
| Starting-patch NCC ≥0.8 | 95.2% | 24.7% |
| Both checks (primary) | 96.2% | 24.0% |

For the primary rule, the video-bootstrap 95% intervals are 94.3–97.8% for
precision and 16.6–33.0% for visible-correct recall. Every held-out video has at
least one accepted estimate. Pooled descriptive counts are 4,245 accepted:
4,133 within tolerance and visible, 97 outside tolerance while visible, and 15
occluded. Macro averages give each video equal weight; do not confuse them with
ratios of these pooled counts. Raw results and deterministic sequence-bootstrap
summaries are in `results/tapvid-davis-v1/identity-*-v1.json[.gz]`, generated by
`analyze_tapvid_identity.py`. The source-patch rule is a classical baseline,
not the novel contribution. This completed correspondence experiment and its
coverage limitation are now included in the paper; unfinished model-training
results and development-failure narratives are not.

This measures spatial agreement with reviewed point annotations, not an exact
physical-identity certificate. Four pixels at the 256×256 scoring scale is a
larger displacement than four native pixels in these videos; this experiment
does not validate the tighter native-pixel frustum bounds. Accepted points may
also be moving. Remaining requirements include matching annotation/measurement
uncertainty to geometric tolerances and validating staticness and camera-level
decisions. The held-out result must not be reused for threshold tuning.

The official [TAP-Vid DAVIS release](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/README.md)
is staged at `/data3/kmatzen/plumbline/research/tapvid-davis-v1/staged` on Anima.
It contains 30 videos, 1,999 frames, and 650 annotated tracks. These point
positions and visibility flags were produced with human review and optical-flow
assistance. They supply external position/visibility references for evaluating
track checks; they do not label static scene points, camera accuracy, or DNF.
They are not perfect independent hand clicks on every frame.

`stage_tapvid_davis.py` pins the official archive SHA-256
`60eb6239c57a9877900c269d14637c2b6eece8fa280f59e4592324c6392031b6`
and byte count 1,668,710,491. Its MD5 also matched the Google Storage response.
The loader checks the archive before reading its NumPy pickle through a restricted
constructor allowlist. Subsequent consumers use per-video NPZ files with
`allow_pickle=False`. The complete manifest and release license/source notes are
retained in `results/tapvid-davis-v1/`; source videos remain on Anima, not in Git.
All 30 NPZ archives and all 90 decoded array hashes were checked after staging.

The split was fixed before evaluating an identity checker: sort video names by
SHA-256 of `plumbline-track-identity-v1:` followed by the name, assign the first
ten videos to development, and reserve the other twenty. The split is by video,
not frame or track. Development has 633 frames and 215 tracks; held-out has
1,366 frames and 435 tracks. At staging, no held-out metric had been computed. Thresholds
and presentation choices must be fixed using development only; held-out data
must not be used for fine-tuning. This is a project-specific validation split,
not the standard full TAP-Vid benchmark, and prior exposure of pretrained models
to these public videos is unknown.

Keep the original normalized `(x,y)` coordinates. Raster conversion multiplies
by `(width,height)`, following the pinned upstream reader, not by `(width-1,height-1)`.
All stored videos have height 480, but widths vary (854, 910, and 1152); do not
assume a common width. Occluded points must not become positive visible-identity
examples. There are 28,824 visible and 13,376 occluded observations, no entirely
occluded tracks, and no visible coordinates outside the unit square. No tracks
or observations were filtered during staging.

Next use: evaluate image-only identity checks against these annotations before
allowing those checks to admit tracks into a camera-consistency decision. Source
coordinates may initialize a tracking query; later annotation coordinates and
visibility are scoring targets, not checker inputs. This validates correspondence
evidence only. Staticness and camera-failure attribution still require separate
validation. No performance claim or manuscript result follows from staging data.

### Camera-versus-track intervention control

`probe_consistency_attribution.py` tests whether counts of contradictory tracks
identify the source of an error. It constructs ten seeded ideal pinhole scenes,
each with 24 frames and 12 static points at depths 3–6. The camera translates
0.8 units horizontally. Coordinates have independently sampled bounded noise of
0.25 pixels per axis. The first 12 frames fit the point; all remaining frames
are jointly checked at 1- and 2-pixel tolerances. The evaluator receives only
candidate cameras, tracks, and calibration, never the intervention label or
generating 3D points. This is an analytic control, not a rendered video benchmark.

All seven conditions and ten seeds are retained in
`results/consistency-attribution-v1.json.gz` (70 input cases, 140 evaluations):

| Intervention | Contradictory tracks at 1 px | Contradictory tracks at 2 px |
|---|---:|---:|
| Clean inputs | 0/120 | 0/120 |
| Camera yaw ramp ending at 1° | 120/120 | 114/120 |
| Camera yaw ramp ending at 5° | 120/120 | 120/120 |
| One track jumps vertically by 16 px after fitting | 10/120 | 10/120 |
| Three tracks jump vertically by 16 px after fitting | 30/120 | 30/120 |
| All query tracks share a 5° erroneous coordinate warp; cameras correct | 120/120 | 120/120 |

The seventh condition removes every query observation: all 120 slots remain
unresolved at both tolerances, not compatible. All 646 reported compatible joint
witnesses were independently reprojected and satisfied positive depth and pixel
bounds. Tests check unchanged fitting inputs, unchanged unaffected tracks,
missing-data accounting, and deterministic generation. A batched Euler-angle
shape error was corrected before the completed experiment; no partial results
were used.

The six rows above are repeated measurements on ten synthetic scenes, not 120
independent clips per condition. Broad contradiction can result from a shared
track error even with correct cameras. A threshold on contradiction counts
therefore cannot by itself attribute failure to the camera estimator. The next
required evidence is independent image-based track identity/staticness evidence,
with measured error rates under shared as well as isolated tracking errors.
No vote threshold was selected, no DNF labels were assigned, and these controls
have not been added to the manuscript. They do not validate real tracking,
human judgments, or a learned evaluator.

### Joint later-frame consistency (development)

`probe_joint_query_consistency.py` extends the final-frame check to every later
observation. It requires one fixed 3D point to satisfy the fitting observations
and all available later observations, rather than allowing a different point
for each query. It retains all 12 original track slots, fitting prefixes
1–10/1–30/1–60, and coordinate-wise tolerances 1/2/4/8 pixels. Missing later
observations are counted but impose no constraint. Missing fitting observations
and inconsistent fitting geometry remain separate outcomes. This is classical
multi-view feasibility, not a new triangulation method or a camera-failure label.

Results are retained as `joint-query-consistency-v1.json.gz` in both
`results/static-rgbd-control-v1/` and `results/office-rgbd-replication-v1/`.
These are gzip-compressed complete JSON records, including individual checks,
joint witnesses, input provenance, and code hashes. The runner emits uncompressed
JSON; compression uses `gzip -n`. Both camera methods use their existing full-clip
RGB-only predictions. Evaluation does not read reference cameras or sensor depth.

Across the repeated prefix/tolerance/method/slot configurations, 23 near-scene
and 18 office-scene cases passed every available individual query but failed
the joint check. These are overlapping development configurations, not independent
examples or an accuracy estimate. Independently projecting all 87 near-scene and
109 office-scene joint witnesses confirmed positive depth and the requested pixel
bounds. A synthetic unit test also constructs individually feasible queries with
no common point, and another detects a middle-frame error despite a passing final
frame.

At prefix 60 and 4 pixels, the near scene has five jointly compatible DAGE tracks
and five contradictory DA3 tracks. The office scene has three compatible and two
contradictory DAGE tracks, versus five compatible DA3 tracks. Each setting also
has seven slots with insufficient fitting observations. One of the five office
tracks has no final-frame observation but does have earlier query observations;
thus this denominator differs from the endpoint test. At 8 pixels, both methods
have five compatible tracks in both scenes. These results do not establish a
general method ranking. Correspondence error, scene motion, calibration error,
and camera error can all produce a contradiction. Compatibility does not prove
accuracy, and missing observations do not imply successful tracking. No DNF
threshold is selected, and these development results have not been added to the
manuscript.

`inventory_pose_cache.py` inventories camera predictions on Anima without loading
the large depth and point-cloud arrays. `export_realestate_poses.py` runs in the
original `/home/kmatzen/git/geobench/.venv` environment and reconstructs seed-0,
ten-frame samples from `/data3/kmatzen/data/re10k_sub50`. It matches the old input
fingerprints and exports poses for native input and Gaussian blur sigma 1.5.
All 300 predictions matched (50 clips, three methods, two conditions).

`analyze_realestate.py` recomputes metrics locally. The recovered AUC aggregates
match the original blur-control records. ATE and AUC disagree on 27 of 149
decisive native-input method pairs, with one additional tied pair. This measures
disagreement, not accuracy against a quality target.

The legacy cache fingerprint covers image shape, intrinsics, and only the first
1 KB of image bytes. It cannot prove historical frame identity. Exported
timestamps are reconstructed; they were not stored at inference time. The
export includes the original loader's source hash for audit. Treat this as a
pilot until fresh runs store complete input hashes and ordered timestamps.

Run the local analysis with:

```sh
uv run python paper/analyze_realestate.py
```

The blur level was selected during earlier development on these clips. Bootstrap
intervals describe paired variation within this selected subset, not held-out
generalization or correction for the earlier parameter search. Ten sparse views
are suitable for geometric comparisons but insufficient to validate temporal
jitter judgments on dense video. The camera-evaluation contribution still needs
dense outputs, independent outcomes or reviewer labels, and held-out comparison
against ATE, RPE, pose AUC, and a combined metric baseline.

## Independent image evidence pilot

`extract_image_matches.py` runs on the original native frames, with SIFT capped
at 1,500 features and a mutual 0.75 ratio test. It uses neither candidate poses
nor reference poses to select matches. Dataset intrinsics normalize the pixel
coordinates; this tests calibrated extrinsics and does not evaluate a model's
predicted intrinsics. Each source image and camera metadata file is SHA-256 hashed.

`analyze_image_evidence.py` measures square-root Sampson residuals for each
candidate on those shared correspondences. At least 16 matches support a frame
pair and at least three common frame pairs support a method comparison. Each
frame pair contributes one median; the clip score is the median across pairs.
Zero-baseline predictions abstain because epipolar geometry is undefined there.
The score cannot detect translation scale or direction sign and is a standard
geometric baseline, not a novel preference evaluator.

On 149 decisive native-input method comparisons from 50 clips, agreement is
70.5% with pose AUC (clip-bootstrap 95% interval 62.4–78.4%) and 63.1% with ATE
(54.7–71.1%). Among 27 AUC/ATE disagreements, image evidence agrees with AUC in
19. The reference trajectory has a lower clip residual than all three estimates
in 49 of 50 clips, a useful but imperfect geometric control. The study is a
development-set pilot; no generalization or human-preference claim follows.

The complete 35 MB correspondence file remains on Anima at
`/data3/kmatzen/plumbline/research/image-evidence/matches.json`, using the large
data disk. Its content hash, source image hashes, camera metadata hashes, and
pair counts are committed in `results/image_evidence_manifest.json`. Pose-level
results and grouped uncertainty are in `results/image_evidence_analysis.json`.

To recompute locally after copying the correspondence file:

```sh
uv run python paper/analyze_image_evidence.py tmp/image-evidence/matches.json
```

The feature and residual definitions follow the standard
[OpenCV feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
and [epipolar geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)
methods. These are baselines against which the proposed review methodology must
be tested.

## Proposed functional-DNF study

### Reference-free setting: DAVIS

The primary intended use includes videos without reference camera trajectories,
not just rejudging ground-truth error plots. DAVIS is the first proposed dataset
for this arm. The official [DAVIS 2017 release](https://davischallenge.org/davis2017/code.html)
provides video frames and segmentation annotations; these annotations are not
camera-pose ground truth. Use the official 480p train/validation archive at
`https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip`.
The archive is being staged on Anima under `/data3/kmatzen/data/davis-downloads/`.
Do not infer camera intrinsics, frame rate, or capture timestamps from filenames.
Record ordered frame indices and mark unavailable calibration/timing explicitly.

After the archive download completes, `stage_davis.py ARCHIVE NEW_OUTPUT`
extracts only the official split lists and their JPEG frames. It refuses an
existing output directory, overlapping splits, duplicate archive entries,
unsafe paths, missing sequences, and ambiguous numeric frame order. Each frame
and the complete archive receive a SHA-256 hash; unavailable camera poses,
intrinsics, and capture timestamps remain null. The three unit tests use a
synthetic archive and do not establish successful staging of the real release.

`run_davis_dage.py ROOT CHECKPOINT_FILE SEQUENCE NEW_OUTPUT --limit 60` is the
initial pose-only development runner. It reads a staged inventory, verifies frame
hashes, and calls DAGE with `intrinsics=None`. It records the exact prefix,
decoded-image hash, checkpoint and source hashes, settings, runtime, prediction
hash, and failures. It does not create dummy reference poses, timestamps, or
intrinsics. It currently refuses validation sequences. Its six selection tests
do not substitute for a successful real-data run; no DAVIS result exists yet.

Use DAVIS training sequences for evidence-design development. Reserve validation
sequences for a frozen-protocol assessment; all windows, candidate methods, and
variants of an original sequence must stay in the same partition. Do not choose
test clips by looking for favorable model outcomes. A sequence roster and an
attempt log must include inference errors and unreviewable cases.

The main evidence packet must not contain reference poses, pose-error metrics,
human segmentation masks, or method identities. It should combine source frames,
synchronized estimated camera motion, independently extracted image tracks, and
candidate reprojection or epipolar evidence when supported. Pose-only output
does not provide depth: do not invent 3D reprojections. Predicted depth/intrinsics
must be identified as estimates, and their errors can confound pose judgments.
Image matches and estimated static-background regions can also be wrong; missing
support should lead to uncertainty, not automatic rejection of the tracker.
Segmentation-assisted analysis is a separate oracle ablation, and annotated
objects must not be equated with all moving pixels or static background truth.

Ask independent reviewers and VLMs whether the estimate is consistent with the
visible camera motion and whether the provided evidence establishes task failure.
Require supporting frames/intervals and permit `cannot judge`. Evaluate agreement
and false rejection against independent human judgments, with disagreements and
coverage retained. Without camera truth, this measures agreement about visible
plausibility and task usability, not geometric accuracy. Complement it with
reference-available datasets and controlled corruptions. The existing TUM plots
and failed plot-only VLM pilot are supporting diagnostics, not this primary
reference-free method. The DAVIS arm has no tracking or judging results yet.

### Failure definition and validation

### Harness first, then evaluator training

**Correspondence component corpus (not camera judgments).**
`build_tapvid_judge_corpus.py` now constructs annotation-supervised examples from
the ten TAP-Vid development videos. It never opens the twenty held-out videos.
Within development, the same name-hash ordering assigns six videos to training,
two to validation, and two to calibration. All tracks, frames, and derived images
from a video remain together. Candidate frames are fixed at lags 5, 15, 30, and
60 after each first-visible query, without selecting by checker score or outcome.

The corpus has 860 planned records: 435 labeled, 163 borderline/unlabeled, 146
beyond the video, and 116 with an unavailable candidate. Labels refer to the
reviewed annotation: `close` is within two native Euclidean pixels, `far` is at
least eight pixels away, and `occluded` means the reference point is annotated
invisible. The 2–8 pixel margin has no training label. No synthetic explanations,
VLM answers, camera-quality targets, or staticness claims are used as supervision.

| Partition | Videos | Close | Far | Occluded | Labeled total |
|---|---:|---:|---:|---:|---:|
| Train | 6 | 69 | 84 | 35 | 188 |
| Validation | 2 | 46 | 37 | 45 | 128 |
| Calibration | 2 | 89 | 17 | 13 | 119 |

Each panel contains the starting frame and candidate frame with context above
native-resolution 161×161 crops. Cyan rings leave the marked center pixels
unobscured. Rendering receives only images and the proposed coordinates, never
reference coordinates, visibility, residuals, or labels. Filenames use opaque
example IDs. Targets are stored separately in export records, not painted into
the images. The renderer's first training example of each class was inspected
for layout; that is not human label validation. Some fur/texture examples remain
visually ambiguous despite their annotation-derived target, so this is not proof
that every target is inferable from two images.

Artifacts are on Anima at
`/data3/kmatzen/plumbline/research/tapvid-davis-v1/judge-corpus-v1`.
The manifest and three JSONL exports are retained in
`results/tapvid-judge-corpus-v1/`; image paths in those exports resolve against
the remote corpus directory. All 598 rendered image hashes and annotation
targets, and all 435 labeled export records, were independently checked.
The partitions and class imbalance are explicit. This small, six-video training
set supports a bounded component pilot, not a claim of a large training corpus.
The training pilot below now uses the training partition. Validation/calibration must not be merged
into training, and the earlier held-out baseline result is not fresh evidence
for any future model-selection loop.

**LoRA feasibility and fixed one-pass pilot.**
`train_tapvid_lora_pilot.py` completed three training-only steps, one per target
class, using the pinned Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. It uses NF4 double quantization,
fp16 quantized matrix computation, fp32 language-attention scores, and PEFT's
k-bit preparation. This modified arithmetic is not certified equivalent to the
unquantized model. PEFT 0.21.0 is isolated in
`/data3/kmatzen/plumbline/research/peft-0.21.0`; bitsandbytes 0.48.2 uses its
existing isolated directory. The shared environment was not upgraded.

Only 1,843,200 LoRA parameters in language attention `q_proj`/`v_proj` are
trainable (rank 8, alpha 16, dropout 0). The vision encoder and base model remain
frozen. Image/prompt tokens are masked from the supervised loss; only the
assistant target extension is supervised. All three losses and gradient tensors
were finite, with nonzero gradient norms. Peak allocation was 5,121,092,608 bytes.
The adapter and executed source are retained on Anima under
`/data3/kmatzen/plumbline/research/tapvid-lora-pilot-v1/smoke-1`; its complete
manifest and source hashes are in `results/tapvid-lora-pilot-v1/`. This verifies
training feasibility, not improved predictions or generalization.

A separate one-pass run over all 188 training examples was launched under
`/data3/kmatzen/plumbline/research/tapvid-lora-epoch-v1/run`. It starts again from
the pinned base, not the smoke adapter. Order is shuffled with Python seed 0;
batch size is 1, AdamW learning rate 1e-4, and gradient norm clipping is 1.
No epoch count or hyperparameter was selected from validation responses. The
run stopped on example 92 with a non-finite forward loss after 91 finite updates.
It is terminal, not waiting, and no final adapter was saved. Its complete failure
manifest and executed runner are retained in `results/tapvid-lora-epoch-v1/`.
The separate one-example probe from the unchanged base reproduced the non-finite
loss on that same input, before any updates. Changing only quantized matrix
computation to fp32 made the probe complete with loss 0.24543 and a finite,
nonzero gradient norm of 0.91493. Both probe manifests and exact executed scripts
are retained beside the failed epoch record. This isolates an arithmetic-sensitive
failure that does not require learned adapter updates; it does not certify
equivalence to an unquantized reference model.

The fixed 188-example pass has been restarted from the original base under
`/data3/kmatzen/plumbline/research/tapvid-lora-epoch-fp32-v1/run`, retaining the
same seed, example order, optimizer settings, and one-pass budget. Quantized
matrix computation is now fp32. The uploaded evaluator reads this setting from
the training manifest for both arms. The remote command starts validation only
after the trainer exits successfully; neither the failed fp16 run nor the
one-example diagnostic adapter is used as the evaluated trained model. Completion
is now verified: all 188 unique training examples were processed in the frozen
seeded order, with finite losses and positive finite gradient norms at every
step. Peak allocated GPU memory was 5,122,803,200 bytes. The completed manifest
and exact executed training/evaluation sources are retained in
`results/tapvid-lora-epoch-fp32-v1/`. The saved adapter is retained on Anima, with
its file hashes recorded in that manifest. The paired validation run completed
successfully; the results and limitations are recorded below. Successful
training alone is not evidence of improved recognition.

`evaluate_tapvid_lora.py` freezes the next comparison: all 128 validation panels
under the same prepared quantized model, once with adapters disabled and once
enabled. Greedy generation permits eight new tokens. The parser accepts only a
single `close`, `far`, `occluded`, or `uncertain` label (case/outer whitespace
normalized); everything else is invalid. Report the full class confusion,
overall and class-balanced correctness, per-video results, invalids and
abstentions. Do not condition accuracy on answered examples alone. There are
only two validation videos, so these observations are not 128 independent scenes.
The 119 calibration examples remain unused. The evaluator verifies identical
processed image hashes and the saved base/adapter file hashes. GPU evaluation
completed from the fp32-compute training pass. Both arms contain all 128 selected
examples, with matching processed-image hashes and 609 image tokens per example.
The completed report's training-manifest, validation-export, and executed-source
hashes match the retained files.

`analyze_tapvid_lora.py` rejects incomplete or mismatched paired generation records
and includes abstentions/invalids in all-example accuracy. Its frozen baselines
are the majority label from training only and the original image check, which
may admit `close` or abstain but never infer `far` merely from rejection. Per-video
confusions, close-admission precision, class recall, and paired correctness counts
are retained. Answer coverage and answered-only accuracy are also reported to
separate response-format compliance from recognition. Answered-only accuracy
uses a model-selected subset and cannot replace all-example accuracy. Tests
cover fully formatted but entirely wrong answers and zero-answer cases. This
reporting code has unit tests and has now analyzed the complete paired results.

Before obtaining validation predictions, the report also adds a descriptive
within-video label-association check. For each video, the expected accuracy
under uniformly permuted answers is computed exactly from the target and answer
frequencies, including abstentions and invalid answers. Observed accuracy minus
this expectation is averaged equally across videos. Constant answers within a
video have zero excess, even if its dominant label makes accuracy high. This is
not a significance test, a deployable baseline, or proof of causal image reliance;
it only checks whether answers distinguish cases within the same video. The
implementation is checked against exhaustive permutations on a small example.

**Completed development validation (not a manuscript result).** Raw generations
and the complete analysis are retained in
`results/tapvid-lora-epoch-fp32-v1/validation-v1.json` and
`validation-summary-v1.json` in that directory. All-example accuracy was 37/128
(28.9%) for the base and 54/128 (42.2%) for the trained model. Both produced a
valid class label on every example, so the difference is not format compliance.
The base always answered `far`, matching the training-majority baseline.
The trained model's recalls were 41/46 for `close`, 13/37 for `far`, and 0/45
for `occluded`; balanced accuracy was 41.4%. It predicted `close` 101 times,
of which only 41 were correct. This is unsuitable as a reliable correspondence
admission rule, despite higher aggregate classification accuracy.

The frozen NCC/forward-backward check admitted 30 cases, 29 correctly, and
abstained on 98. Its 96.7% close precision comes with only 23.4% coverage; it
does not attempt a three-class decision. Its 22.7% all-example accuracy must
not be interpreted as worse evidence reliability than the trained classifier.
These comparisons use the preselected labeled panels, not all video points.

Trained accuracy was 25/78 (32.1%) on breakdance and 29/50 (58.0%) on judo.
Within-video accuracy above the shuffled-answer expectation was -1.9 and +22.4
percentage points, respectively. Thus the aggregate change does not demonstrate
consistent discrimination across scenes. There are only two validation videos;
we make no significance or camera-DNF claim. The reserved calibration partition
remains unused. Neither low-confidence output filtering nor another training
budget has been selected from these results. Before any larger training study,
the next diagnostic should distinguish failure to fit the training task from
failure to transfer to new videos, using training-only examples and the same
saved adapter. These development findings are not added to the manuscript.

The training-fit diagnostic is now launched at
`/data3/kmatzen/plumbline/research/tapvid-training-fit-v1/run`. It generates
answers for all 188 original training panels using the completed adapter,
without weight updates, resampling, or changing decoding. It reads the training
export only; validation and calibration exports are not used. The report is
explicitly marked `partition: train`, and the validation analyzer rejects it.
Comparing fit on the training set with development validation can help separate
poor task fit from poor transfer; it does not identify a unique cause or
establish generalization. The run completed all 188 examples. Its executed
scripts, raw predictions, and reproducible summary are retained in
`results/tapvid-training-fit-v1/`; `analyze_tapvid_training_fit.py` verifies the
training/corpus hashes, full membership, targets, and response parsing.

Training-fit accuracy is 98/188 (52.1%), versus 84/188 (44.7%) for the training
majority label. Class recalls are 56/69 close, 42/84 far, and 0/35 occluded;
balanced accuracy is 43.7%. Every response is a valid label, with no abstention.
The model gives a constant close answer on camel, drift-straight, and goat.
Equal-video excess accuracy over the within-video shuffled expectation is 4.0
percentage points. The low fit and absent occlusion predictions occur on seen
data as well as validation data, so a transfer-only explanation is insufficient.
This does not isolate optimization, representation, arithmetic, or annotation
ambiguity as the cause. The next bounded diagnostic should test whether the
same training setup can fit a small class-balanced training-only subset before
expanding the corpus or spending a larger training budget. Calibration remains
unopened; the manuscript remains unchanged.

**Fixed small-set learning diagnostic.** A separate run starts from the original
base checkpoint at `/data3/kmatzen/plumbline/research/tapvid-memorization-v1/run`.
The first three opaque IDs in each training class define nine examples, without
using model predictions to select them. Each is presented twenty times in
twenty balanced passes, shuffled with the pass index as seed: 180 updates total.
Optimizer, adapter, image processing, loss masking, and fp32 quantized compute
are unchanged. The final adapter generates labels for the nine examples with
the same greedy decoding, without further updates. This tests small-set fit,
not generalization; no validation or calibration export is read. The budget is
fixed before execution, with no intermediate model selection. The run completed
all 180 updates with finite losses and positive finite gradient norms. Its final
greedy answers were correct on all nine examples, including all three occlusions.
The exact schedule, selected IDs, targets, script/corpus hashes, processed-image
hashes, and saved-adapter hashes were independently checked. Processed images
match those used in the earlier training-fit inference. The completed manifest
and executed source are retained in `results/tapvid-memorization-v1/`.

This verifies small-set fit and shows that the setup can generate the occlusion
label after training. It does not establish visual reasoning or generalization:
the examples span only three videos, and two close examples share a track.
Different sample composition, class balance, and repeated exposure all changed
relative to the original one-pass experiment, so the diagnostic does not isolate
a single explanation for that model's low fit. A longer full-training study
must start from the original base with a fixed training-only schedule, preserve
the current validation results, and leave calibration untouched until a model
and decision rule are fixed. The nine-example adapter is not a candidate for
held-out performance claims. This implementation diagnostic remains outside
the manuscript's reported experiments.

**Fixed twenty-pass full-training study.** This study starts from the
original pinned base, not either learned adapter, under
`/data3/kmatzen/plumbline/research/tapvid-lora-twenty-pass-v1/`. All 188 training
examples receive exactly twenty updates each (3,760 total). The first pass
matches the original seed-0 one-pass order; later passes shuffle that order
with their zero-based pass index as seed. Class frequencies, optimizer, learning
rate, adapter rank, masking, and arithmetic are unchanged. The schedule is
fixed before execution, with no validation reads, early stopping, or intermediate
checkpoint selection. This tests increased exposure, not class resampling.

After successful training, the queued command measures full training fit and
then repeats the paired base/adapted development-validation comparison using
the final adapter. Calibration remains unopened. The validation videos have
already informed development and are not a fresh test set; no cross-scene
significance or camera-DNF claim can follow from this comparison alone. The
training manifest records unique training membership separately from the
repeated update schedule. Training completed all 3,760 updates with finite
losses and finite, positive gradient norms. The saved adapter files match their
recorded SHA-256 hashes. The complete training record is preserved in
`results/tapvid-lora-twenty-pass-v1/training.json`. Saved-adapter training-fit
evaluation completed: 188/188 correct (69 close, 84 far, 35 occluded), with no
invalid responses or abstentions. The complete predictions and hash-checked
analysis are retained as `training-fit.json` and `training-fit-summary.json`
in the same directory. This establishes fit on the training set, not
generalization or camera-failure detection.

The paired two-video development validation also completed. Base accuracy was
37/128 (28.9%); adapted accuracy was 73/128 (57.0%), with balanced accuracy
58.1%. Adapted class recalls were 41/46 close, 29/37 far, and only 3/45
occluded. Of 57 adapted close predictions, 16 were wrong (71.9% precision).
The frozen image gate made 30 close predictions with one error (96.7%
precision), abstaining on the other 98 examples. These different coverage
levels must accompany any comparison. Both model arms returned valid labels
for every example. Input hashes and paired processed-image hashes were checked;
`validation.json` and `validation-summary.json` preserve the full results.
This development result does not establish a reliable DNF rule. The unchanged
adapter's predeclared external-video comparison has completed, as recorded
below. These development diagnostics remain outside the
manuscript's reported experiments.

The development order follows the useful separation in
[3DCodeVerse](https://3dcodeverse.com/): improve the evidence/checking workflow,
collect verified data, then compare model training with the workflow held fixed.
Their published recipe trains code generators; our proposed adaptation trains a
camera evaluator. This is an experimental plan, not an achieved result.

First establish that the reference-free evidence supports reproducible human
judgments and that at least one base-model configuration passes basic controls.
Keep every attempt and evidence version, including failures and abstentions.
Only then build supervised examples from independently verified decisions,
supporting frames, and concise evidence-grounded explanations. Do not use the
current Qwen pilot's unsupported answers as positive training targets. Maintain
label provenance and distinguish human labels, adjudications, teacher proposals,
and controlled-corruption metadata. Human disagreement must not be erased.

Evaluate the base and fine-tuned model on the identical frozen evidence pipeline,
alongside conventional metric and learned multi-metric baselines where available.
Split original scenes before constructing windows or examples, and keep all
methods, transformations, and repeats of a scene together. Reserve tracking
methods as an additional generalization test. For DAVIS, human review measures
visible consistency/usability, not unavailable pose accuracy. Report false DNF
decisions, missed failures, abstention, cost, and scene-grouped uncertainty.

Preference optimization is a later experiment requiring verified better/worse
judgments; RL could study which frames or geometric checks to request under a
budget. Neither should optimize against the evaluator's own unchecked scores.
Any Bayesian optimization of the harness uses development scenes only, and all
ablation choices and model settings must be frozen before held-out evaluation.
The correspondence LoRA pilot has begun; no RL has begun. Camera-failure labels remain insufficient for
a defensible camera-judge training experiment; the narrower correspondence
corpus above supplies a separate component-training pilot, not those labels.

Status: protocol under development, not a preregistration or a validated metric.
The research question is whether a VLM can identify task-failing camera estimates
that deterministic validity checks and conventional metric baselines miss.
Crashes, timeouts, and invalid trajectories are mechanical DNFs. A complete,
valid trajectory can separately be judged functionally unusable. Do not describe
this second category as a failure to execute.

The initial target task is camera-motion recovery for stable placement of static
3D content across a clip, up to one global similarity transform. The proposed
failure criterion is loss of that correspondence, not merely a nonzero pose
error or an unattractive path plot. Failure examples to test include sustained
background misregistration, tracking a moving foreground instead of the camera,
and discontinuities inconsistent with the source video. A stationary estimate
on a stationary-camera clip is not a failure. Rapid real camera motion is not
jitter. Source blur, poor feature matches, uncertain reference poses, and moving
objects can make the evidence inconclusive rather than prove tracker failure.
The annotation pilot must establish concrete severity and duration anchors
before any held-out evaluation; those thresholds are not yet established.

Each candidate should receive a fixed evidence packet with ordered source frames
and timestamps, candidate camera visualization, and independently constructed
static-scene reprojection evidence where available. Record unavailable channels.
Give both humans and the VLM the same evidence. Separate a reference-free arm
from a reference-assisted arm; do not leak ground-truth error plots into claims
about reference-free evaluation. Blind method identity and leaderboard scores.
Require `usable`, `unusable`, or `abstain`, an evidence identifier, and an
explanation identifying the supporting frames or intervals. Missing evidence
must not be converted into a tracker DNF. VLM prose and confidence are not
ground truth or calibrated probabilities.

Collect individual labels from at least two independent reviewers, keep their
disagreements, and adjudicate separately. The existing A/B pilot does not provide
these labels: preferring A over B says neither whether A is usable nor whether B
is unusable. Project-member labels can refine the rubric but should not be
presented as external validation. No current pairwise votes will be relabeled
as functional DNFs.

Compare against validity-only rules, individual ATE/RPE/AUC thresholds, a
combined-metric classifier, and image-correspondence residuals. Fit thresholds
and any learned baseline only on training/calibration scenes. Freeze the rubric,
prompt, model version, evidence renderer, and acceptance rule before testing.
Split by original scene, grouping all methods, variants, windows, and repeats;
also evaluate a tracking method excluded from calibration. Report false
rejection among human-usable results, failure recall among human-unusable
results, abstention coverage, and task-clustered intervals. Compare gates at a
matched false-rejection target selected before testing, rather than only at
their most favorable accuracy. Report all target violations. Test sensitivity
to evidence channels, frame density, prompt wording, and repeat calls.

`plumbline.tracking_failure.summarize_failure_review` implements the accounting
for one system/configuration. Start from a complete attempt roster. Report
mechanical DNF, functional DNF, and unresolved fractions over all attempts,
alongside both raw complete-output metrics and accepted-output metrics with
their counts. Missing/abstaining judgments remain unresolved. Failure-rate
bounds assign unresolved attempts first to nonfailure and then to failure;
these are accounting bounds, not statistical confidence intervals. No existing
leaderboard is filtered by this experimental code. A reduced conditional error
after rejection is not, by itself, improved tracking or improved evaluation.

Related work to distinguish explicitly:

- [Rabiee and Biswas, Introspective Perception for Mobile Robots (2023)](https://arxiv.org/abs/2306.16698)
  learns perception-error distributions using sensor redundancy and consistency
  constraints, including visual SLAM. Learning reliability from consistency is
  therefore not itself new. Our intended output is an external, candidate-specific
  camera usability decision, rather than an error model integrated into a robot's
  estimator. That distinction still needs an experiment demonstrating value;
  changing the output name to DNF is insufficient.
- [Duan et al., AHA (2024)](https://arxiv.org/abs/2410.00371)
  fine-tunes a VLM for manipulation failure detection using procedurally
  perturbed simulation demonstrations. It also applies failure feedback to
  downstream planning and reinforcement-learning workflows. Thus the general
  failure-harness-to-training recipe and the use of a VLM as failure feedback
  already have precedents. A camera-specific contribution needs validated
  evidence construction and failure criteria, not only a new application label.
- [Tan et al., Natural Language Camera Movement Understanding (ACaM, 2026)](https://arxiv.org/abs/2607.03043)
  benchmarks and fine-tunes VLMs for camera-motion categories. Its task is motion
  recognition, not assessing an estimated trajectory's geometric correctness.
  The authors document confusion between camera and object motion, translation
  and rotation, and movement directions. Consequently, recognizing plausible
  motion language cannot certify our camera estimates. ACaM's training and
  benchmark labels should not be treated as camera-DNF labels. These entries
  were checked against primary abstracts and project material; no reproduction
  or comprehensive full-paper comparison has yet been completed.
- [Kahl, Multiple View Geometry and the L-infinity Norm (ICCV 2005)](https://vision.ucsd.edu/publications/2005/multiple-view-geometry-and-l-infinity-norm)
  formulates multiview triangulation with positive depth and bounded reprojection
  error as convex feasibility, using second-order cones for Euclidean image
  distances. Our coordinate-wise pixel boxes give linear constraints instead.
  The point-support check is therefore an established geometric ingredient, not
  a new triangulation contribution. The paper also notes sensitivity to outliers;
  a contradictory track cannot by itself identify camera error.
- [Thomas, Agnihotri, and Keuper, Failure or Drift? (NeuSLAM 2026)](https://arxiv.org/abs/2608.30690)
  separates explicit tracking failure from drift in still-active systems, and
  compares synthetic corruption conclusions with real adverse conditions.
  Its abstract reports changes in tracker ordering with corruption design.
  Separating failure from conditional trajectory error and testing corruption
  validity are therefore not unique to this project. Bibliographic metadata and
  the abstract were checked against the authors' page and arXiv; a full protocol
  comparison and reproduction have not been completed.

The proposed contribution must instead demonstrate that source-grounded,
localized evidence with explicit abstention supports reliable functional-failure
decisions when reference camera poses are absent. That claim is still unproven.
Missing-observation bounds are basic partial-identification accounting, not a
new statistical estimator. Neither the solver, accounting formula, nor combining
them with an unvalidated VLM suffices as a novelty claim. The additional BibTeX
records are now cited in the revised manuscript's related-work section. The
eight-page PDF was rebuilt with resolved citations and visually checked; this
literature revision does not add the unsuccessful model-training experiments
or establish the proposed camera-level contribution.

[3D-DefectBench](https://arxiv.org/abs/2607.10826v2) studies how the model,
rendering, evidence, and prompt affect human agreement for 3D-asset defects;
[Spotlight](https://spotlight-video.github.io/) studies localized error detection
in generated video. Neither establishes the proposed camera criterion. Merely
using a VLM for failure detection is not a sufficient novelty claim. Full
related-work coverage and actual camera experiments remain outstanding.

### Fresh dense development runs

#### Static-scene geometric control (staging)

The retained caption diagnostics do not provide reliable source-point labels.
Before further judge tuning, the next experiment checks the geometric evidence
on a separate static-scene control: `rgbd_dataset_freiburg3_structure_texture_near`.
The [official sequence catalog](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/)
describes camera motion along textured wooden panels, with textured posters on
the floor, and supplies RGB-D observations and reference poses. This controlled,
textured scene is deliberately favorable. It is not representative validation
of dynamic videos or evidence that every image correspondence is correct.
In particular, occlusion boundaries, depth holes, timing, and calibration can
still invalidate a proposed surface point.

The official archive URL is
`https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_texture_near.tgz`.
Its verified redirect uses `/g/rgbd/dataset/freiburg3/` on the official webshare
host; the catalog's relative archive link without `/g/` returned 404. A HEAD
request reports 567,005,874 bytes. Download and gzip integrity verification
completed successfully in `/data3/kmatzen/plumbline/research/static-rgbd-validation-v1`
on Anima. Selective extraction completed with 90 RGB frames, 87 associated
depth images, and 89 associated reference poses. No camera prediction or
evaluator result is yet claimed. Existing dynamic
sequence folders are untouched; the pruning staging script is not used.

`stage_static_rgbd_control.py` implements the fixed RGB selection independently
of depth/reference coverage. Its staging manifest records all selected RGB
hashes and timestamp-index line numbers, nearest depth/reference candidates,
offsets, explicit missing/ambiguous statuses, source archive and index hashes,
and executed-script hash. Reused depth observations remain associated with
their original RGB rows; duplicates at the selected reference/depth timestamp
are not silently resolved. It does not interpolate poses or certify depth
accuracy. Five synthetic tests cover sample independence from missing data,
local timestamp ambiguity, unsafe paths, duplicate archive names, symlinks,
and nonfinite timestamps. The extraction uses only ordinary archive files and
refuses an existing destination.

The retained manifest is `results/static-rgbd-control-v1/staging-manifest.json`;
the local source copy is `tmp/static-rgbd-validation-v1/selected`. All selected
RGB/depth file hashes and index hashes were checked after transfer, and all
nearest associations and the stride-three RGB selection were reconstructed
from the retained indices. RGB images decode as 640x480 uint8 color and depth
as 640x480 uint16. The archive SHA-256 is
`c7dfd479b6bf84d87848882d8dfa93715bfa55f4f45d4adc072a67d2f0363df9`.

Depth associations are missing at original frames 12, 46, and 69. Among the
87 associated depth images, absolute timestamp offsets have median 0.022 ms
and maximum 0.620 ms; nonzero pixel coverage ranges from 85.93% to 88.34%
(median 87.41%). These whole-image fractions do not certify any selected
point's depth. Reference frame 0 is unavailable: its nearest reference time
is 112.608 ms later and outside the 20 ms tolerance. The other 89 reference
associations have median absolute offset 2.606 ms and maximum 12.587 ms.
The raw reference index contains no duplicate timestamps. Associated pose
values are finite and quaternion norms range from 0.999931 to 1.000072.
This is an input check, not proof of reference-pose accuracy.

**Prospective availability amendment, before camera inference:** retain the
original frame-zero feature selection and all 90 tracking frames, but compare
reference and candidate geometry using the same ten available fit frames
1–10. Keep the originally planned test frames 11–89 (12 × 79 = 948 planned
observations). Record frame 0 as reference-unavailable; do not extrapolate a
pose, resample RGB, or move the test boundary to fit eleven replacement frames.
This is explicitly a ten-fit-frame control, not the earlier eleven-fit-frame
DAVIS protocol. The amendment is based only on reference availability, not on
candidate errors or track survival. Assistant inspection of source frames 0,
45, and 89 shows the textured panels and floor posters; it is not independent
validation of all correspondences.

The development protocol is fixed before camera predictions are inspected:

1. Retain the archive and timestamp indices with hashes. Select the first 90
   chronological RGB frames at stride three. Preserve missing depth/reference
   associations explicitly; do not shift the clip to improve tracking results.
2. Verify reference timing and the published Freiburg-3 calibration before
   evaluating camera estimates. Keep source, timing, and measurement validity
   separate from camera validity. Do not silently collapse duplicate reference
   timestamps or interpolate outside the reference span.
3. Select the same 12 first-frame grid features using only RGB, with no access
   to candidate poses, reference geometry, or later tracking outcomes. Retain
   empty slots, missing tracks, and invalid point fits in the denominator.
4. Use depth and reference poses only on the validation side: inspect world
   spread and depth-boundary ambiguity of the measured tracks. Evaluate the
   existing early-fit/later-frame geometric diagnostic on reference cameras,
   the unchanged five-degree later-yaw control, and actual model predictions.
   Report shared-support comparisons and coverage separately. These are
   continuous diagnostics, not automatically assigned functional-DNF labels.
5. Check whether reference cameras and the known rotation intervention produce
   distinguishable evidence on valid support. Inspect discrepancies rather
   than selecting a threshold or dropping inconvenient points after seeing
   model rankings. A successful control only supports this implementation
   check; it does not establish novelty, generalization, or human usability.

This is a prerequisite for validating the proposed evaluation method, not a
replacement contribution. A publishable end-to-end evaluator still requires a
defined functional-failure decision, independent validation targets, comparison
with existing metrics, and a frozen held-out evaluation. The manuscript is
unchanged while this control is being prepared.

`probe_static_rgbd_control.py` has now run the reference/rotation comparison;
the full record is `results/static-rgbd-control-v1/reference-control-v1.json`.
It reconstructs source selection and associations, verifies RGB/depth/index
hashes, tracks all 12 original grid slots, and maps every result back to the
90 original frame indices. The synthetic test verifies exact projection under
translation and that the rotation intervention starts at original frame 11,
not inside the fit window.

Eleven slots have an early fit and one loses its track during fitting. On
579 common later observations out of 948 planned, the median early-fit
projection errors are 679.896 pixels for reference cameras and 540.249 pixels
for the five-degree later-yaw control. Thus this point-estimate diagnostic is
not a valid camera-quality criterion on this control. These values are internal
development records, not results added to the manuscript.

A post-hoc depth-assisted check identifies a severe anchor-depth discrepancy.
The diameter of reference camera centers over fit frames 1–10 is only 0.004707 m.
For the ten fitted slots with a measured depth at frame 1, fitted axial depths
at that frame range from 0.164 to 0.265 m, while sensor depths range from 1.031
to 1.415 m. Fixing a point from that frame's depth and reference pose, then
projecting it into later frames without refitting, yields median error 3.486
pixels on 564 common observations; the original fitted points give 710.660
pixels on that same subset. The remaining slots and missing observations are
not included in that paired median and remain in the full record. Depth-derived
points are validation-only and are not passed to the reference-free evaluator.

This supports poor early depth estimation as the immediate problem, with the
very short fitting baseline as a plausible cause; it does not certify every
track or eliminate calibration, timing, or sensor noise. A visible track can
have useful later evidence while its early 3D point estimate is unreliable.
Before interpreting model rankings, the evaluator must represent this point
uncertainty or return insufficient evidence. The next check should test the
existing set-valued point-support formulation on this control, not tune a
failure threshold against these two observed medians. The DAGE comparison has
not been run, and no functional-DNF decision has been assigned.

The set-valued check is now retained in
`results/static-rgbd-control-v1/point-support-v1.json`, generated by
`probe_static_point_support.py`. It reuses the existing classical box-frustum
solver and unchanged 1, 2, 4, 8 pixel axis-wise tolerance grid. It uses no depth
input. All original tracks and the common ten-frame fit window are preserved.
The following are counts of positive-depth witnesses among the 11 checked
slots; the twelfth has insufficient early observations at every tolerance.

| Axis-wise tolerance (px) | Early: either trajectory | All observed: reference | All observed: later yaw +5 degrees |
| --- | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 2 | 9 | 0 | 0 |
| 4 | 11 | 4 | 0 |
| 8 | 11 | 11 | 0 |

Other checked slots return numerical solver infeasibility, not exact proofs
of camera failure. The 11 checked tracks supply 579 later image observations
out of 948 planned. At eight pixels the reference trajectory admits a point
for every checked track while the rotation control does not. At tighter
tolerances reference tracks can also be inconsistent. No tolerance is selected
as a calibrated validity threshold. Full-track support fits all observed
frames jointly; it is not the earlier held-out projection task and must not be
presented as an improved prediction score. The synthetic test checks the
common frame mapping, the expected rotation-control separation, and explicit
missing-observation accounting. This is an implementation control for a
classical geometric component, not evidence of novelty or a validated DNF rule.

A fresh DAGE run completed at
`/data3/kmatzen/plumbline/research/static-rgbd-validation-v1/dage-v1`.
`run_static_dage.py` reads the fixed RGB sample directly, without reusing the
dynamic-dataset loader or a prediction cache. It supplies only RGB and published
Freiburg-3 calibration to the model; reference poses and depth are not model
inputs. It records the decoded RGB hashes, pinned checkpoint contents, executed
adapter source hashes, settings, environment, runtime, and prediction hash.
The loader test verifies that validation metadata is ignored and modified RGB
bytes are rejected. The checkpoint and 252-pixel/float16 settings match the
earlier DAGE development configuration. Executed source hashes, rather than
a local branch name, identify the implementation used for this comparison.

The complete prediction and executed runner are retained under
`results/static-rgbd-control-v1/dage-v1`. The comparison loader re-read and
hashed all RGB files and decoded images, checked exact frame ordering and
prediction bytes, and validated all 90 rigid camera poses. Checkpoint contents,
executed adapter source files, and settings match the earlier walking-XYZ run
exactly. Runtime including load was 181.65 seconds, with 6.54 GB peak allocated
GPU memory; this is not a controlled runtime benchmark.

`results/static-rgbd-control-v1/dage-point-support-v1.json` contains the model
comparison. It reproduces all eight prior reference/rotation tolerance cases
exactly and adds DAGE without changing tracks, calibration, fitting frames,
tolerances, or solver settings. The original support script is archived as
`executed-point-support-v1.py` beside the original results. Among the same 11
checked tracks (579 later observations / 948 planned), DAGE has positive
full-track witnesses for 0, 3, 11, and 11 tracks at 1, 2, 4, and 8 pixels,
respectively. All 11 early windows have witnesses at each tolerance. The
twelfth slot remains insufficiently observed.

On the 89 frames with reference associations, conventional Sim(3)-aligned
ATE-RMSE is 0.006427 m, adjacent-frame translation RPE-RMSE is 0.006296 m, and
rotation RPE-RMSE is 0.315421 degrees. These use the retained nearest-timestamp
reference, not interpolated poses. The reference metrics are kept separate
from the reference-free support calculation, which uses no pose alignment.
The smaller pixel tolerance needed for DAGE than for the supplied reference
does not mean it is more physically accurate: consistency with image tracks,
reference timing/noise, and calibration are different quantities. Both DAGE
and reference are supported at eight pixels, while the known rotation
intervention is not. This is a favorable single-sequence component control,
not an operational failure threshold, method ranking, or validation of the
proposed complete evaluator. No manuscript claims are added from this run.

#### Controlled rotation response (development only)

`probe_static_rotation_response.py` evaluates the same reference poses and
tracks at signed local-yaw offsets of 0, 0.25, 1, and 5 degrees. Two schedules
leave frames 1–10 unchanged: a step at frame 11, and a linear ramp reaching the
specified angle at frame 89. All 56 schedule/angle/tolerance cases are retained
in `results/static-rgbd-control-v1/rotation-response-v1/response.json`. This is
a controlled sensitivity check, not a new dataset or functional-failure label.

At eight pixels, the unmodified reference and both one-degree steps support
11 of 11 checked points. Both five-degree steps support zero. The five-degree
ramps support 10 points in the negative direction and eight in the positive
direction. At four pixels the unmodified reference supports four points,
versus zero for both one-degree steps and zero/two for negative/positive
five-degree ramps. One of the 12 original slots remains insufficient in every
case. No pixel tolerance was selected as a failure threshold.

All early solver results and observed-frame masks are unchanged across the
perturbations. Both zero-offset schedules reproduce the archived reference
results exactly, and the positive five-degree step reproduces the earlier
rotation control exactly. Unit tests check local-axis composition, ramp
endpoints, unchanged camera centers, and preservation of source arrays.

The differing step and ramp responses limit the interpretation of the earlier
five-degree result: rejecting an abrupt rotation does not establish sensitivity
to gradual drift. Joint fitting can accommodate some camera errors with a
different static point, and missing tracks can reduce the observed corruption.
These controls do not justify a camera DNF decision. They remain in this research
log, not the manuscript; any eventual method claim must respect these limits.

`audit_rotation_witnesses.py` independently reprojects every positive witness
from all 56 cases in world coordinates, checks positive depth and the original
axis-wise pixel bounds, and verifies each observation mask against the retained
tracks. `rotation-response-v1/witness-audit.json` records every slot, last
observed frame, actual yaw at that frame, and verified residuals. Five checked
tracks reach frame 89; six end earlier. At eight pixels, four of those five
full-length tracks still have positive witnesses under the negative five-degree
ramp, and three under the positive ramp. The endpoint residuals for those
witnesses range from 3.68 to 7.79 pixels. Thus endpoint censoring cannot fully
explain the gradual-drift result: these retained 3D points satisfy the bounds
even after observing the full injected rotation. Requiring final-frame coverage
alone would not resolve the ambiguity. This is a post-hoc diagnostic, not a
new admission rule or a calibrated failure threshold.

#### Sensor-depth diagnostic

`probe_depth_constrained_rotation.py` repeats all 56 rotation cases with two
additional first-camera axial-depth intervals: measured frame-one depth ±10%
and ±25%. These are diagnostic bands, not calibrated sensor uncertainties.
Depth is external evidence unavailable to the intended DAVIS evaluator. Missing
frame-one depth is not replaced by a later observation. The result therefore
contains ten eligible slots, one missing-depth slot, and one insufficient-track
slot in each case. All 112 cases are retained in
`rotation-response-v1/depth-oracle.json`.

At eight pixels, all ten eligible reference slots have witnesses with either
band. Five-degree negative/positive ramps have six/six witnesses with the 10%
band and nine/seven with the 25% band. Among the five tracks reaching frame 89,
the corresponding counts are three/two and four/three. Thus even the tighter
sensor-depth band does not eliminate the gradual-drift ambiguity. A single
depth constraint is not sufficient evidence for a DNF rule.

The optional `first_depth_interval` in `frustum_support.box_support` constrains
axial depth in the same units as the camera translations; default behavior is
unchanged. Tests cover a feasible interval, a conflicting interval, scale
changes, and invalid bounds. All 271 positive witnesses in this diagnostic
were independently reprojected in world coordinates, verifying positive depth,
pixel bounds, and the specified sensor-depth interval. These findings remain
development evidence outside the manuscript.

#### Concrete visual review instructions

`render_static_drift_review.py` makes unblinded teaching illustrations for all
five slots observed in every frame 1–89 with measured frame-one depth. It fixes
one world point using that initial depth and reference pose, then projects it
without refitting under the reference path and the positive five-degree ramp.
The three columns show frames 1, 45, and 89. Each column contains scene context,
an unmarked crop, the reference projection, and the corrupted projection.
Yellow rings show the fallible image tracker; cyan crosses show projected points.
Crops follow the image tracker and must not be mistaken for stabilized camera
output. Selection is by observation/depth availability, not residual size.

The instructions ask whether the cyan cross stays on the same surface detail,
not whether a trajectory plot looks plausible. In slot 6 the starting detail
is inside the printed letter “e”: the reference cross remains there, while the
corrupted cross moves left to another letter. The assistant inspected all five
rendered sheets for layout and visible correspondence; this is not a human
judgment or blinded validation. Slot 6 is an explanatory example selected after
inspection, not a representative accuracy result.

Generated sheets are in `tmp/static-drift-review-v1/`; their hashes, point
coordinates, and projections are retained in `drift-review-v1.json` under the
static-control results directory. These sensor-depth illustrations explain the
desired visual task but do not solve reference-free initialization on DAVIS.

#### Early depth observability

`probe_static_early_depth_range.py` uses only reference cameras and tracked
coordinates from frames 1–10. It does not use sensor depth or later image
coordinates. The classical linear bound calculation in `frustum_depth_range.py`
first requires a positive-depth witness, then minimizes and maximizes first-
camera axial depth over the closure of the box intersection. Boundary endpoints
are not necessarily positive-depth witnesses. Bounds are numerical solver
results, not calibrated statistical intervals or a novel triangulation method.

All 48 slot/tolerance pairs are in `early-depth-range-v1.json`. At one pixel,
all 11 trackable slots are infeasible. At two pixels, nine are feasible; five
have unbounded upper depth and four have finite intervals. At four and eight
pixels, all 11 are feasible with unbounded upper depth. The twelfth slot is
insufficient throughout. No tolerance was selected as an admission threshold.
Tests check analytically known disparity bounds, world-coordinate and scale
changes, zero-baseline unbounded depth, and inconsistent observations.

These results explain why a single early fitted point is a poor initialization
on this short-baseline control. They do not imply that all later projections
are uncertain: pure rotation can constrain image position without constraining
depth. A usable abstention criterion must measure uncertainty in the projected
position, not reject every point with unbounded depth. The diagnostic remains
outside the manuscript and does not establish a reference-free DNF decision.

`probe_static_projection_range.py` then bounds query-image coordinates directly
using early frames 1–10 and query cameras at frames 11, 45, and 89. No query
image coordinates or sensor depths enter the calculation. The linear-fractional
problem is written in homogeneous coordinates with query depth fixed to one;
points at infinity are allowed in the closure. Bounds are conditional on
positive query depth, not a visibility certificate. Separate coordinate extrema
need not form an attainable rectangle corner. No bounds are clipped to the image.

All 144 slot/tolerance/query combinations are retained in
`projection-range-v1.json`. At four pixels, all 11 eligible slots have finite
coordinate bounds. Median horizontal widths at frames 11, 45, and 89 are
7.80, 931.18, and 1282.10 pixels; median vertical widths are 5.99, 46.24, and
174.47 pixels. The image is 640 by 480 pixels. Thus this early window supports
a relatively precise immediate prediction but not precise later placement.
This is a reference-camera diagnostic, not validation of estimated cameras or
a calibrated abstention threshold.

Analytic tests distinguish unconstrained depth with bounded same-view projection
from translation-induced unbounded projection, check reverse-view infeasibility,
known-disparity prediction bounds, and world-coordinate invariance. A runner
test replaces all later image coordinates and sensor data and verifies that
every predicted bound remains unchanged. These classical geometric bounds are
an observability diagnostic, not the paper's claimed novel contribution.

#### Fitting-window sensitivity

The projection-range runner also tests fixed prefixes ending at frames 10, 30,
and 60, with the same query at frame 89. Every original slot and the full
1/2/4/8-pixel tolerance grid remain in the records
`projection-prefix-{10,30,60}-v1.json`. Every frame in a fitting window must
have an observation; missing tracks are not replaced or silently dropped.
The original ten-frame runner is archived as `executed-projection-range-v1.py`.
The new ten-frame final-query results reproduce its archived output exactly.

At eight pixels, feasible counts out of the original twelve slots are 11, 10,
and five. Median final-query horizontal ranges are respectively 1581.33,
132.35, and 26.25 pixels; vertical ranges are 223.16, 28.24, and 11.00 pixels.
At four pixels, feasible counts are 11, nine, and three, with horizontal
medians 1282.10, 56.52, and 7.97 pixels. The other slots are explicitly
insufficient or geometrically infeasible; these are not camera-failure labels.
Medians use different surviving sets and must not be interpreted as an
unconditional improvement. For every common feasible slot/tolerance pair,
longer-window intervals were checked to lie within the shorter-window intervals.

This development comparison supports testing longer prefixes while retaining
the precision/coverage tradeoff. It does not select a fitting duration or pixel
tolerance for evaluation. Tests reject queries inside the fitting window and
verify that arbitrary changes to later image coordinates and sensor depth do
not change any bounds for all three prefix lengths. No manuscript claim or
human validation result follows from this sensitivity study alone.

#### Conditional final-observation check

`probe_static_query_consistency.py` tests whether the fitting observations and
the frame-89 observation admit a common positive-depth point. It separates
missing fitting observations, unestablished fitting geometry, missing query
observations, compatible queries, contradictions, and numerical failures.
Query-image data enter only this comparison, not the preceding projection bounds.
All 84 prefix/tolerance/signed-yaw cases are in `query-consistency-v1.json`.

At eight pixels, the 10- and 30-frame prefixes accept all five available
final-frame tracks for every tested yaw, including ±5 degrees. With 60 fitting
frames, all five accept the reference query and all five contradict either
five-degree query perturbation. Both signs of one-degree and quarter-degree
perturbations remain compatible. Seven original slots have insufficient fitting
observations in the 60-frame case; they are not accepted or rejected cameras.
All tolerances remain recorded; eight pixels is not a calibrated DNF threshold.

The fitting poses are unmodified reference poses. Only the final camera
orientation is changed, so this experiment does not test errors accumulating
within the fitting window or distinguish a step from a ramp between the prefix
and final frame. Every fitting record was checked to remain identical across
query perturbations. Compatible witnesses were independently reprojected to
verify positive depth and all original pixel bounds. This demonstrates a narrow
controlled sensitivity result, not a validated novel evaluator, and remains
outside the manuscript pending broader validation.

#### Estimated-camera query checks

`probe_predicted_query_consistency.py` applies projection bounds and conditional
query comparisons to the saved DAGE and DA3 predictions. RGB bytes are decoded
and rehashed; input records, executed runners, prediction hashes, model names,
and all 90 rigid poses are verified before evaluation. No reference poses or
sensor depth enter the geometric checks. Both methods use the same published
calibration and tracks. Results are in `predicted-query-consistency-v1.json`.
The inference models saw all 90 RGB frames: restricting geometric fitting to
a prefix is not causal prefix-only model inference or independent validation.

With 60 fitting frames and eight-pixel tolerance, both methods are compatible
on all five final-frame tracks; seven original slots are insufficient. Median
horizontal/vertical prediction widths on those five tracks are 30.29/15.60
pixels for DAGE and 28.70/14.24 for DA3. At four pixels, DAGE remains compatible
on all five while DA3 has two compatible queries and three contradictions;
median widths are 14.06/6.89 and 10.17/4.70 pixels respectively. Tighter bounds
do not by themselves establish higher accuracy. No tolerance is selected here
as a quality threshold, and this comparison does not establish a general ranking.

All three prefix lengths and all four tolerances retain every slot and separate
fitting failure from final-query contradiction. Tests modify the final image
observation and verify that only the comparison changes, not prediction bounds.
Every compatible combined witness was independently reprojected to check
positive depth and original pixel bounds. These development results remain
outside the manuscript until the evaluation protocol has broader validation.

#### Separate-sequence replication protocol (before results)

The next sequence is `rgbd_dataset_freiburg3_long_office_household`, selected
from the [official TUM catalog](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg3_long_office_household)
for its different household/office scene while retaining Freiburg-3 calibration.
This choice is based on the catalog description, before inspecting selected
images, tracking outcomes, or prediction scores. It is a separate-sequence
replication, not a guarantee that the models never encountered this data in
training or that every observed point is static.

Freeze the existing sampling and analysis settings: first 90 chronological RGB
frames at stride three; original 12 grid/LK slots; prefixes ending at 10, 30,
and 60; final query 89; all 1/2/4/8-pixel tolerances; both methods and their
existing inference settings; published Freiburg-3 calibration; no pose alignment
for consistency tests. Controlled final-yaw offsets remain 0, ±0.25, ±1, and
±5 degrees. Preserve all missing tracks, unavailable reference associations,
infeasible fitting windows, and numerical failures. Do not move the sample or
window to obtain more favorable coverage. Reference/depth associations use the
unchanged nearest-within-20ms rule and are validation-only. If a fixed reference
window is unavailable, report it as unavailable rather than interpolating or
substituting frames. Evaluate estimated cameras independently of reference availability.

The source archive is the catalog's
`https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz`;
HEAD returned 1,483,556,251 bytes before acquisition. Store source data on Anima's
`/data3` disk. Any implementation changes needed to handle missing data must be
tested and documented separately from these frozen choices. No paper results
are claimed before the replication completes.

The archive transfer completed on Anima under
`/data3/kmatzen/plumbline/research/office-rgbd-replication-v1/office.tgz`.
It is exactly 1,483,556,251 bytes, passed `gzip -t`, and has SHA-256
`c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c`.
The staging utility now accepts the explicitly named office sequence without
changing sampling or association rules. Seven staging tests pass for both
supported archive roots and unsafe-name rejection; the original executed
stager is preserved under `executed-staging-v1.py` in the first control's results.

`build_static_rgb_tracks.py` separates the unchanged grid/LK extraction from
reference/depth diagnostics, so unavailable reference poses do not block the
estimated-camera tests. It retains the original OpenCV grayscale decoder.
Re-running it on the first control exactly reproduces every one of the 12×90
original track records; that check is retained as `rgb-only-tracks-v1.json`.
The office sample has not yet been scored. Staging began only after the transfer
completed and passed archive integrity checks. The unchanged DAGE/DA3 runners
and query-chunk helper have been copied to the separate replication directory;
16 targeted staging, input, runner, and geometry tests passed before inference.

`audit_reference_availability.py` reconstructs every nearest-timestamp reference
association from the hashed raw index and checks rigid poses without replacing
missing frames. It reports availability separately for the fixed fitting windows
and final query. Tests retain out-of-tolerance associations, duplicate timestamps,
and invalid zero quaternions, reject altered associations, and confirm that a
missing frame zero does not shift windows starting at frame one. On the original
control it reproduces 89 available poses and one out-of-tolerance association;
all three fixed windows remain available. That audit is retained as
`reference-availability-v1.json`.

Office staging has now completed. All 90 selected RGB frames, raw indices,
depth associations, and 86 native uint16 depth images were reverified locally.
Depth associations are unavailable at frames 13, 61, 64, and 66; reference pose
zero is unavailable, but all fixed fitting/query windows are available. The
unchanged RGB tracker retains all twelve slots: twelve, nine, and five have
complete observations through frames 10, 30, and 60; four reach frame 89.
All records are under `results/office-rgbd-replication-v1`.

The fixed reference-camera comparison has completed all 84 cases. At eight
pixels, prefixes ending at 10 and 30 accept all four final-frame tracks for
both ±5-degree query errors and the reference. The 60-frame prefix accepts
all four reference queries and contradicts all four for either five-degree
error. Seven slots lack complete fitting observations and one lacks the final
observation. All 1/2/4/8-pixel and signed-yaw results remain recorded, without
selecting a threshold. Compatible witnesses were independently reprojected.
Median horizontal prediction widths at eight pixels are 994.73, 142.93, and
32.20 pixels over twelve, nine, and five feasible slots respectively; these
different surviving sets must not be presented as unconditional improvement.

Both DA3 and DAGE completed the office run. Their decoded inputs, 90 rigid poses,
archived runners, and saved prediction hashes were verified; DA3's helper and
loader were also verified. Checkpoint files, implementation source, and settings
match the first control exactly, and the two methods consumed identical decoded
RGB frames. The paired estimated-camera comparison is retained in
`predicted-query-consistency-v1.json`.

At both four and eight pixels with 60 fitting frames, both methods are compatible
on all four observed final-frame tracks. Seven slots have insufficient fitting
observations and one has no final observation. On the four comparable slots,
median horizontal/vertical prediction widths at four pixels are 15.58/6.49 for
DAGE and 14.81/5.79 for DA3; at eight pixels they are 36.81/17.28 and 34.98/16.44.
The four-pixel contradiction difference observed on the first sequence does
not replicate here. All 24 method/prefix/tolerance cases remain retained; every
one of the 61 compatible combined witnesses was independently reprojected to
check positive depth and original pixel bounds.

For context only, Sim(3)-aligned ATE over the 89 associated reference frames is
0.006997 m for DAGE and 0.008564 m for DA3. Translation/rotation RPE is
0.006783 m / 0.239158 degrees and 0.008862 m / 0.266156 degrees respectively,
using the existing trajectory metric functions and nearest-time reference poses.
These are not functional-failure labels. Two sequences with sparse tracked
points do not establish which method is generally better.
This replication supports only the narrow controlled endpoint sensitivity
finding, not human agreement, general camera ranking, or a validated DNF rule.

#### Paired-image identity diagnostic

`build_track_identity_pairs.py` creates matched native-resolution crops around
the tracked locations at frames 1 and 89, without camera/depth filtering. The
five available pairs on the first control and four on the office sequence are
retained, alongside explicit unavailable-slot records and identical-image
positive controls. Source images, crop coordinates, and output hashes are recorded
in each sequence's `identity-pairs-v1.json`. Temporal pairs have no correctness
labels. Duplicate controls cover only available temporal slots; there are no
negative controls, so they cannot establish specificity. A pixel test confirms
identical duplicate panels and unobscured center pixels. All nine temporal
sheets were inspected by the assistant for rendering, not labeled by a human.

Before model responses, the bounded pilot selects the first available slot in
each original slot order: slot 1 on the first sequence and slot 4 on office.
Run each temporal pair and its duplicate control, four images total, with the
pinned Qwen2.5-VL-7B NF4 setup, fp32 attention scores, greedy decoding, and the
previous 576 minimum / 768 maximum visual-token budget. The new `track_identity`
prompt asks for the same exact marked detail, different detail, or abstention,
with descriptions of both panels; it explicitly excludes camera/DNF judgments.
Parsing checks the response schema and evidence-panel names only. This pilot
cannot provide temporal accuracy, training labels, or camera-quality evidence.

The four-image job was launched in
`/data3/kmatzen/plumbline/research/track-identity-pairs-v1` on Anima after 16
crop/parser/input-audit tests passed. The executed runner and numerical helper
are retained locally in `results/track-identity-pairs-v1`. The first result file
appeared and the existing connection proceeded to another model load, but its
response has not been inspected: a separate result-copy connection failed when
SSH reported the YubiKey device unavailable. The existing pilot connection
remains active. No correctness or response-format result is inferred from file
existence or successful model loading.

SSH access subsequently recovered without using the token typed into chat.
The first two completed outputs were retrieved and checked against image and
executed-runner hashes. Both processed-image audits contain finite tensors and
608 image tokens with matching placeholders. The raw duplicate-control and
temporal responses both abstain, citing ambiguity/blur without identifying the
marked detail. Both wrap their JSON in Markdown fences and therefore fail the
unchanged strict parser. These are preserved as raw outputs, not accepted labels.
The two office responses have now completed and been inspected. Both also
abstain, including the identical-image control, and both fail the unchanged
strict parser because they wrap JSON in Markdown fences. All four runs use the
same prompt/checkpoint/settings, contain distinct finite processed pixel
payloads with 608 matched tokens, and reproduce their saved parse outcomes.
The completed raw records are retained in `results/track-identity-pairs-v1`.

Decision: do not expand this VLM setup or use its outputs as identity gates,
training labels, or camera-quality evidence. Removing Markdown fences would
address formatting only, not the lack of useful identity judgments. This
four-input diagnostic is too small and lacks negative controls to estimate
accuracy or specificity. Keep its development record outside the manuscript.
Continue the geometric evaluation separately; it must not be described as
validated by a VLM. Revisit model-assisted identity checking only with a materially
different model/setup and a stated validation plan, not another equivalent
prompt or image-scale sweep.

DA3 has now completed the same 90-frame control. Inputs are byte-identical to
DAGE after RGB decoding, and the checkpoint, adapter source files, settings,
and query-chunk helper hash match the earlier DA3 DAVIS run. The new runner is
`run_static_da3.py`; outputs, the executed runner, input loader, and attention
helper are retained in `results/static-rgbd-control-v1/da3-v1`. It uses 252-pixel
upper-bound resizing and 256-query chunks, with all keys/values retained.
Kernel equivalence is unit-tested; end-to-end equivalence with unchunked
inference is not established. Runtime including load was 24.04 seconds and
peak allocated GPU memory 5.11 GB; these timings are not a controlled benchmark.

An adapter inspection clarifies the inference inputs: the earlier DAGE runner
passed the optional calibration argument, but the adapter does not forward it
to the network. Both networks therefore infer cameras from RGB only. The
geometric comparison uses the same published Freiburg-3 calibration for both,
not either model's estimated intrinsics. DA3's processed-image intrinsics are
retained separately and are not interpreted as native-resolution calibration.

`results/static-rgbd-control-v1/da3-point-support-v1.json` reproduces the eight
reference/rotation cases exactly and adds DA3. The preceding DAGE comparison
script is archived as `executed-point-support-dage-v1.py`. Model labels are
checked against each manifest so a DA3 result cannot silently be called DAGE.
The full-track comparison on the same 11 checked slots is:

| Axis-wise tolerance (px) | Reference witnesses | DAGE witnesses | DA3 witnesses | Later-yaw control witnesses |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 3 | 0 | 0 |
| 4 | 4 | 11 | 4 | 0 |
| 8 | 11 | 11 | 11 | 0 |

One additional slot is insufficiently observed for every case, and the shared
later support is 579 / 948 planned observations. DA3's early window supports
10 slots at one pixel and 11 at the other tolerances. Over original frames
1–89, its Sim(3)-aligned ATE-RMSE is 0.007964 m, translation RPE-RMSE is
0.007796 m, and rotation RPE-RMSE is 0.358041 degrees. These are comparable with
the DAGE metrics above, not a new sampled estimate of method superiority.
The single favorable scene and close trajectory errors do not justify a
general ranking, significance claim, or functional-failure label. This
comparison remains in the development records, outside the manuscript.

#### RGB-D evidence validation

The existing TUM development subset contains only RGB and associated reference
poses. `stage_rgbd_validation.py` stages depth from an official sequence archive
into a separate directory, matching the existing RGB sample without selecting
on tracking outcomes. It checks each RGB file against the experiment manifest,
associates the nearest depth timestamp within 20 ms (earlier time on ties), and
retains unmatched frames and repeated depth associations. The 20 ms tolerance
is a declared development association policy, not a calibrated validity bound.
Archive, index, RGB, and selected depth hashes are recorded.

`rgbd_track_audit.py` backprojects the nearest depth pixel and transforms it
with a reference camera-to-world pose. The
[official format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
specifies registered RGB/depth pixels, axial depth divided by 5000, and zero
as missing. The audit retains the 3x3 neighborhood's nonzero depth range and
missing count to expose depth-edge ambiguity; it does not fill holes or
reject inconvenient measurements. Synthetic tests cover camera translation,
a shifted track, axial versus radial depth, and missing observations.

The intended experiment measures world-position change along the previously
selected image tracks, without passing reference poses or depth to the
reference-free evaluator. Scene motion, incorrect tracks, occlusion, sensor
noise, and calibration/time association can all produce change. Therefore
these measurements are additional validation evidence, not automatic
staticness, tracking-correctness, or camera-DNF ground truth. RGB-depth and
RGB-reference timing offsets must both be reported. The development results
below are retained here, not included in the paper's claims.

The audit command now consumes the existing anchor-track record and a staged
depth manifest, verifies both against the source experiment, and checks depth
file hashes before reading measurements. It reports both timestamp offsets
per frame and retains every original track/frame pair. Displacement is always
relative to the frame-zero depth observation: if that observation is missing,
later world points remain in the record but displacement is unavailable. It
does not substitute a later origin or choose tracks based on these distances.
The per-track median/max summaries exclude frame zero and explicitly report
their number of comparable observations. An end-to-end synthetic fixture
checks this missing-origin behavior and detects changed depth bytes.

The development command is:

```sh
uv run python -m paper.rgbd_track_audit \
  paper/results/dense/dage-walking-xyz-v2 \
  paper/results/dense/tum-anchor-xyz-v1.json \
  /path/to/staged-depth \
  /path/to/new-audit.json
```

The walking-XYZ archive was downloaded and its gzip integrity checked on Anima.
The separate staging directory is
`/data3/kmatzen/plumbline/research/rgbd-validation-v1/selected-depth`;
the local copy is `tmp/rgbd-validation-v1/selected-depth`.
`results/dense/tum-depth-manifest-xyz-v1.json` retains archive, source,
staging-script, index, and depth-image hashes. Of 90 RGB frames, 88 have a
depth association within 20 ms. Frames 0 and 64 do not; their nearest depth
offsets are -30.015 and -25.973 ms. Among the 88 associated frames, absolute
RGB-depth offsets have median 0.018 ms and maximum 6.168 ms; absolute
RGB-reference offsets have median 2.120 ms and maximum 4.868 ms.

The initial audit, `results/dense/tum-rgbd-track-audit-xyz-v1.json`, therefore
cannot measure displacement from frame zero for any slot. Its executed source
is retained as `tum-rgbd-track-audit-v1-executed.py` in the same directory.
The v2 audit preserves every original measurement and adds two summaries that
do not require an origin: the largest separation between any two measured 3D
observations, with the two frame indices, and distances between measured
observations in consecutive original frames. Missing frames are not bridged,
interpolated, or replaced. This addition was made after observing the missing
frame-zero depth; it is exploratory, not a predeclared validation endpoint.

Across all 12 slots and 90 frames, there are 136 measured observations, 859
missing image tracks, 73 zero-depth samples, and 12 observations without a
depth association. These mutually exclusive counts retain all 1,080 planned
observations. Seven slots have measured 3D observations:

| Slot (zero-based) | Measured / 90 | Measured frame range | Largest separation (m) | Adjacent pairs / 89 | Median adjacent distance (m) |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 | 3 | 1–5 | 1.216 | 1 | 1.216 |
| 5 | 6 | 1–7 | 0.075 | 4 | 0.039 |
| 6 | 46 | 1–53 | 0.314 | 40 | 0.031 |
| 7 | 25 | 1–25 | 1.138 | 24 | 0.048 |
| 8 | 2 | 1–2 | 0.015 | 1 | 0.015 |
| 9 | 3 | 1–3 | 0.031 | 2 | 0.024 |
| 10 | 51 | 1–51 | 1.250 | 50 | 0.013 |

Slots 0, 1, 2, 3, and 11 have no measured observations. Smaller spread over a
shorter observed interval is not evidence of a better track. Large spread also
does not identify its cause: slot 10's diameter witness includes frame 12,
whose 3x3 depth neighborhood ranges from 2.365 to 3.208 m. Slot 4 samples
depths above 8 m with broad local variation. No depth-range cutoff was applied.
Slot 7 has narrow local depth ranges at both diameter endpoints, but that alone
does not distinguish scene motion from a track following different surfaces.
The source-image inspection below examines these witnesses without converting
the distances into automatic failure labels.

`render_rgbd_witnesses.py` renders frame zero and both diameter endpoints for
every slot with a measured diameter. It checks source RGB, depth, track, and
audit hashes. Each figure shows the full source image, an enlarged RGB crop,
and the matching depth crop on a fixed 0–10 m scale. The local figures are in
`tmp/rgbd-witnesses-v1`; their hashes and selected frames are retained in
`results/dense/tum-rgbd-witness-manifest-v1.json`. Selection uses the observed
3D spread and is therefore exploratory, not a blinded validation set.

Assistant inspection of all seven figures suggests distinct evidence problems:

- Slot 4 marks a small background fixture. The target appears similar at the
  endpoints, but its distant depth values vary substantially. These images
  alone do not establish a tracking error.
- Slot 5 marks a keyboard edge near a cable; only six depth-backed observations
  exist. The shown endpoints appear consistent, but cannot certify the full
  track or a precise surface correspondence.
- Slot 6 lies near the monitor support and adjacent background. Its displayed
  location remains similar, but the precise surface sampled by depth is
  ambiguous near the boundary.
- Slot 7 lies on a person's patterned shirt. The person changes position; the
  repeated pattern and blur also make exact point identity uncertain. This is
  not suitable as an assumed static-world anchor.
- Slots 8 and 9 lie on a person's forearm and near the hand/knee. Their small
  spreads cover only two and three depth-backed frames, respectively, and do
  not establish stationary scene points.
- Slot 10 follows the apparent intersection of a chair outline and the desk
  edge. Such an occlusion intersection need not denote the same physical point
  as the viewpoint changes. Its depth crop also crosses a surface boundary.
  The large spread is not evidence of camera failure.

These are assistant observations, not independent human labels, and are not
training targets. They motivate a source-evidence qualification task before
camera scoring: distinguish a named surface feature from a moving subject,
repeated texture, or an occlusion intersection. A visible object or plausible
2D track is not sufficient. The next evaluation should test that distinction
on a separately selected source-only sample, with geometric results hidden.
No existing measurements, selections, or camera rankings were changed by this
inspection, and these development observations are not added to the manuscript.

The next diagnostic uses `build_source_crops.py` to retain all 12 original
frame-zero grid targets, without selection on later survival, depth, or camera
errors. Each input is a native-resolution 161x161 RGB crop centered exactly on
the rounded target pixel. Outside-image pixels are neutral gray; they are not
replicated scene content. No marker, depth image, or camera diagnostic is shown.
The source crops remain local in `tmp/source-crops-v1` and on Anima in
`/data3/kmatzen/plumbline/research/source-crops-v1/source-crops-v1`.

`run_vlm_dnf_pilot.py --task crop_description` asks for one or two sentences
describing the center surface or object, explicitly excluding camera quality
and temporal claims. Nonempty prose is recorded without JSON parsing; this
does not certify visual correctness. This changes both presentation and prompt
relative to the earlier identity pilot, so it is not a controlled single-factor
comparison. It is also the same development sequence already inspected, not
held-out evaluation.

The 12-item batch completed sequentially on Anima using the pinned 7B
checkpoint, NF4, fp32 language-attention scores, a 768-token image ceiling, and
greedy generation. The input manifest, executed runner/helper, and completed
responses are retained in `results/source-crops-v1`. All 12 image,
runner/helper, checkpoint, prompt, and settings hashes were checked. All inputs
were finite with matching image-placeholder counts; all outputs were nonempty
prose. Generation took 18.31–28.59 seconds per item, excluding loading. These
are execution checks, not accuracy measurements. Every processed crop has 36
image tokens; the ceiling does not force upsampling.

Assistant comparison with the rendered source crops shows mixed specificity.
Slot 5 names the keyboard edge. Slot 7 identifies a checkered shirt, but also
claims it is tucked in, which is not supported by the visible hem. Slot 6 names
the whole display rather than the marked monitor-support boundary. Slot 8
describes a hand resting on a desk instead of the center near the forearm.
Slot 10 describes a possible desk edge and cannot distinguish its surface
from an outline or intersection; it does not identify the chair/desk occlusion
intersection. Slot 11 supplies a hand-near-chest detail that is not supported
by the crop. These observations do not produce a numerical correctness score:
there are no independent target descriptions or human judgments for this set.

The captions do not establish a reliable source-point qualification rule. No
DNF decisions, staticness labels, training targets, or camera rankings are
derived from them. This experiment changes both prompt and presentation from
the earlier identity task, so completion without JSON errors is not evidence
of improved visual grounding. A useful next controlled check would preserve
the crop, prompt, and model while varying only presented image scale; any such
follow-up must remain separate from held-out evaluator validation. The paper
is unchanged. The non-weight regression suite passed 1,836 tests, with one
missing KITTI-bundle skip and six weight-backed tests excluded; that verifies
software behavior, not the proposed metric's scientific validity.

A matched scale diagnostic completed in
`/data3/kmatzen/plumbline/research/source-crops-scale-v1`. It reuses all 12
original crop files, the exact caption prompt, checkpoint, quantization,
attention implementation, greedy decoding, and 768-token ceiling. The only
intended inference-setting change is `--min-visual-tokens 576`, implemented
through the processor's `min_pixels` argument. This enlarges the same sampled
image; it does not add scene information. Actual processed grid and token
counts must be verified, not assumed from the requested minimum.

The updated runner first executes slot 0 without a minimum as a replay control,
then processes slots 0–11 at the new scale. Its source and attention helper are
retained in `results/source-crops-scale-v1`. Default arguments preserve the
previous processor options; tests cover the optional size bounds and reject
invalid minima before model loading. All results will be retained. The completed
`default-slot-0.json` replay exactly matches the original processed-input audit,
raw/parsed response, image, prompt, checkpoint files, settings, and attention
helper/implementation hashes. All 12 scaled runs now pass the paired checks
described below. This remains a development diagnostic, not held-out validation.

`compare_crop_scale.py` checks
the paired image, prompt, checkpoint, software, quantization, attention, and
decoding settings, allowing only the declared minimum-pixel change. All 12
actually use 576 rather than 36 image tokens. The earlier partial accounting in
`results/source-crops-scale-v1/comparison-partial-v1.json` retains all 12 slots:
two matched responses and ten then not yet retrieved, with no inferred execution
status for those ten. The completed accounting is
`results/source-crops-scale-v1/comparison-complete-v1.json`, with 12 matched
pairs and no unavailable rows. Raw responses are kept alongside the comparison; changed
wording is not counted as an improvement. All 12 responses changed, but the
inspection does not establish improved target recognition. The comparison test rejects altered images, prompts, seeds,
decoding settings, incomplete attempts, and an incorrect realized image scale.

Assistant inspection finds persistent localization and factual errors. At the
larger scale, slot 6 describes a keyboard rather than the monitor-support
boundary. Slots 8 and 9 describe desk surfaces although their centers lie near
the forearm and hand/knee. Slot 10 identifies a chair but adds an unsupported
patterned backrest and armrest, and does not explain the chair/desk occlusion
intersection. These are illustrative inspection observations, not independent
accuracy labels. The larger scale changes the model's answers but does not
establish a dependable source-point qualification rule. The crop-only caption
setup will not be used to accept or reject camera evidence or generate training
labels. No further scale sweep is justified by this diagnostic alone; any
replacement needs a defined target-recognition task and independently checked
answers, rather than more detailed prose. No findings from this development
comparison are added to the paper.

#### Reference timing sensitivity

`probe_tum_pose_timing.py` reproduces the retained reference poses from the
original trajectory file before comparing timestamp policies. It interpolates
translation linearly and rotation with quaternion SLERP at RGB timestamps,
without extrapolation or interpolation across gaps above 50 ms. On the XYZ
development clip, absolute associated-pose offsets have median 2.13 ms and
maximum 15.70 ms. Frame zero precedes the reference trajectory; the original
90-frame comparison is therefore unavailable, retained as
`results/dense/tum-pose-timing-xyz-v1.json` with its executed source.

The explicitly separate v2 comparison starts at original frame 1 for both
variants, fits frames 1–11, and evaluates frames 12–89. No image tracks are
recomputed or selected by outcomes. Both variants have 141 comparable later
observations across four slots, out of 936 planned track/frame pairs. Their
aggregate medians are 3.217 px (associated poses) and 3.228 px (interpolated).
The median absolute paired change is 0.350 px, but it is not uniformly small:

| Slot | Comparable observations | Median absolute change (px) | Maximum (px) |
| --- | ---: | ---: | ---: |
| 1 | 40 | 0.601 | 1.664 |
| 6 | 46 | 0.230 | 1.809 |
| 7 | 14 | 54.758 | 72.858 |
| 10 | 41 | 0.246 | 1.856 |

Slot 7 already has very large residuals under both policies: at frame 20,
692.115 versus 619.257 px. Thus the large change does not represent a switch
from good tracking to failure. The timing comparison demonstrates sensitivity
in this early-fit diagnostic, not that either reference association resolves
the scene-motion, track-correctness, or conditioning confounds. Original
production references and reported method scores remain unchanged.

The stationary TUM clip supplies a second development check. Its trajectory
file has two different poses at timestamp 1341846251.4587, outside the selected
clip. The original global strict-monotonicity check rejected the file; that
attempt is retained as `tum-pose-timing-static-failed-v1.json`, alongside the
executed runner. Interpolation now rejects duplicates when the query or either
bracketing timestamp uses them; unrelated duplicates do not invalidate the
whole sequence. No duplicate poses are averaged. The existing loader's
last-row timestamp mapping is used only to reproduce its retained reference.

The stationary clip also starts before the reference span. Its full-window
unavailable result is retained as `tum-pose-timing-static-v2.json`. The v3
comparison uses the same explicit frame-1 start and frames 1–11 fit as the XYZ
follow-up. All twelve slots fit under both policies, but only 380 of 936 later
observations are comparable. Associated/interpolated aggregate medians are
7.922/7.045 px; the median absolute paired change is 0.952 px and the maximum
is 256.146 px. The change is not uniformly beneficial: slot 0's median increases
from 52.792 to 291.308 px, while slot 5's decreases from 5.914 to 4.943 px.
Changing timestamp policy does not resolve these early-fit diagnostic
confounds or validate a failure threshold. The two sequences are development
examples, not a sampled estimate of general timing sensitivity.

#### Source-track comprehension pilot

The corrected 7B NF4 configuration was subsequently evaluated on the same three
source-identity inputs and identical prompt used by 3B. All three attempts
completed and their raw answers abstain because the marked detail appears too
small and blurred; all use Markdown fences and fail strict parsing. Thus the
larger configuration supplied no scene identities on this pilot, even apart
from formatting. This does not establish that the details are objectively
unidentifiable or that the model cannot perform broader scene interpretation.
That pilot alone lacks a clear-target positive control and human answers, so excessive
abstention versus appropriate uncertainty remains unresolved. Records are in
`paper/results/source-identity-7b-v1/`; input and prompt hashes match the 3B trials,
and runner hashes match the archived corrected 7B runner. No prompt retries,
labels, staticness judgments, or DNF decisions were generated from these answers.

`build_identity_controls.py` constructs a follow-up coarse-identity sanity check:
an orange triangle, blue rectangle, and purple circle on a plain background.
Three inputs differ only in which shape contains the cyan marker and its
magnified crop. The exact source-identity prompt, pinned 7B checkpoint, NF4
configuration, 768-token visual budget, and corrected fp32 attention-score
implementation are unchanged. These are generated images, not video frames;
the prompt's source-video framing is therefore a limitation. The marked
interiors are uniform, so even identifying the shape would not establish a
unique trackable point. This check addresses coarse target identification,
not staticness, temporal tracking, or camera quality. Inputs and their known
shape identities are retained in `results/clear-identity-controls-v1/`.

All three attempts completed. Every raw answer says `abstain`, with almost
identical claims that the marked detail is too small and blurry; none names
the shape. Every answer also uses Markdown fences and fails the unchanged
strict JSON parser. Input, runner, helper, and prompt identities were checked
against the retained inputs and current executed source. These results do not
demonstrate coarse target identification, and weaken an interpretation of the
earlier abstentions as evidence that only the real targets were unreadable.
They do not isolate prompt effects, model limitations, quantization, or the
image-processing path. A generic image-description and processed-input audit
is needed before assigning this configuration any camera-failure judgments.
No accepted labels or training examples were produced.

The follow-up in `results/description-controls-v1/` audits the processed input
and asks the same 7B configuration to describe the triangle-target image, using
only “Describe the visible shapes, colors, and layout in this image.” The
payload is finite and nonconstant, with grid `[1, 40, 68]`, merge size 2, and
680 image placeholders matching the grid. This verifies payload presence and
token accounting, not the correctness of every downstream operation.

The description names the orange triangle, blue rectangle, purple circle,
left-to-right layout, cyan ring in the triangle, and orange crop. It also
incorrectly claims there is a second cyan ring inside the purple circle.
Thus this configuration demonstrates some image-conditioned description but
not reliable factual reporting. A nonempty description is stored verbatim;
the description task's parser does not certify its visual correctness or
convert it to a camera label. The exact executed runner and attention helper
are archived alongside the output. This is a development diagnostic, not a
held-out accuracy result or a revised camera-evaluation prompt.

An original-identity-prompt repeat using this same instrumented runner again
returns the same fenced abstention about a small, blurry target. Its processed
pixel hash, grid, image tokens, source image, checkpoint-file hashes, generation
settings, runner, and attention implementation match the description trial.
This matched pair shows prompt-dependent behavior on this one image; it does
not identify which instruction causes abstention or establish a generally
better prompt. In particular, the description's invented extra ring prevents
treating its greater willingness to answer as sufficient evaluator reliability.

The larger-model follow-up is Qwen2.5-VL-7B-Instruct at pinned revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, staged in Anima's `/data3` HF cache.
`bitsandbytes==0.48.2` is installed without dependencies in the isolated directory
`/data3/kmatzen/plumbline/research/bnb-0.48.2`, rather than modifying the shared
tracker environment. A CUDA Linear4bit smoke test produced finite outputs on
compute capability 6.1. The runner's optional `--quantization nf4` uses NF4,
double quantization, fp16 compute, eager attention, and a single-GPU device map;
the default unquantized mode is unchanged. It records the configuration, package
version, and actual count of quantized linear modules, and rejects a requested
quantization that produces no quantized layers.

The initial 7B control job uses the same geometry prompt, two control images,
768-token visual budget, greedy decoding, seed, and response cap as the retained
3B trials. This is a comparison of model configurations, not an isolated
parameter-count experiment, because precision also changes. The remote attempt
directory is `/data3/kmatzen/plumbline/research/qwen7b-nf4-controls-v1`.
Both scheduled controls completed with 358 quantized linear layers. Each output
is 256 exclamation marks, reaches the 256-token cap, and fails JSON parsing.
Generation took 119.63 and 120.29 seconds for on/off controls, respectively.
Input and prompt hashes match the retained 3B trials, and the executed runner
hash matches its archived copy. These are unusable responses, not interpretable
geometry judgments; an execution/numerical issue must be investigated before
attributing them to visual reasoning. Both records and the exact executed runner
are retained in `paper/results/qwen7b-nf4-controls-v1/`.

The runner now offers a separate `--numeric-preflight-only` mode: it installs
leaf-module output checks, runs one uncached forward pass without generation,
and records the first nonfinite output's module, shape, dtype, and count. A
completed finite preflight is not an accuracy result. The initial on-line control
preflight stopped on nonfinite output from
`model.language_model.layers.0.self_attn.o_proj`. Its failed record and exact
diagnostic runner are archived alongside the responses. This localizes a
numerical failure before interpreting the punctuation as a visual judgment; it
does not yet distinguish invalid projection weights, upstream attention values,
or arithmetic overflow. Neither scheduled response is overwritten or repaired.

The separate `audit_vlm_attention.py` traced the first language attention call
while returning the original attention output unchanged. Queries, keys, and
values are finite. The unscaled fp16 query-key product contains 157 nonfinite
entries; the same operands multiplied in fp32 are finite with maximum magnitude
66,003.890625, above fp16's finite range. Scaling occurs after this overflowing
product in the installed eager implementation. Its softmax has 891 nonfinite
entries; the fp32-product softmax has none. The trace records the installed
implementation hash and is retained with the failed audit run.

The opt-in `--fp32-attention-scores` path computes only language attention score
products, scaling, masking, and softmax in fp32, then casts probabilities back
to the query dtype for value aggregation. It preserves grouped-query expansion,
masking, dropout semantics, and output layout; vision attention is unchanged.
The helper is separately hashed. Tests cover a constructed fp16 overflow and
agreement with explicit grouped, causal-mask reference arithmetic. This changes
numerical precision, not the prompt, checkpoint, inputs, or output parsing.
The full on-control numerical preflight passed with finite leaf outputs and
final logits in 3.67 seconds (7,083,801,088 peak allocated bytes). This is not
end-to-end equivalence to unquantized 7B inference or evidence of judgment quality.
Both new `fp32-control_on/off.json` generation attempts completed. They return
readable fenced JSON, so both still fail the unchanged strict parser. The raw
on-control verdict is `inconsistent`, incorrectly rejecting the all-zero-distance
control. The raw off-control verdict is also `inconsistent`, matching that
control's aggregate relation, but its explanation incorrectly says panels 1--4
are aligned even though all six distances are 50 pixels. Thus resolving arithmetic
overflow did not establish correct geometric grounding. No output is accepted as
a structured judgment or DNF label. The runner/helper hashes were verified and
their executed versions are archived with the results. These two development
controls do not estimate model-wide reliability or compare unquantized models.

`analyze_vlm_controls.py` produces `paper/results/vlm-control-accounting-v1.json`
for all seven retained geometry attempts: one OOM failure, two 3B fp16 attempts,
two 7B NF4 attempts with overflowing score arithmetic, and two with corrected
score precision. It preserves execution state, strict parsing state, expected
control relation, and accepted verdict separately. There are no accepted
structured judgments. Raw-text observations above remain descriptive audits,
not repaired labels. Tests ensure abstention and execution failure do not count
as correct answers, and that fenced JSON is not silently accepted.

Version 2 (`vlm-control-accounting-v2.json`) adds a separate, explicitly labeled
wrapper-only semantic audit. It recognizes one exact Markdown JSON fence and
applies the same schema validator to its unchanged contents. Prose, multiple
fences, malformed JSON, invalid verdicts, and invalid fields remain invalid. It
does not overwrite raw outputs, change strict API acceptance, fill missing
fields, or promote answers to training labels. This separates a transport failure
from semantic behavior instead of letting rejected formatting hide the latter.
Both the 3B fp16 and corrected 7B NF4 configurations say `inconsistent` for both
controls: each rejects the aligned case and gives the expected aggregate verdict
on the displaced case. Their explanations still contain the documented visual
errors. Two constructed controls do not support a general accuracy estimate.

A separate source-identity VLM pilot uses the first frame of each of these three
examples. It asks only what physical detail the cyan circle marks, or to abstain;
it explicitly prohibits staticness, motion, camera-quality, and DNF judgments
from a single frame. The task is not temporal correspondence validation. The
existing pinned Qwen2.5-VL-3B checkpoint was run on Anima at the previously used
768 visual-token budget, fp16/eager attention, greedy decoding, and a 256-token
response limit. No model downloads, prompt retries, or answer repairs occurred.

All three attempts completed. Each raw answer abstains because the target is
small and blurred. Each also encloses its JSON in Markdown fences and therefore
fails the unchanged strict parser. No accepted identity labels, static-region
labels, or DNF decisions are produced. These three abstentions do not establish
model accuracy or that the physical details are objectively unidentifiable;
human answers are still absent. The raw responses and model, prompt, input,
runtime, and runner hashes are retained under `paper/results/source-identity-v1/`.
Downloaded records were checked against the local runner and exact input-image
hashes. The source-identity schema rejects camera verdicts and unknown panel
names, separately from the older geometry and DNF tasks.

`build_track_review.py` generates three source-only videos from the first three
nonempty first-frame grid slots of the retained walking-XYZ tracks. Selection
does not inspect later survival, poses, fits, or error scores. Each 18-second
video shows the full image, a cyan circle, a 4x detail crop, and frame numbers,
at five displayed frames per second. Lost tracks remove the circle and are
explicitly marked; no further position is claimed. The task asks whether the
circle follows the same physical detail, slips, or cannot be judged. A separate
question asks whether the detail belongs to a fixed part of the room or an
independently moving object, allowing uncertainty. No camera-quality rating is
requested. Partial replies are accepted directly in the conversation.

The local packet is `tmp/source-track-review-v1`; its manifest is retained as
`paper/results/dense/source-track-review-v1.json`. Initial frames and a lost-track
frame were visually checked, the first video's duration was verified, and the
page returned HTTP 200 at localhost:8769. This is a project-author comprehension
and evidence-review pilot, not independent validation. No answers have yet been
recorded. The selection-invariance test verifies that changing later survival
does not change which initial slots are shown.

#### Placement-consistency bounds with missing observations

`placement_bounds.paired_bounds` treats every planned anchor/frame observation
as a binary placement-consistency outcome at a stated pixel tolerance. If N is
the planned count, s the number of measured errors at or below that tolerance,
and u the unavailable count, the hypothetical completed success fraction lies
in `[s/N, (s+u)/N]`. For two methods A and B, subtract B's upper bound from A's
lower bound and B's lower bound from A's upper bound to bound A minus B. A strict
ordering is reported only when that interval excludes zero. There is no
missing-at-random or independent-frame assumption. These are identification
bounds, not statistical confidence intervals. Empty evidence cannot imply
perfect consistency. A missing fit remains unknown, not an automatic failure.

This calculation is conditional on the validity of measured image tracks and
errors. It neither proves staticness nor supplies a meaningful hypothetical
placement outcome where the task itself is undefined. Such cases need a separate
evidence-validity decision. The calculation does not establish functional DNF,
camera accuracy, or a general method ranking.

`analyze_placement_bounds.py` applies a fixed descriptive sweep of 1, 2, 4, 8,
and 16 source pixels to six DAVIS clip/image-tracker cases and two TUM controls.
No operating cutoff is selected. All 588 planned DAVIS observations or 948 TUM
observations remain in the denominator. Of 40 case/tolerance comparisons, 36
remain unresolved. The four strict comparisons favor DA3 placement consistency
on bear at 2 and 4 pixels, under both LK and CoTracker3, even if missing outcomes
are assigned adversarially. This remains conditional on measured evidence and
does not validate a usability preference. Both TUM comparisons remain unresolved,
including reference versus the five-degree rotation control: low coverage can
prevent a robust full-plan conclusion even when measured errors differ greatly.
All bounds and counts are in `paper/results/placement-bounds-v1.json`. Tests
compare small missing-data cases against exhaustive binary completions.

#### Early-fit image anchors with a reference-camera control

The same point-support sweep was applied to the six existing DAVIS training
outputs by `probe_davis_point_support.py`, without reference poses or new
inference. Native predicted calibration is converted back to source pixels with
the retained half-pixel convention. Manifest and prediction hashes are verified.
The existing LK tracks and annotation-zero admission are reused; excluded mask
frames and lost observations remain recorded, and neither is certified static.

At a four-pixel axis-wise tolerance, bear has 11 early witnesses for each method,
but seven full-observed-track witnesses for DAGE and one for DA3. Boat has 11/6
early and 3/0 full-track witnesses for DAGE/DA3. These maximum-constraint results
differ from the earlier typical-placement-error comparisons: one bad observation
can invalidate a complete-track constraint set, and the point can be chosen using
later observations. This is not an improvement in camera ranking or held-out
prediction. No tolerance was chosen as a failure threshold.

The first result record (`paper/results/davis/point-support-lk-v1.json`) exposed
an accounting issue: BMX slot 8 has an early witness but no observed frame after
the 11-frame prefix. A positive `all_observed` solver outcome there says nothing
about later consistency. Version 2 retains the solver result but explicitly counts
such cases as `no_later_observations` rather than full-track support. At eight
pixels, each BMX method has only one positive track with later evidence, containing
three later observations; nine other slots lack a complete early track. Both
versions are preserved. All 163 positive witnesses in version 1 were independently
checked in the original normalized-camera projection convention. Version 2 changes
coverage reporting, not the underlying constraints or tolerance sweep.

`probe_point_support.py` tests a set-valued alternative to trusting one fitted
point. For each complete 11-frame prefix, a linear program searches for any
positive-depth 3D point whose projections lie within axis-wise pixel boxes
around the tracked locations. A separate full-track check uses all available
observations and records missing frames. It is joint consistency, **not held-out
prediction**: later observations constrain the full-track solution. Staticness,
correct correspondence, calibration, and synchronization remain assumptions.
Solver infeasibility is numerical evidence under those assumptions, not a DNF
certificate. A positive witness also does not prove the trajectory correct.

The fixed tolerance sweep is 1, 2, 4, and 8 pixels, with no selected operating
cutoff. On walking-static at 4 pixels, reference cameras admit early positive
witnesses for all 12 tracks, including the three rejected by the earlier fitted
point's depth. Nine tracks admit full-track witnesses; the later five-degree yaw
control admits none. At 1 pixel, all 12 reference tracks are solver-infeasible
even on the early prefix. At 8 pixels, reversed reference camera centers admit
ten full-track witnesses, so coarse tolerance can also conceal a wrong geometry.
On walking-XYZ, eight slots lack a complete early track. At 4 pixels, reference
and DAGE each admit three early witnesses and two full-track witnesses; the yaw
and reversed-center controls admit no full-track witness. These are conditional
development results, not calibrated failure detection or a method ranking.

All solver outcomes, numerical margins, witness coordinates, and observation
indices are retained in `tum-point-support-static-v1.json` and
`tum-point-support-xyz-v1.json`. Synthetic tests cover exact stationary cameras
with indeterminate depth, later contradictory observations, and missing early
evidence. Positive witnesses were independently reprojected into the original
camera coordinates and checked against the stated boxes and positive depth.

The same unmodified probe was also run on the retained 90-frame TUM
walking-static clip (`tum-anchor-static-v1.json`). Reference cameras support nine
fits and 302 later measurements; three fits have nonpositive depth. DAGE supports
12 fits and 392 later measurements. On the 302 common observations, reference,
DAGE, and the later five-degree reference-yaw control have medians of 10.683,
1.191, and 59.119 pixels. These are not evidence that DAGE is more accurate than
the reference. All 90 source-image hashes were verified; no image regions were
certified static. The source clip's name is not a zero-motion ground-truth label.

`analyze_anchor_conditioning.py` records the fitting-ray normal-matrix eigenvalues
and angles for every available early track. In walking-static, the median maximum
angle from each first ray over the 11 fitting frames is 0.451 degrees using
reference cameras, versus 3.128 degrees in walking-XYZ. Full-clip maximum reference
center displacement from the first frame is 0.0791 m and 0.5771 m, respectively.
Model translations are recorded in unscaled model units, not compared to meters.
These descriptive measurements are consistent with weaker point-depth constraints
in the stationary-camera case, but do not isolate conditioning from image-track
error, independently moving features, calibration, or reference timing. No
observability threshold or DNF decision is inferred from them. Per-slot geometry
is retained in `tum-anchor-static-conditioning-v1.json` and
`tum-anchor-xyz-conditioning-v1.json` in the dense results directory.

`probe_tum_anchors.py` applies the fixed 4-by-3 first-frame feature grid and
11-frame point fit to the retained 90-frame walking-XYZ run. All 90 source image
hashes are verified. Dataset calibration is used for every camera variant;
predicted DAGE calibration is not tested. All image regions are admitted without
semantic masks. Zero-valued admission masks in this experiment are **not static
background annotations**; the shared kernel's `insufficient_early_background_track`
status means insufficient early image tracking here. Features and image tracks
are selected without reference poses or projection errors.

The reference cameras, DAGE, and a control with a five-degree camera-local yaw
added only after the fitting frames all support the same 130 later observations
out of 948 possible. Their median placement errors are respectively 2.995, 1.851,
and 49.177 pixels. Each has three admissible point fits, eight insufficient early
tracks, and one nonpositive-depth fit. Reversing all reference camera centers
leaves one admissible fit with 15 later observations and a 742.245-pixel median;
this is a separate coverage set and is not directly comparable as a ranking.

This is an important negative result: the score responds strongly to the chosen
rotation corruption, but the estimated cameras score lower than the reference
cameras on the unmodified video. It therefore cannot yet serve as a trustworthy
camera-accuracy ranking. Image-track drift, independently moving features,
reference timestamp association, calibration, and the early 3D fit can contribute
to the discrepancy. This experiment does not isolate their contributions. It is
one development clip, not held-out evaluation, a functional-DNF label, or proof
that DAGE is more accurate than the reference. All tracks, failed fits, projections,
and common-observation indices are retained in
`paper/results/dense/tum-anchor-xyz-v1.json`.

The follow-up `probe_anchor_prefix.py` holds image tracks fixed and fits the
first 5, 11, 21, or 31 frames, evaluating only frames 31 onward. It retains all
eight camera/prefix cases, not the best prefix. Only 49 of 708 potential later
observations (two grid slots) are available in every case. On this common subset,
reference/DAGE medians are 2.398/3.067, 2.681/3.055, 2.626/2.804, and 2.728/3.236
pixels for the four prefix lengths. These values do not show general accuracy
or select a preferred fitting window.

Crucially, the original 11-frame fit on *all* its 70 available observations from
frame 31 onward gives 3.337/2.577 pixels, with the opposite reference/DAGE order.
Restricting those same fits to the 49-observation intersection alone reverses
the ordering. This is a selection effect, not evidence that a longer prefix
improves the evaluator. The result makes coverage accounting essential: an
intersection can make comparisons matched while still changing what scene
content is represented. Full per-case outcomes and observation membership are
retained in `paper/results/dense/tum-anchor-prefix-v1.json`. Staticness and image
track correctness remain unverified; the first-frame inspection alone cannot
establish either across the video.

`run_dense_dage.py` bypasses the prediction cache and rescans TUM timestamp files.
It records ordered timestamps, hashes of each source file and decoded frame,
checkpoint contents, Python source files, runtime versions, settings, and output
hashes. It refuses to overwrite an attempt. Runs use the existing DAGE environment
on Anima with `/home/kmatzen/git/geobench/src` on `PYTHONPATH`; the environment's
actual source hashes, not the local checkout's revision, describe executed code.
Artifacts live under `/data3/kmatzen/plumbline/research/dense-dnf/`.

The first walking-XYZ attempt (`dage-walking-xyz-v1`) failed before inference
because the runner passed a directory to a loader expecting `model.pt`. This is
a development harness error, not a measured tracker failure. The corrected
attempt is `dage-walking-xyz-v2`. Preserve both records. An attempt directory or
`running` manifest alone does not prove a process is live; check the process or
active execution handle. A `complete` result is still not a human usability label.

After copying a completed run's `manifest.json`, `prediction.npz`, and
`reference.npz` locally, use `analyze_dense_run.py RUN NEW_OUTPUT_DIRECTORY`.
It verifies prediction-file and reference-array hashes, pose counts, and ordered
timestamps before rendering reference-assisted diagnostics. It does not recheck
the remote source images or assign a functional-DNF label. Its summary reports
the sampled duration and interval; a 90-frame sample is not an entire sequence.

The corrected walking-XYZ development run completed: 90 frames spanning 8.980 s,
with a median interval of 0.1000 s. Inference including model loading took
173.96 s, with 6,539,740,672 bytes peak allocated CUDA memory. Prediction-file
and reference-array hashes passed local verification. A separate check on Anima
rehashed all 90 source-image files with zero mismatches. Its mechanical trajectory
gate passes. Reference-assisted diagnostics report mean rotation error 2.388
degrees and mean position error 0.04452 times the reference trajectory radius
after Sim(3) alignment. These are descriptive observations from one development
clip, not validation of a DNF evaluator. Functional usability remains unreviewed.

The complete manifest and pose arrays are retained in
`results/dense/dage-walking-xyz-v2/`, with derived diagnostics in
`results/dense/dage-walking-xyz-v2-analysis/`. Source images remain on Anima;
their hashes are in the manifest. A walking-static development run has been
launched separately to test the stationary-camera/moving-foreground case; no
result or usability label is implied by that launch. That control subsequently
completed and is retained in `results/dense/dage-walking-static-v1/` with derived
diagnostics beside it. Its 90 frames span 9.016 s. The prediction and reference
hashes verify and the mechanical gate passes; functional usability is unreviewed.

This control exposes an alignment caveat: the reference camera-position RMS
radius is only 0.023385 m. Mean orientation difference in the shared
first-camera coordinate frame is 0.434 degrees, but mean orientation error after
position-based Sim(3) alignment is 9.828 degrees. These are different gauges, not
interchangeable scores. A large aligned orientation value alone is not a DNF
label. Review packets must state the alignment and include evidence beyond the
position-aligned orientation plot. The sequence name does not imply exactly
zero measured reference motion.

For exactly zero translational extent, the review code now marks the Sim(3)
diagnostic unavailable instead of collapsing a moving prediction to zero error
or crashing on two stationary paths. Pose validity remains separate. Such pairs
cannot receive a preference inferred from missing diagnostic images. This fixes
an exact degeneracy; it does not solve alignment sensitivity near zero motion.

New review artifacts use `gate_policy: structural_only_v2`. Frozen predictions
no longer receive an automatic invalid-output label: they can be valid on a
stationary-camera task, and poor motion recovery belongs to the functional
assessment. A frozen candidate may still lack a Sim(3) diagnostic. Invalid
reference poses block pair construction instead of counting against either
candidate. Historical artifacts and votes retain their original policy; do not
pool old automatic frozen-track rejections with the new structural-only gate.
The active human pilot is unchanged. The manuscript source reflects this policy;
the previously built PDF has not yet been regenerated.

`build_source_preview.py MANIFEST DATA_ROOT NEW_OUTPUT` verifies the source-image
hashes before staging frames and encoding a source-only MP4. Playback uses the
median sampling interval; exact timestamps are retained and remain authoritative
for temporal analysis. The walking-XYZ preview on Anima is
`/data3/kmatzen/plumbline/research/dense-dnf/walking-xyz-source-v1/`: ffprobe
confirmed 90 frames at 640x480, with a duration of 9.001708 s. Maximum frame-time
displacement from recorded capture timestamps is 0.084282 s. Its SHA-256 is
`4007ff3039d3dc51c6854b5d5edea6f3ba91753e32602f9a447ba663167b148d`.
The preview and retained PNG frames are for context, not evidence of tracker
quality by themselves. A timing-faithful candidate overlay is still needed for
the functional-DNF annotation study.

## Human diagnostic-review pilot

### Diagnostic-only VLM development pilot

`run_vlm_dnf_pilot.py` is a single-image, reference-assisted development probe,
not the proposed full-evidence DNF metric. It uses
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
with snapshot revision `66285546d2b821cf421d4f5eb2576359d3770cd3` selected for
the initial trial. The download is stored on Anima's large data disk. The
existing `/home/kmatzen/git/geobench/.venv` supports the model; the separate
DAGE environment does not include Transformers and is left unchanged.

The script takes a pinned local checkpoint, one diagnostic PNG, and a new output
JSON path. It records prompt/image/runner/checkpoint hashes, software versions,
hardware, generation settings, raw output, parse status, and any runtime failure.
It uses float16 eager attention, greedy generation, a 1,280-visual-token pixel
budget, and at most 256 generated tokens. No prediction is silently repaired or
retried. Invalid JSON is an evaluator-format failure, not a tracker DNF.

The prompt explains that source video and reprojection evidence are missing and
allows abstention. It explicitly warns about position-based alignment on nearly
stationary tracks. This probe tests whether the pipeline runs and what evidence
the model claims to use; it cannot measure correctness without independent labels.
Any diagnostic-only judgments must remain distinct from full-evidence judgments.
The six response-parser tests use synthetic strings, not model or human outcomes.

Three actual pilot calls are retained in `results/vlm-dnf/`: walking-XYZ,
walking-static, and a reference-against-itself control. All calls completed on
the GTX 1080 Ti, taking 7.2–7.8 s for generation and approximately 10.32 GB peak
allocated CUDA memory. All returned fenced JSON and therefore failed the strict
response parser. Their raw text claimed `unusable`; none is an accepted DNF
label. The image and executed-runner hashes for both real cases were verified
against the committed evidence and script.

More importantly, the identical-trajectory control received a reason claiming
large path deviations and high errors. That contradicts the construction:
candidate and reference are the same stored trajectory, with maximum normalized
position residual 7.97e-16 and maximum reported rotation residual 0.01876 degrees
from floating-point rotation matrices. The control image is retained alongside
the response. It is not a real tracker output or a human usability label.
These three selected probes do not estimate general accuracy, but the control
failure prevents this pipeline from serving as the proposed DNF gate. Audit
image preprocessing and plot readability, and compare evidence presentations
and model choices before any deployment or superiority claim. Do not silently
repair the recorded responses or tune on held-out labels.

A post-hoc replay of the original processor on Anima yielded distinct finite
pixel tensors for the three images. Each input contains 1,118 image tokens and
1,374 total tokens, with image grid `[1, 52, 86]`. The hashes and scope are retained
in `results/vlm-dnf/preprocessing-audit.json`. This rules out absent image tokens
or identical preprocessed tensors in the replay, not numerical problems inside
the model or failures to interpret the plots. It is not an original-run trace.

### Pairwise review interface

`build_review_study.py` creates 150 native-input comparisons and 15 hidden
repeated presentations, with method names removed and candidate sides randomized.
The study asks reviewers to compare reference-based trajectory and rotation-error
plots. It does not ask about rendered-video quality. The present plots clip large
errors (with clipping counts shown), so reviewers may lack enough information to
rank some failures. An abstention is a valid response.

The active local packet is `tmp/review-study-v1`, served on port 8766. Keep this
packet unchanged while reviewers work. To create a separate packet:

```sh
uv run python paper/build_review_study.py --help
```

Serve only the packet's `public/` directory; `private/study.json` contains method
identities and repeat mappings. The browser stores responses locally and exports
a JSON file through **Download responses**. Partial sessions are supported. Use
one browser profile per reviewer: changing the reviewer code in a populated
session does not clear its answers. Retain only the latest cumulative export
from each reviewer, not several exports of the same person's responses.

Analyze received exports with:

```sh
uv run python paper/analyze_review_study.py \
  tmp/review-study-v1/private/study.json reviewer-one.json reviewer-two.json \
  --output tmp/review-analysis-v1.json
```

The analyzer rejects duplicate reviewer codes, duplicate responses, and mismatched
studies. Anonymous codes cannot prove that reviewers are distinct people; that
must be checked during recruitment. Only original presentations count toward
consensus. Repeats measure within-reviewer consistency, and repeat-only responses
are not substituted for missing originals. Consensus requires at least two
non-abstaining reviewers and a strict majority. Ties, both-poor judgments, and
abstentions remain distinct. Inter-reviewer exact agreement reports its overlap
and usable-pair denominators, excluding repeats and abstentions; it is descriptive,
not a chance-corrected reliability estimate.

No human outcomes have been analyzed yet. Project-member feedback is pilot data,
not independent external validation. A small partial session cannot establish
generalization, evaluator superiority, or a publication-ready reliability claim.

For later automated-judge calibration, `calibrate_judge_votes` reports decisive
accuracy and coverage separately. Missing judge responses reduce coverage.
Its percentile intervals resample complete tasks, retaining correlations among
method comparisons from the same clip. Intervals are unavailable without task
IDs or with fewer than two tasks. Accuracy intervals condition on bootstrap
samples with at least one decisive judge response; the output reports how many
draws qualify. High conditional accuracy at low coverage is not evidence of a
generally accurate judge. Two-task intervals are only a calculation, not enough
evidence for a stable population estimate. Input labels must be aggregated
explicitly before calibration, and shared task or candidate-side metadata must
match. These checks have unit-test coverage but no human-calibration result yet.

Build from the repository root:

```sh
uv run python paper/make_figures.py
uv run python paper/run_experiments.py
uv run python paper/run_sensitivity.py
uv run python paper/check_claims.py
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The release PDF is built separately into `output/pdf/` so generated files do not
need to be committed.

## Direct reference-free baseline: ORE

### First DAVIS development output (not a quality result)

The official 480p train/validation archive was staged on Anima with SHA-256
`e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`:
90 sequences, 6,208 frames. The first listed training sequence, `bear`, was
selected for a 60-frame development prefix, without selecting on output quality.
Validation sequences have not been used for development inference.

`results/davis/davis-dage-bear-v1/` retains the prediction and execution manifest.
This run completed in 75.30 seconds including model loading, with peak allocated
GPU memory of 7,043,777,536 bytes. It is one run, not a runtime benchmark.
Prediction bytes and all 60 source-frame hashes were verified locally.

`build_reference_free_review.py` renders source frames, three coordinate views
of the estimated trajectory, camera +Z direction glyphs, and between-frame
translation/rotation. There are no reference poses, alignment errors, or time
units. The plot is a diagnostic, not a validity metric. Its record is retained
as `review.json`; the source-containing image is local at
`tmp/davis-bear-review-v2/review.png`, not redistributed in the repository.
The recorded verdict remains **unreviewed**. This sheet lacks synchronized
dense video and independent reprojection evidence; it cannot establish DNF.

### DAVIS image consistency with estimated calibration

The pose-only DAGE adapter previously discarded the native intrinsics, and its
docstring incorrectly described them as pixel-space. Inspection of vendored
`DAGE.get_intrinsics` and `utils3d.image_uv` establishes width/height-normalized
intrinsics with half-pixel centers. The research runner now retains that array;
the production adapter's behavior is unchanged and its documentation is fixed.

`results/davis/davis-dage-bear-v2/` retains a fresh run with native calibration.
The 60 poses are elementwise identical to v1. Runtime was 72.08 seconds including
loading, with the same peak allocation; this is not a controlled runtime study.
The estimated horizontal focal length ranges from 1.38521 to 1.42794 in units of
image width. These values are model outputs, not measured camera calibration.

`analyze_davis_image_evidence.py` selects mutual SIFT matches with a 0.75 ratio
test without using poses or calibration. It then computes Sampson residuals
under the candidate poses and estimated calibration. Pair lags are 1, 10, and 30
supplied frames, with starts every 5 frames. The retained JSON contains pixel
coordinates and individual residuals: 28 pairs and 22,472 matches, all with
finite residuals. Median pair-median residuals by lag are respectively
0.000204, 0.000777, and 0.001353 in normalized camera coordinates. These are
descriptive development measurements, not a quality ranking or validation.
No motion mask or candidate-conditioned outlier removal was applied; matches
can lie on moving objects or be incorrect. Near-zero camera translation also
limits interpretation of epipolar residuals, regardless of numerical validity.
This baseline is not ORE and does not establish functional DNF.

### Controlled rotation sensitivity on the DAVIS development clip

`probe_davis_rotation.py` holds image matches and predicted calibration fixed,
then changes only the target camera's orientation for each pair. It evaluates
local X/Y/Z rotations at -15, -5, -1, 0, 1, 5, and 15 degrees. These are pairwise
interventions, not a single modified full trajectory. All 588 measurements
(28 pairs, 3 axes, 7 angles) are retained in
`results/davis/davis-dage-bear-v2/rotation-probe.json`.

The median across pair medians is 0.000659 at zero perturbation. At +15 degrees
it is 0.172628, 0.083314, and 0.030510 for local X, Y, and Z respectively.
Each of those +15-degree conditions increases the residual on all 28 pairs.
This is a development sensitivity check, not evidence of human preference
accuracy: the unperturbed estimate is not ground truth, and these changes do
not define functional-failure labels. Both signs and all magnitudes remain
available for inspection, including changes that might improve an estimate.

The local overlay `tmp/davis-rotation-probe-v2/overlay.png` shows frames 0 and 30,
using six matches selected at evenly spaced indices without residual-based
selection. Target-image lines are predicted epipolar constraints; circles are
observed feature matches. The lower row changes local Y rotation by +15 degrees.
Titles disclose this intervention, so this image is explanatory evidence and
must not be used as a blinded judge test or training example. Static correct
matches should meet their lines, but satisfying this constraint does not prove
correct pose. Moving objects and false matches remain possible confounders.

### VLM geometric-reading controls: failed positive control

`build_geometry_controls.py` constructs two six-panel images from the fixed
DAVIS feature set. It retains each line's direction, but moves the line to pass
through its marked point (all six distances zero), or displaces it by 50 pixels
(all six distances 50). The images disclose neither the expected answer nor
camera quality. These are drawing-reading controls, not physical camera outputs.
Both images were visually inspected and all 12 distances numerically checked.

Three actual Qwen2.5-VL-3B attempts are retained in `results/geometry-controls/`,
along with the executed runner versions and control hashes. Source-containing
control PNGs remain local in `tmp/geometry-controls-v1/`. At the original 1,280
visual-token budget the matching control failed with CUDA out-of-memory; the
second control was not attempted at that setting. At 768 visual tokens both
calls completed. Both raw answers said `inconsistent`: the matching control
incorrectly claimed panels 4--6 were not aligned. The displaced control's raw
category matched its construction. Both answers used Markdown fences and were
rejected by the unchanged strict JSON parser. There are no valid structured
verdicts and no usability labels from these calls.

This failed positive control is substantive, not just a formatting issue. It
does not establish failure rates for other models or inputs. It does rule out
treating this pilot configuration as a reliable reader of geometric evidence.
The next method revision should compute geometric relations deterministically,
and test any VLM role separately: identifying potentially moving scene regions,
recognizing missing evidence, and explaining measured contradictions. Semantic
judgments still need independent annotation; a VLM must not override measured
point-line distances or turn low residuals into proof of camera usability.

### Expanded DAVIS development pilot and annotation-assisted regions

The next two entries in the recorded training list, `bmx-bumps` and `boat`, were
run with the same DAGE checkpoint, first-60-frame selection, resolution, and
precision as `bear`. Both completed. Their prediction and execution manifests
are retained in `results/davis/davis-dage-bmx-bumps-v1/` and
`results/davis/davis-dage-boat-v1/`. All 120 additional source-frame hashes and
both prediction hashes were verified before image analysis. Neither run has
a functional-usability label. The validation split remains unused for inference.

Each clip contributes the same 28 frame pairs. The BMX and boat runs yield
11,626 and 18,258 mutual SIFT matches, respectively. This is still a three-clip,
one-method development sample, not a comparative benchmark.

`stage_davis_masks.py` stages official instance masks separately from model
inputs and checks that archive image bytes match the run. `analyze_davis_regions.py`
partitions the existing correspondences without reselecting them. It follows
the official DAVIS encoding (0 background, positive non-255 object IDs, 255
void). Background is not verified static, and object membership does not imply
motion. This analysis uses human dataset annotations and must not be reported
as an automatic, annotation-free evaluator.

| Clip | Both-background matches | Same-object matches | Background median of pair medians | Same-object median of pair medians |
| --- | ---: | ---: | ---: | ---: |
| bear | 20,746 | 1,717 | 0.000643 | 0.002344 |
| bmx-bumps | 11,034 | 574 | 0.001391 | 0.012484 |
| boat | 15,699 | 2,500 | 0.000297 | 0.000248 |

Residuals are in normalized camera coordinates under predicted calibration.
Cross-region counts are 9, 18, and 59; these are retained separately. None of
these clips has void/outside matches under nearest-pixel sampling. These
descriptive summaries neither validate the camera estimates nor provide DNF
thresholds. In particular, the object-region residual is lower in `boat`, so
discarding annotated objects is not uniformly supported even in this sample.

Mask manifests and region summaries are retained with each run. Masks remain
local and on Anima, not redistributed. An early boat mask staging linked to its
then-running manifest; it is preserved remotely but excluded from analysis.
The used boat `masks-v2` staging was made after completion, and the staging
script now rejects unfinished runs explicitly.

### Second-method development comparison: DA3

DA3-LARGE-1.1 was run on the exact same 60 input frames for `bear`, `bmx-bumps`,
and `boat`, using pinned snapshot `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`.
An initial full-attention `bear` attempt failed with CUDA out-of-memory and is
retained under `results/davis/davis-da3-bear-v1/`. A research-only query-chunk
wrapper then allowed all three runs to complete without dropping frames or
keys/values. It splits queries into blocks of 256, preserves full attention
context, rejects causal/dropout modes, and restores the original function after
inference. Small CPU tests match full attention within 1e-6, including masked
inputs; a successful unchunked 60-frame output was not available for direct
end-to-end numerical comparison. The wrapper and its hash are retained.

Successful outputs are `davis-da3-bear-v2`, `davis-da3-bmx-bumps-v1`, and
`davis-da3-boat-v1` under `results/davis/`. Inputs were processed to 140x252.
DA3 pixel intrinsics were converted by reversing its source-verified scale-only
resize policy, then encoding input pixel centers as normalized UV coordinates.
All source-frame and prediction hashes were checked. Runtime including loading
was 9.43, 9.47, and 9.41 seconds respectively. These development settings and
precision/memory implementations do not constitute a controlled speed benchmark.

`compare_davis_geometry.py` reuses the exact previously selected correspondences
and checks equality of frame records and decoded input hashes. It reports
symmetric point-to-epipolar-line distances in **input pixels**, not the normalized
Sampson units used in the earlier within-method analysis. This avoids directly
comparing camera-coordinate residuals under different focal scales. A 2x2 swap
of poses and calibration separates their influence diagnostically; off-diagonal
conditions are interventions, not outputs of either method.

| Clip | DAGE pose / DAGE K | DAGE pose / DA3 K | DA3 pose / DAGE K | DA3 pose / DA3 K |
| --- | ---: | ---: | ---: | ---: |
| bear | 1.1139 | 0.6823 | 1.6766 | 1.3317 |
| bmx-bumps | 3.8438 | 3.9757 | 4.6681 | 4.6964 |
| boat | 0.7875 | 0.4192 | 1.2049 | 0.9213 |

Entries are median pair-median distances over 28 pairs per clip, retaining all
matches including moving-object and erroneous matches. All conditions have
finite measurements on every pair. Calibration alone changes these diagnostics
substantially on two clips. Neither these values nor their ordering establish
human preference, camera accuracy, or functional DNF. Three sequences and two
models remain development evidence, not a validated comparative benchmark.

### Translation-sign and depth diagnostics

`probe_davis_cheirality.py` tests a known epipolar ambiguity on the same fixed
matches: reversing every camera center leaves the essential matrix unchanged up
to sign. It therefore leaves the absolute epipolar residual unchanged, even
though reconstructed point depths change sign. Positive depth (cheirality) is a
classical constraint, not our contribution; see
[Zisserman's two-view geometry lecture](https://www.robots.ox.ac.uk/~az/lectures/cv/lecture5.pdf).

All 168 native pairs across six outputs had unchanged residuals under the sign
intervention. The baseline-normalized least-squares two-ray diagnostic reports
every match, including numerical degeneracy and mixed-sign depths. Synthetic
checks cover known depths, rotated cameras, view swaps, positive scale, world
coordinate changes, and parallel rays with fabricated translation.

| Output | Matches | Positive depth in both cameras | Negative depth in both | Median pair-median ray angle |
| --- | ---: | ---: | ---: | ---: |
| DAGE bear | 22472 | 83.38% | 16.57% | 0.1318 degrees |
| DA3 bear | 22472 | 85.19% | 14.70% | 0.1137 degrees |
| DAGE bmx-bumps | 11626 | 68.31% | 31.44% | 0.2039 degrees |
| DA3 bmx-bumps | 11626 | 59.33% | 40.46% | 0.2939 degrees |
| DAGE boat | 18258 | 78.05% | 21.91% | 0.5013 degrees |
| DA3 boat | 18258 | 49.56% | 50.35% | 0.3865 degrees |

Percentages pool correspondences, not independent samples; repeated features and
frames occur across pairs. Remaining matches have mixed-sign or zero depths.
No observed match was numerically rank deficient, but the small angles warn that
depth signs can be sensitive to calibration and measurement errors. This is not
an empirically calibrated observability gate. Moving objects and mismatches are
retained, and intrinsics are predicted. These fractions are not tracking accuracy
or DNF rates. Per-pair counts and angle quantiles are retained in each run's
`cheirality-probe.json`. No acceptance threshold has been fitted.

### Fixed-grid anchor coverage probe

`probe_davis_anchors.py` replaces the manual wall choice with 12 fixed first-frame
grid cells. Each contributes its strongest corner in an eroded annotated-background
mask, or an explicit empty slot. Both methods receive exactly the same image
observations. Fits use frames 0–10 only. Later observations cannot change the
fitted point (covered by a regression test).

Chained Lucas–Kanade tracks stop permanently after a failed forward/backward
check (1 pixel). A separate first-frame-to-current-frame LK pass provides a
cross-check, requiring agreement within 3 pixels and annotated background.
These are fixed development choices, not calibrated correctness thresholds;
the two tracking paths can share errors. The stricter comparison also requires
cross-check agreement throughout the fitting frames. No score-based anchor
selection or restart occurs. All slots, observations, failed fits, projections,
and comparison subsets are retained in `paper/results/davis/grid-anchors-v1/`.

Each clip has 12 slots × 49 later frames = 588 possible test observations.
The following are pooled medians over **common available** background observations;
the stricter row is a different subset and must not be compared as an improvement
in camera quality.

| Clip | Common observations | Common anchors | DAGE pixels | DA3 pixels |
| --- | ---: | ---: | ---: | ---: |
| bear, chained | 494 / 588 | 11 / 12 | 4.936 | 2.323 |
| bear, cross-checked | 167 / 588 | 9 / 12 | 4.344 | 2.018 |
| bmx-bumps, chained | 3 / 588 | 1 / 12 | 10.516 | 17.747 |
| bmx-bumps, cross-checked | 1 / 588 | 1 / 12 | 10.516 | 17.747 |
| boat, chained | 127 / 588 | 3 / 12 | 14.212 | 10.609 |
| boat, cross-checked | 68 / 588 | 3 / 12 | 8.495 | 8.059 |

Fit accounting also matters. Bear has 11 fits and one insufficient image track
for each method. BMX has nine insufficient early tracks for each; DAGE has two
fits and one nonpositive-depth fit, while DA3 has three fits. All 12 BMX image
tracks are lost by the final frame. Boat has ten DAGE fits and five DA3 fits;
the remaining two and seven respectively have nonpositive fitted depth. Those
unavailable cases are retained, not assigned zero error or omitted from coverage.

This probe does not establish method rankings or DNF. It shows that this tracker
and long-window protocol supply almost no comparable evidence on BMX, and that
fit availability can differ substantially between camera estimates. Background
may move, tracks may drift, and predicted calibration and poor parallax confound
point fits. Better image tracks and separately reported temporal-window coverage
are required before a useful evaluation can be claimed.

### CoTracker image-evidence comparison

The next development probe uses [CoTracker3](https://github.com/facebookresearch/co-tracker)
at repository commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, with the
`facebook/cotracker3` `scaled_offline.pth` checkpoint at revision
`bf55ea50d4390e1820a267f131cd6587240fb2c5`. Full checkpoint and source-file hashes
are recorded for every run. Upstream code and weights stay outside this repository;
the upstream license is CC BY-NC 4.0. This is a research dependency, not a claim
that our repository's license applies to it.

`run_cotracker_anchors.py` sends the same 12 first-frame query points as the LK
probe. It preserves upstream support-grid, visibility, and query-correction
behavior. Every clip is processed twice: its first 11 frames alone, then all 60.
Only prefix-only observations enter the early 3D point fit. Full-video observations
are used for the later comparison. This separation matters because offline image
tracking can use later images to change earlier coordinates. Maximum early-coordinate
differences between prefix and full processing were 1.642 pixels (bear), 12.923
(BMX), and 8.483 (boat). These are tracker-consistency differences, not errors
against known point tracks.

`analyze_cotracker_anchors.py` verifies input hashes and evaluates both sets of
camera estimates on the same observations. All visibility failures, lost support,
and nonpositive-depth fits remain in the records.

| Clip | Common later observations / 588 | Anchors / 12 | DAGE median pixels | DA3 median pixels |
| --- | ---: | ---: | ---: | ---: |
| bear | 472 | 11 | 5.104 | 2.398 |
| bmx-bumps | 98 | 3 | 25.973 | 39.806 |
| boat | 174 | 4 | 17.795 | 12.472 |

BMX coverage increases from 3 to 98 common observations, but remains only 16.7%
of planned observations. Neither this increase nor the higher measured distances
establishes tracker accuracy: the available subset changed. The model's visibility
output is not independent verification. Background can move, and both tracking
systems can make related mistakes. There are only five later BMX observations with
both usable LK and CoTracker coordinates, so agreement with LK supplies little
validation on the difficult clip.

Inference manifests and raw prefix/full arrays are under
`paper/results/davis/cotracker-{bear,bmx-bumps,boat}-v1/`. Their `anchor-analysis.json`
files retain observations and per-method point fits. Tests explicitly verify that
full-video early coordinates cannot enter the fitting observations and that empty
query slots, invisible points, and off-image predictions do not become zero errors.

### DAVIS author-review pilot (withdrawn record)

**Withdrawn from further review:** the project author reported that the examples
did not make clear what to look for. This is qualitative evidence of an interface
failure, not a usable camera-quality label. Retain the original packet and its
mapping for audit, but do not ask for more ratings in that format.

The replacement development preview, `build_anchor_preview.py`, shows a cyan dot
from image tracking and an orange ring from projection of a fixed fitted 3D point.
It fits only frames 0–10 and projects into frames 11–59 without refitting. A zoom
follows the observed dot. The initial strongest-background-corner trial selected
a leaf and lost tracking; that attempt is retained. A second, explicitly manual
first-frame wall ROI `[200, 20, 400, 120]` produces an interpretable prototype.
This manual choice was made after inspecting the first attempt and is development
work, not an independent or automatically selected benchmark sample.

The local preview is `tmp/anchor-preview-bear-wall-v3/preview.mp4`. Metadata and
all per-frame observations/projections are archived under
`paper/results/davis/anchor-preview-development/`. Cyan tracks can drift, background
annotations do not certify static surfaces, and triangulation uses predicted
calibration. This is a joint pose/calibration/point-fit consistency check, not a
test of pose alone. Synthetic tests verify known-point recovery, held-out camera
perturbation sensitivity, and unavailable stationary/behind-camera cases. A
training example with explicitly simulated attached and drifting overlays is now
available from `build_anchor_lesson.py` at `tmp/anchor-lesson-v2/index.html`.
Both panels use the same image track; the right ring receives a known horizontal
offset, reaching 50 pixels. Neither panel is a method output or validation label.
The page explains image-track failure, moving surfaces, and calibration/point-fit
confounds. Multi-anchor coverage and independent track/static-region checks remain needed
before requesting a new review session.

The same manual first-frame wall ROI and fitting frames were also used with DA3.
All 60 image observations are identical between the two previews, with 49 test
frames after the 11 fit frames. Median test-frame marker distances are 5.618 pixels
for DAGE and 2.214 pixels for DA3. This one manually selected development anchor
does not rank the methods. It does illustrate why the pooled epipolar diagnostic
(which was lower for native DAGE on bear) and a particular held-out placement test
need not tell the same story. There is no validated usability threshold, and the
image track itself has not been independently verified. DA3's per-frame record is
`paper/results/davis/anchor-preview-development/da3-manual-wall-v1.json`.

The eight-presentation pilot contains six outputs (DAGE and DA3 on the three
development clips) and two repeated presentations. The public packet hides method
names and shows source video plus reference-free trajectory diagnostics. Video
playback is 10 displayed frames per second, not known capture timing. This packet
does not show a reconstructed scene or demonstrate successful static-content
placement, so insufficient evidence is an expected and useful response.

The available reviewer is a project author. Responses can test the instructions,
evidence sufficiency, and within-reviewer consistency; they cannot establish
independent validation, evaluator accuracy, or a generalizable method ranking.
No human responses have been incorporated yet.

Serve only the generated `public/` directory. Keep `private/study.json` and its
method mapping out of the browser. The version-one packet is retained locally at
`tmp/davis-usability-pilot-v1`; its study record is archived under
`paper/results/davis/author-review-pilot-v1/`. Do not rebuild over a live study.

Validate a downloaded partial or complete export with:

```sh
uv run python -m paper.analyze_davis_review \
  paper/results/davis/author-review-pilot-v1/study.json downloaded-responses.json
```

Keep the raw export unchanged. The analyzer reports presentation counts separately
from unique outputs, treats conflicting repeats as inconsistent, and distinguishes
unanswered presentations from explicit abstentions. It does not turn these answers
into ground-truth labels or calculate confidence intervals from repeated displays.

### ORE comparison prerequisites

Chen et al., [Object Reprojection Error (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Paper-Datasets_and_Benchmarks.pdf)
already evaluate trajectories without reference camera poses. This is a required
comparison, not a novelty claim for this project.

`src/plumbline/object_reprojection.py` is a development ORE-style kernel, not a
verified reproduction. It uses a fixed first-box center, a supplied finite depth
grid, normalized L1 distance outside boxes, and rejects depth hypotheses behind
any observed camera. It reports depth-boundary solutions and unavailable cases.
Synthetic tests check projection direction, depth recovery, coordinate/scale
changes, and input rejection; they do not validate real-data ranking quality.
The official NeurIPS supplement has now been audited and its one-sequence demo
executed. The development kernel is **not numerically equivalent** to that code.
The demo uses radial ray distance, normalization by width plus height, capped
penalties, and image-border rules that suppress the other coordinate's penalty.
It does not explicitly reject projections behind a camera. Its two brute-force
depth grids are `0.1:0.05:3.0` and `1.0:2.0:200.0` (upper limits excluded), with
no optimizer refinement. It chooses a minimum per static track, then averages
the four track scores for its supplied sample. Upstream pose loaders interpolate
and extend some sparse estimates; this behavior is retained, not silently changed.

`run_ore_demo.py` executes the external released code, with only a temporary
`np.int = int` compatibility alias. The original source files are unchanged.
The [official supplement](https://papers.neurips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Supplemental-Datasets_and_Benchmarks.zip)
contains sample poses for one sequence, not the full benchmark. All seven supplied
methods completed on its four eligible annotated-static tracks:

| Supplied sample poses | Demo mean ORE |
| --- | ---: |
| DROID-SLAM | 0.068128 |
| COLMAP | 0.053058 |
| ORB-SLAM2 | 0.093732 |
| ORB-SLAM3 | 0.099536 |
| TartanVO | 0.209516 |
| ParticleSfM | 0.098904 |
| Monodepth2 | 0.138053 |

These are evaluations of author-supplied poses, **not fresh model inference**,
not a reproduction of full-dataset published numbers, and not evidence that our
proposed evaluator is better. Logs, source hashes, per-track values, and dependency
versions are retained in `paper/results/ore-official-demo-v1/`. The archive and
source remain under `tmp/ore-upstream/`; no upstream code is vendored.

`audit_ore_semantics.py` compares the external kernel and our prototype on six
synthetic cases using the same numeric depth grid. The meaning of depth differs,
so this is a policy audit, not a fairness-adjusted accuracy comparison:

| Constructed case | Released kernel minimum | Prototype minimum |
| --- | ---: | ---: |
| Exact center-ray projection | 0 | 0 |
| Reversed translation | 0.075 | 0.150 |
| Off-axis ray (radial versus axial depth) | 0.008310 | 0 |
| Box touching image border | 0 | 0.066667 |
| Point behind later cameras | 0 | unavailable |
| Large projection displacement | 0.666667 | 5.000000 |

The behind-camera case has positive initial depth but later negative depth. The
border case triggers the released code's axis-suppression rule. These constructed
cases identify precise behaviors; they do not estimate their frequency in real
data or invalidate ORE generally. Retain the official metric unchanged as a
comparison and report any safety checks as separate variants. The audit record
is `paper/results/ore-official-demo-v1/semantics-audit.json`.

#### Positive-depth audit of the supplied real estimates

`audit_ore_depth_support.py` repeats both published demo search grids, verifies
each selected score against the retained official execution, and records camera
z for every annotated frame. It also computes a separate constrained minimum,
requiring positive z throughout that object's annotations. The official scores
remain unchanged. A constrained mean is unavailable if any of its four tracks
has no admissible candidate; the other tracks are not silently averaged instead.

| Supplied poses | Official mean | Positive-depth constrained mean | Selected fits with nonpositive z / 4 | No admissible grid depth / 4 |
| --- | ---: | ---: | ---: | ---: |
| DROID-SLAM | 0.068128 | 0.074437 | 2 | 0 |
| COLMAP | 0.053058 | 0.053058 | 0 | 0 |
| ORB-SLAM2 | 0.093732 | 0.093732 | 0 | 0 |
| ORB-SLAM3 | 0.099536 | unavailable | 1 | 1 |
| TartanVO | 0.209516 | unavailable | 4 | 4 |
| ParticleSfM | 0.098904 | unavailable | 4 | 4 |
| Monodepth2 | 0.138053 | 0.156302 | 1 | 0 |

Twelve of 28 method/object fits have at least one nonpositive projected depth at
the official optimum. Nine have no admissible depth in the released grids. For
each ray, camera depth is affine in the initial radial distance, `z_i(d)=a_i*d+b_i`.
Intersecting the strict inequalities `z_i(d)>0` with `d>0` shows that all nine also
have an empty continuous feasible interval: extending the depth grid cannot fix
those particular first-box-center rays under the supplied poses and calibration.
Nine official minima lie on an endpoint of one of the search grids; that separate
fact is retained without assuming the endpoint is the cause of every failure.

This is **one sequence with four annotated objects and seven supplied estimates**,
not 28 independent videos. Ray infeasibility is conditional on the anchor choice,
pose interpolation, calibration, and annotations. It does not establish that
every possible point in the first object box is infeasible, or certify camera
DNF. The result motivates testing geometric admissibility explicitly; it does not
validate a new metric or a method ranking. The complete search curves, negative
depth counts, and continuous intervals are retained in
`paper/results/ore-official-demo-v1/depth-support-audit.json`.

#### Attempt to relax the first-box-center assumption

The optional `--box-support` audit uses a linear program to search for any common
3D point in the observed box frusta with positive camera depth. A second mode
uses only the first box's image constraints, retaining depth constraints at all
annotated frames. Returned witnesses are checked by explicit reprojection.
Solver infeasibility is a numerical result, not an exact certificate. A common
point is also a stronger assumption than a static extended object whose visible
surface changes; failure would not by itself establish camera DNF.

On the supplied sample, input validation prevents either mode from running:

| Object track | Annotated frames | Reversed processed boxes |
| --- | ---: | ---: |
| 2 | 143 | 24 |
| 1 | 974 | 371 |
| 4 | 1309 | 40 |
| 6 | 961 | 429 |

None of these raw annotation rows has negative width or height. The released
preprocessing rescales coordinates, undistorts two opposite corners, and casts
to integers without restoring box ordering. Thus the observed reversals occur
in the processed representation; this audit does not establish whether the
underlying cause is calibration, distortion inversion, or annotation geometry.
The boxes are identical across methods, so these are four affected object tracks,
not 28 independent input failures. We neither sort the corners nor discard bad
frames. Both modes conservatively return `invalid_boxes` for the entire track.
Consequently this experiment provides no comparison of box-support feasibility
between methods and does not relax the earlier center-ray finding on real data.

The initial attempt stopped on input validation and is retained as
`box-support-failed-v1.json`. The completed diagnostic records every invalid
frame and box in `box-support-audit-v2.json`, in the same results directory.
Published ORE scores remain unchanged. Six synthetic tests cover positive
witnesses, conflicting boxes, opposing cameras, coordinate gauge changes, and
rejection of reversed corners; they do not validate real-data failure decisions.

`audit_ore_box_preprocessing.py` traces the rescaling, two-corner undistortion,
and integer conversion separately. Its output reproduces every reversed box
in the retained audit. Before undistortion, no rescaled box is reversed. After
undistortion, tracks 2, 1, 4, and 6 contain respectively 24, 372, 40, and 430
reversed boxes; integer truncation hides one reversal in each of tracks 1 and 6.

Forward-distorting the floating-point corrected corners should recover their
input pixel coordinates if inversion succeeds. Using the same calibration and
OpenCV distortion model, the median round-trip errors are respectively 99.40,
86.48, 0.0954, and 85.66 pixels. Of 6,774 corners, 3,995 have residuals above one
pixel. That cutoff describes inversion error, not camera quality. All residuals,
including extreme values, are retained in `box-preprocessing-v2.json` in the
same results directory. Sorting the corners would not resolve these inversion
residuals. This establishes a problem with this sample's processed evidence;
it does not identify which calibration or annotation assumption is responsible,
nor establish its frequency in the full benchmark. Official scores are not
changed. A valid replacement would require a justified distortion/calibration
policy and verification against source images, not an automatic box repair.

The source-track author review now uses an identity-first presentation
(`build_track_review.py`, manifest `results/dense/source-track-review-v2.json`).
The original three videos are unchanged and hash-checked. Each starting image
remains above its video. A reviewer must name the marked detail before entering
a tracking judgment; an unidentifiable or unanswered target exports as
`not_assessed`, not a tracking failure. Answers are downloaded locally with
the evidence hashes, not silently collected. This responds to author feedback
that the earlier examples did not make the inspection task clear. No new human
judgments have been collected, and the revised display has not been validated
for comprehensibility. These are source-track checks, not camera DNF labels.

DAVIS segmentation objects cannot be assumed static. Intrinsics are also absent
from our DAVIS inventory. Do not use arbitrary focal lengths or moving masks as
ORE ground truth. An annotated static-track subset and an explicit calibration
policy are needed. Predicted calibration would define an estimated-intrinsics
variant and must be reported separately from known-intrinsics evaluation.

### Reference-seeded pixel transfer audit

`audit_rgbd_seed_transfer.py` checks whether the existing RGB-D controls can
supply precise image-position labels. It fixes frame 1 as the source, samples
its nearest-pixel depth for each original grid track, and transfers the resulting
3D point using reference poses to frames 11 through 89. All twelve slots remain
in the accounting; an unavailable source is never replaced with a later one.
It compares transferred pixels with RGB-only LK observations, not with candidate
camera outputs. Depth and reference poses are scoring-only evidence here.

The retained `seed-transfer-v1.json` files in `results/static-rgbd-control-v1/`
and `results/office-rgbd-replication-v1/` contain all 948 planned observations
per sequence. The near control has 564 comparisons, 158 unavailable-source
observations, and 226 missing image tracks; median compared disagreement is
3.486 native pixels, with 81 comparisons within two pixels. The office control
has 441 comparisons, 237 unavailable-source observations, and 270 missing image
tracks; its median is 3.225 pixels, with 124 comparisons within two pixels.
All 790 and 711 available projections, respectively, were independently replayed
using explicit backprojection and camera rotation/translation equations.

These residuals do not identify an erroneous source: image tracking, depth,
timing, calibration, and scene effects can all contribute. Target visibility was
not independently established. Therefore this audit does not provide exact
pixel ground truth or camera-DNF labels. A two-pixel training target built by
silently transferring these depth samples would be unjustified; reference
uncertainty and visibility need separate treatment. This development audit is
not added to the manuscript's experiments.

The timestamp sensitivity arm is retained as `seed-transfer-interpolated-v1.json`
in both directories. It changes only reference-pose association: translation is
linear and rotation uses quaternion SLERP at RGB timestamps, with a maximum
50 ms bracket gap and no extrapolation. Depth samples, tracks, fixed source
frame, and query frames are unchanged. In the near sequence, frame 1 precedes
the available reference span, so all 948 planned observations are unavailable
under this policy; selecting a later source after seeing this outcome would be
a different experiment. In the office sequence, the same 441 comparisons remain
available. Median error changes from 3.225 to 3.309 pixels; 130 rather than 124
comparisons are within two pixels. The median paired error change is +0.0238
pixels (193 improve, 248 worsen). This timing change does not establish exact
image labels or remove the need to account for depth, visibility, and image-track
uncertainty. No failure threshold was selected from these diagnostics.

### Rendered correspondence control

`render_camera_control.py` renders a textured plane and an optional moving
foreground rectangle by ray/plane intersection. Camera poses, static-point
projections, and visibility are exact for this procedural scene. They remain
separate from RGB inputs. This is a controlled development fixture, not a
replacement for real-video or camera-level validation.

Run `uv run python -m paper.probe_rendered_tracking <new-output.json>` to reproduce
`results/rendered-tracking-control-v1.json`. The fixed protocol uses seeds 0–9,
24 frames, 12 static anchors, and clean/occluder paired conditions. LK tracks
survive only with a forward/backward error at most one pixel and never restart.
All 2,760 later observations per condition remain recorded. Without occlusion,
all are accepted and within two pixels of their analytic projections. With
occlusion, 1,011 are accepted: 979 correct, 25 occluded, and seven visible but
more than two pixels away. These counts are correlated within sequences, not
independent trials. Source hashes, RGB hashes, versions, and every observation
are retained. The scene is a single plane with procedural texture; no VLM or
estimated camera trajectory was evaluated. No DNF threshold is inferred.

The renderer tests check deterministic output, unchanged background pixels,
projection against a separate homogeneous-transform calculation, and analytic
visibility against the rasterized surface IDs. The next useful control is
camera consistency with observed versus exact tracks, retaining both clean
camera and perturbed camera conditions. This study is not added to the paper.

That paired control is now retained in
`results/rendered-consistency-control-v1.json.gz`, generated by
`probe_rendered_consistency.py`. It uses all 20 cases, all 12 slots, frames 0–11
for fitting and 12–23 for queries, a two-pixel per-coordinate tolerance, and
suffix yaw ramps of 0, 1, and 5 degrees. No thresholds were fitted. The measured
arm uses accepted LK coordinates only; tests confirm that reference coordinates,
visibility, and error values do not affect it. The exact-visible arm is explicitly
an oracle diagnostic, not an available reference-free method.

Without the occluder, both arms give 120 compatible slots for zero and one degree,
and 120 contradictions for five degrees. With the occluder and correct cameras,
measured tracks give 27 compatible slots, five contradictions, 86 insufficient
fitting tracks, and two missing-query tracks. Exact-visible tracks give 60
compatible and 60 insufficient fitting tracks, with no contradictions. At five
degrees the measured arm gives 11 contradictions, 21 compatible, and the same
88 unresolved slots; the exact-visible arm gives 60 contradictions and 60
insufficient fitting tracks. Texture seeds share geometry, so these slot totals
are not independent camera trials. No camera-level verdict is assigned.

This separates two limitations: tracking errors can contradict correct cameras,
and permissive geometric compatibility can hide a camera perturbation. The
current fitting rule also requires every prefix observation and loses evidence
when even one is missing. A missing-aware fit needs an explicit observability
check, not an implicit pass. These development results stay outside the paper.

`probe_available_fit_consistency.py` tests available-prefix observations without
changing the original comparison. It requires at least two observations and a
positive fitting witness, retains missing-query counts, and uses the same joint
query constraints. This is not a triangulation-precision or observability test.
`results/rendered-available-fit-control-v1.json.gz` retains every paired transition.
The unobstructed conditions are unchanged. For occluded measured tracks, the
numbers of compatible and contradictory slots are also unchanged: 79 slots
instead resolve to missing query evidence and nine to unestablished fitting
geometry. The no-restart tracker cannot recover observations after loss, so
relaxing the prefix requirement alone does not restore usable query evidence.
For exact visible tracks, all 120 slots now have sufficient fitting evidence:
all are compatible at zero/one degree, and all contradict the five-degree
perturbation. This oracle result motivates testing explicit track recovery with
independent identity checks; it does not justify treating restarted tracks as
correct or adding any new DNF claims to the manuscript.

### Direct matching after occlusion

`probe_rendered_recovery.py` uses direct first-to-target LK for every original
query independently of earlier track survival. Its fixed acceptance rule is
forward/backward error at most one pixel and 17×17 source-patch NCC at least
0.8, reused from the TAPvid baseline without tuning on this control. The matcher
receives only the two grayscale images and original query coordinates. Later
visibility and coordinates are read only after predictions are complete.
All rendered RGB hashes are checked against the preceding sequential experiment.

`results/rendered-recovery-control-v1.json` retains 2,760 observations per
condition. Without the occluder, 1,655 are accepted and correct; 1,105 previously
correct observations are rejected. With the occluder, 1,129 are accepted and
correct, including 200 observations rejected by the sequential tracker.
There are no accepted occluded or over-two-pixel observations in these controls.
This is not a claim of zero error on other scenes or independent observations.
The net increase over sequential accepted-correct observations is 150, not 200:
the new rule also rejects 50 previously correct occluded-condition observations.
It changes both matching and admission, so this comparison does not isolate
the effect of either change. Geometric sensitivity must also be checked before
calling it an improved camera evaluator. Results remain outside the manuscript.

Camera consistency is retained separately as
`results/rendered-recovery-consistency-v1.json.gz` (complete-prefix rule) and
`results/rendered-recovery-available-fit-v1.json.gz` (available-prefix rule), using
the unchanged comparison scripts and original 0/1/5-degree perturbations.
With available-prefix fitting, occluded direct matches yield 82 compatible
tracks and 38 missing-query tracks for correct cameras, with no contradictions.
At five degrees they yield 49 contradictions, 33 compatible, and 38 missing
queries. The sequential arm had five contradictions under correct cameras and
11 under the five-degree perturbation. However, unobstructed direct matches
yield only 61 contradictions at five degrees, compared with 120 for sequential
tracks; 50 remain compatible and nine lack query observations. Thus the stricter
matching check improves this occluded control while discarding useful evidence
in the clean one. Neither precision alone nor contradiction count alone is an
adequate evaluator objective. These are track-level counts with shared scene
geometry, not independent sequence-level failure rates or a novel method claim.

### Prior work and the proposed evidence-sensitivity check

The related-work section now includes Hu and Kantor's
[RSS 2017 introspective evaluation](https://www.roboticsproceedings.org/rss13/p33.pdf)
and [CoRL 2017 Bayesian parameter tuning](https://proceedings.mlr.press/v78/hu17a.html),
plus [OdoTest](https://doi.org/10.1145/3808128). The RSS method uses an approximate
posterior to estimate loss; its reliability depends on that approximation.
The CoRL work already combines introspection and Bayesian optimization for
odometry. OdoTest tests VIO through input transformations and metamorphic
relations. RSS method text and the official CoRL/ACM abstracts and metadata were
checked; OdoTest's full implementation has not been reproduced. None of these
ideas alone supports a novelty claim for this project.

A candidate extension is to measure the sensitivity of the evaluator's evidence
to camera changes, alongside its verdict. This is a proposed experiment, not an
implemented or validated metric. Keep images, correspondences, admission rules,
and calibration fixed while changing only the candidate trajectory. Report
signed perturbations and retained query coverage, not just a pooled rejection
fraction. Include identity and global-gauge controls, which should preserve
the geometric result. Rejection of a changed candidate does not validate the
original; a perturbation to an unknown real estimate might even correct an
error. Consequently, only reference-camera controls can supply known camera
errors at this stage, and their magnitude is not automatically a functional
failure label. A yaw-only check cannot establish sensitivity to translation,
other rotation axes, or unseen failure types. Any abstention rule needs to be
fixed on development data and evaluated separately, including correct-camera
false alarms and evidence corruption. This proposal must be compared with
simpler uncertainty and coverage measures, not merely with raw residuals.

The first bounded profile is now implemented in `probe_evidence_sensitivity.py`
and retained in `results/rendered-evidence-sensitivity-v1.json.gz`. It holds all
observed image coordinates fixed and tests identity, one global rigid gauge
change, signed five-degree rotations about each camera-local axis, and signed
quarter-baseline translations along each local axis. Changes ramp across frames
12–23; the earlier cameras remain unchanged except in the gauge control.
Translation amplitudes use the candidate's maximum displacement from its first
camera. These are stress-test scales, not validated functional-failure cutoffs.
The compact joint checker retains the same available-prefix feasibility rule
while omitting redundant per-query diagnostic solves.

All 560 cases and 6,720 slot evaluations are retained. All 480 per-slot rigid-gauge
comparisons preserve status, and all 960 identity/positive-yaw comparisons match
the preceding full-checker results. For unobstructed sequential tracks, both
signed x/y rotations contradict all 120 slots, but signed z rotations contradict
only 60 each. Neither tested z translation contradicts any slot. Negative x
translation contradicts all 120 while positive x translation contradicts none.
For occluded direct matches, signed x rotations contradict 62 slots each, signed
y rotations 49 each, and signed z rotations eight/seven; neither z translation
is rejected. These differences show why yaw sensitivity alone does not establish
general error sensitivity. A changed camera that remains compatible is not
certified correct, nor is rejection itself a validated DNF label. The scene
remains planar, amplitudes are fixed, and texture seeds are correlated. No
admission threshold or learned policy was selected from this profile.

### Real-estimate evidence-sensitivity pilot

`probe_real_evidence_sensitivity.py` applies the same finite-change profile to
the saved DAGE and DA3 estimates on both TUM RGB-D development clips. Each
clip's `evidence-sensitivity-v1.json.gz` retains 336 configurations and 4,032
slot evaluations: both methods, all three existing prefix lengths, all four
existing pixel tolerances, and all 14 changes. Source RGB hashes, decoded input
hashes, prediction manifests, archived runner hashes, and prediction bytes are
reverified. Evaluation uses RGB-derived tracks, estimated poses, and the common
published calibration; it does not use reference-pose or depth values. The
inference models saw the entire clip, so this is not causal prediction.

All 576 per-slot global rigid-gauge comparisons preserve status. As a descriptive
example, office DA3 at prefix 10 and tolerance four pixels has 12 compatible
tracks under the original estimate. Positive five-degree pitch causes all 12 to
contradict; positive five-degree yaw causes only five. At prefix 30, the nine
originally compatible tracks yield nine and seven contradictions respectively.
All settings, not only these examples, remain in the result. The four-pixel
setting was not selected as an admission threshold. On unknown estimated
trajectories a change can improve or worsen accuracy, so these are response
profiles, not detection rates for known failures. They establish neither a
method ranking nor functional DNF. Independent track validity and usability
validation remain necessary before using the profile as a decision rule.

Numerical audit after the pilot: all 6,980 recorded positive fitting/combined
witnesses were reprojected directly with the saved camera matrices and published
calibration. Every used camera depth was positive; the largest excess beyond
the recorded pixel-box tolerance was 3.40e-11 pixels (audit allowance 1e-5).
All recorded implementation hashes matched. This checks positive witnesses,
not solver infeasibility certificates, correspondence identity, or staticness.
Repeated witnesses across configurations are not independent observations.
The broad local command `uv run pytest -q -m 'not weights'` completed with
1,936 passed, one skipped (unstaged KITTI MoGe-eval bundle), and six weight-backed
tests deselected. These are software checks, not publication-readiness evidence.

### One actual-camera review example

`build_letter_camera_review.py` uses the previously shown printed-e detail
(slot 6) with saved estimated cameras. It fits one point per candidate from
frames 1–10 using common published calibration, then projects each fixed point
at frames 1, 45, and 89. There are no deliberately added camera errors and no
reference-pose/depth inputs. The display shows source crops above opaque A/B
rows, and asks only whether each cross stays on the same part of the letter.
The crops follow the fallible RGB track; this is a comprehension example, not a
randomly sampled or independent validation study. The local inspected image is
`tmp/letter-camera-review-v3/review.png`; its hash, mappings, fitting observations,
and all 180 projections are retained in
`results/static-rgbd-control-v1/letter-camera-review-v1.json`.
Projection uses the full inverse transform because saved rotation matrices are
not perfectly orthonormal. All 180 projections match a separate homogeneous
inverse calculation within 1e-8 pixels. No user answer has yet been recorded.

### Preparing disjoint learned-judge test inputs

`build_tapvid_external_corpus.py` freezes the same source/candidate panels,
prompt, lags, and label rule on all twenty TAP-Vid DAVIS videos outside the
ten-video learned-judge development partition. It opens no development-video
arrays or calibration export. All missing and borderline examples remain in
the roster; no class balancing or outcome-based selection is added. These
videos already have reported classical-baseline results, so this is not an
untouched benchmark. The intended comparison is base versus the fixed trained
judge on video-disjoint correspondence examples, not camera-DNF validation.
Preparing panels performs no model inference. No test-derived model selection,
threshold adjustment, or success claim is authorized by this preparation.

Preparation completed on Anima under
`/data3/kmatzen/plumbline/research/tapvid-external-corpus-v1/corpus`:
1,740 planned examples and 971 labeled exports. The manifest and export are
retained locally in `results/tapvid-external-corpus-v1/`; image panels remain on
Anima. Every exported image hash, label/prompt mapping, source-code hash, and
export hash was checked. All test video names and example IDs are disjoint from
the development corpus. No learned-judge inference has been performed on this
partition at preparation time, and the calibration export/images were not opened. The omitted
borderline/unavailable observations must remain visible in subsequent coverage
accounting rather than being counted as correct or usable.

The test runner now requires `--partition test --development-manifest <original>`.
It verifies that training used that exact development manifest, checks all
twenty test video names and example IDs against development, and requires the
same staging source and prompt. Test inference uses both base and adapted arms
with identical image payloads, fixed greedy decoding, and the unchanged parser;
planned and labeled counts remain explicit. The planned tested adapter is the
completed fixed 20-pass model, not an adapter selected after viewing
test results. The runner is staged separately under the external-corpus directory;
the active training run's queued evaluator remains its original frozen file.
A CPU-only selection preflight verified all 971 test examples and image hashes.
No model weights were loaded or test responses generated during that check.

`analyze_tapvid_external.py` fixes the reporting policy before test inference:
the original training-majority class, the frozen FB/NCC close-or-abstain baseline,
and base/adapted confusion matrices, false-close counts, abstentions, and format
errors. Coverage is reported both on labeled examples and the entire planned
roster; unscored borderline/missing cases are not imputed as correct or wrong.
The paired base-to-adapted macro-accuracy difference uses whole-video bootstrap
resampling (20,000 draws, seed 677), retaining within-video dependence and equal
video weight. Videos without labeled examples are listed explicitly. The
percentile interval is descriptive uncertainty, not proof of external validity.
Input hashes must match the completed fixed 20-pass training manifest, original
development manifest, test corpus, and original tracker results. Synthetic tests
check that test-label frequencies cannot choose the majority baseline and that
an all-abstaining model has zero answer coverage.

The fixed external comparison completed under `base-adapted-twenty-pass-v1`
on Anima. Both arms contain all 971 labeled examples from all 20 videos, with
matching processed-image hashes. The training, development, corpus, export,
tracker, and evaluator hashes were verified. Complete predictions and the
predeclared analysis are retained in
`results/tapvid-lora-twenty-pass-v1/external-test.json` and
`external-test-summary.json`.

The base model predicted far for every example: 400/971 correct (41.2%), the
same as the training-majority baseline. The adapted model scored 588/971
(60.6%), with 56.2% balanced accuracy. Its class recalls were 317/375 close,
208/400 far, and 63/196 occluded. Mean per-video accuracy rose from 39.7% to
62.5%; the paired whole-video difference was 22.9 percentage points, with a
descriptive 95% bootstrap interval of 9.9 to 37.7 points. Both arms returned
valid labels on every scored example, with no abstentions.

The adapted model made 444 close predictions, of which 127 were wrong
(71.4% precision). The frozen image check made 255 close predictions, of which
10 were wrong (96.1% precision), and abstained on 716 labeled examples. The
learned model therefore did not establish a safer acceptance check than this
classical baseline. Neither arm evaluates camera-level DNF. The full planned
roster remains 1,740: 971 labeled, 306 borderline, 170 unavailable candidates,
and 293 beyond the video. Model response coverage is 100% of labeled examples
but only 55.8% of the planned roster; excluded cases are not successes.

This result establishes that the fixed fine-tuning procedure improves this
correspondence classification task over its pinned base model on video-disjoint
examples. It does not validate static-scene membership, camera poses, or a DNF
decision. No model or threshold is selected using these test responses, and
calibration remains unopened. The result is retained in this research ledger;
it has not been added to the manuscript's reported camera experiments.

### Non-planar camera-control preparation

`render_layered_camera_control.py` adds a separate static scene at depths three,
four, and six world units, using the existing 24-frame pinhole camera path.
Two bounded foreground panels occlude an infinite rear wall. Twelve anchors
(four per surface) are fixed by geometry, not selected by tracking results.
The original planar controls and their results are unchanged. This preparation
addresses the planar-scene limitation of the sensitivity experiments; it does
not yet establish detection performance or a DNF threshold.

RGB rendering uses nearest forward ray/surface intersections and seeded
textures. Reference projections, surface IDs, and analytic visibility are
returned separately for scoring and must not enter a reference-free evaluator.
Tests check exact nearest-surface cases, independent homogeneous projections
for every frame, raster visibility away from silhouette boundaries, repeatable
RGB, and invalid reference rejection. Rounded pixels at a silhouette are not
substituted for analytic reference visibility. Seven combined planar/layered
renderer tests pass.

`probe_layered_camera_control.py` now runs the fixed sequential LK check and
the unchanged 14-change camera profile on seeds 0 through 9. All 12 anchors,
24 frames, and both observed-track and exact-visible-reference arms are retained
in `results/layered-camera-control-v1.json.gz`: 280 profiles and 3,360 slot
evaluations. The tracker receives only RGB and initial query coordinates.
Reference visibility and later coordinates enter only the scoring and explicitly
named oracle arm. No threshold was selected from these results.

Of 2,760 later observations, LK accepted all 2,760: 2,573 were visible and within
two pixels, 140 were occluded, and 47 were visible but more than two pixels off.
Nevertheless, all 120 original-camera tracks in each arm were jointly
compatible. Both signed pitch/yaw changes contradicted all 120 tracks in each
arm, whereas both signed forward/backward translations contradicted none.
Signed roll changes contradicted 40 tracks each. Negative lateral translation
contradicted 100 observed tracks versus 80 oracle tracks; positive lateral
translation contradicted 40 in both. Vertical translation contradicted all
120 observed tracks but only 100 oracle tracks in either direction. All 240
global-gauge slot comparisons preserved the original status.

Thus non-planar scene geometry does not remove the tested directional blind
spots, even with exact visible correspondences. More contradictions in the
observed arm are not automatically better: corrupted observations can create
additional contradictions. This remains one scene/path with correlated texture
seeds, not a camera-method benchmark or a functional failure label. These
controls stay outside the manuscript's reported experiments. Six focused tests
cover the new renderer/probe, including separate missing, occluded, and wrong
observation accounting and tracking without later reference inputs.

`analyze_layered_displacement.py` audits the image-space size of those changes
using the true scene points for scoring only. It keeps true 3D points fixed
except for the global-gauge control, where world coordinates change with the
cameras. It verifies the saved dependency hashes and oracle projections, and
collapses identical texture-seed oracle geometry to 12 unique tracks. Full
visible-query displacements are retained in `results/layered-displacement-v1.json`.

For each signed forward/backward translation, eight compatible tracks have
maximum coordinate displacement at most two pixels; four exceed two pixels.
The largest coordinate displacements are 6.06 and 6.60 pixels respectively.
Those above-tolerance tracks nevertheless admit a different fitted 3D point
within all image boxes. Signed lateral translations retain four/eight tracks
with above-tolerance true-point shifts; signed roll retains six above-tolerance
and two below-tolerance tracks each. Thus small perturbation size explains
some, but not all, compatibility. The audit does not convert these changes to
functional failures: reference-free feasibility permits refitting unknown
structure, unlike this true-point scoring calculation. Two tests verify
independent projections, consistent world/camera gauge changes, and unchanged
prefix projections under a suffix camera rotation. These are control diagnostics,
not manuscript DNF results.

`probe_layered_minimum_error.py` adds a classical continuous-error baseline:
numerically bracket the minimum coordinate-wise reprojection tolerance by
bisecting the existing feasibility solver. The fixed search is zero to 64
pixels with .01-pixel resolution, using all visible frames of each of the 12
unique oracle tracks for every camera variant. An unresolved solver result
stops the search without being treated as infeasibility. This is classical
bounded-reprojection fitting, not the proposed novel contribution.

All 168 brackets completed and their upper-bound 3D witnesses passed a separate
positive-depth/reprojection check. Identity and global-gauge controls lie in
[0, .0078125] pixels. Every tested nonidentity/nongauge track has a positive
numerical lower bound. Signed forward/backward translations require roughly
.08 to 1.68 pixels after structure refitting, below the previous two-pixel
gate but separated from exact reference cameras. Thus binary tolerance hid
differences present in continuous fitting error; this control does not establish
unobservability in the zero-noise limit. Real track uncertainty, staticness,
calibration, and application usefulness remain separate unresolved questions.
Full solver traces are preserved in `results/layered-minimum-error-v1.json.gz`.
Three tests check bisection against a known monotone oracle and preserve both
mocked numerical uncertainty and a real coincident-camera boundary case.

`probe_layered_observed_error.py` repeats the continuous baseline with the saved
RGB-derived tracks, without filtering observations using reference visibility
or coordinates. It keeps every seed, track, and camera variant, and compares
numerical error brackets rather than rounded point estimates. An overlap is
not a win; an unresolved solve is not assigned an error score. All positive
upper-bound witnesses are independently reprojected before saving.

The full 1,680 track/variant fits are retained in
`results/layered-observed-error-v1.json.gz`. All 120 rigid-gauge comparisons
have overlapping brackets. Across the 1,440 nonidentity/nongauge comparisons,
1,425 have higher error after the camera change, 11 have lower error, one has
overlapping brackets, and three are unresolved. All 11 lower-error cases occur
under the positive camera-local z translation; that direction has 108
higher-error cases and one overlap. The three unresolved cases occur under
positive x rotation. These are dependent track comparisons in one synthetic
scene, not independent method-level ranking trials.

This baseline can prefer a perturbed camera when supplied with imperfect RGB
tracks, despite preferring the reference on every exact-track control. It gives
a concrete requirement for an evidence-admission method: reduce these ranking
errors while retaining enough evidence to distinguish the other camera changes.
No admission rule or threshold has been fitted to these results. Five focused
tests cover the continuous solver and observed-track comparison, including
missing evidence, overlapping brackets, and unresolved solves.

`filter_layered_image_evidence.py` tests the existing fixed 17x17 source-patch
NCC threshold (.8) on the same sequential coordinates. Admission sees only
grayscale images and tracked coordinates, not cameras or reference labels;
initial supplied queries are retained. RGB hashes are reverified. The saved
filtered tracks are in `results/layered-ncc-control-v1.json.gz`, with their
unchanged continuous-error evaluation in `results/layered-ncc-error-v1.json.gz`.

The filter retains 2,376/2,760 later observations. All retained observations are
visible and within two pixels in this control; it removes all 187 incorrect or
occluded observations but also 197 correct ones. This is not a real-data
precision claim. Across the same 1,440 perturbed-track comparisons, 1,199 have
higher error, 240 have overlapping brackets, one is unresolved, and none has
lower error. The overlaps are all from slots 1 and 3 in every seed: the filter
removes every suffix observation from those tracks. Their remaining prefix
cameras are unchanged under every perturbation, so these are missing-evidence
ties, not successful discrimination. The other 100 seed/track cases distinguish
1,199 of 1,200 changes with one unresolved solve. All gauge comparisons overlap.

Thus the fixed classical filter removes the observed ranking reversals, but
does not recover evidence for the affected tracks. A proposed learned method
must be compared against this precision/coverage tradeoff, not just against
unfiltered tracking. No new threshold was fitted, and these development controls
remain outside the manuscript's reported camera experiments. Fifteen focused
tests pass across the layered rendering, tracking, admission, and scoring code.

`analyze_camera_evidence_coverage.py` makes the missing-evidence distinction
explicit in saved reports, without rerunning or changing the experiments. It
verifies input/dependency hashes, the complete seed/variant/slot roster, and
identical unique observation indices for paired scores. Identity and global
gauge are excluded from physical discrimination counts. A comparison with no
retained observation from any changed camera view is labeled
`no_changed_view_evidence`, while its raw bracket comparison is also preserved.
Having one such observation is only a necessary condition, not an observability
certificate or a usability verdict.

`results/layered-observed-coverage-v1.json` retains all 1,440 comparisons:
1,425 higher-error, 11 lower-error, three unresolved, and one overlapping pair.
`results/layered-ncc-coverage-v1.json` retains the same denominator: 1,199
higher-error, 240 without changed-view evidence, and one unresolved. Changed-view
evidence coverage is 100% versus 83.3%; higher-error comparisons account for
99.0% versus 83.3% of all planned comparisons. These dependent synthetic counts
are not method-level success rates. Four tests check missing-view evidence,
supported ties, numerical uncertainty, mismatched/duplicate indices, and identity
controls. This accounting is a protocol component, not a standalone novelty claim.

`probe_layered_direct_evidence.py` reuses the frozen first-to-target LK recovery
baseline (FB<=1 pixel, 17x17 NCC>=.8) on all layered-control queries. It makes
each match independently from the initial frame; a rejected intermediate match
does not prevent later matching. Images are rehashed, and later reference
coordinates/visibility are used only for scoring after predictions. No camera
trajectory enters matching. One regression test checks recovery after rejection.

The complete direct baseline retains 1,415/2,760 observations, all visible and
within two pixels in this control, versus 2,376 for filtered sequential tracks.
Its unchanged geometric comparison yields 718 higher-error comparisons, two
numerically unresolved comparisons, and 720 without any changed-view evidence
out of 1,440 planned comparisons. Thus direct matching does not recover the
lost evidence here; it reduces changed-view coverage from 83.3% to 50%. Its
zero observed ranking reversals cannot be reported without that coverage loss.
The matching records, full geometric solver traces, and coverage report are
preserved in `results/layered-direct-control-v1.json.gz`,
`layered-direct-error-v1.json.gz`, and `layered-direct-coverage-v1.json`.
These are development baselines on one synthetic scene/path, not manuscript
camera-DNF results or a learned-method comparison.

### Camera-level aggregation audit of the layered controls

`analyze_layered_camera_ranking.py` checks whether the preceding track-level
reversals actually imply trajectory-level ranking errors. It uses a simple
median over all 12 planned track error brackets. Unresolved values are bounded
by [0, infinity], not dropped or replaced by favorable scores. Monotonicity of
the median propagates those intervals to camera-score bounds. Both candidates
use identical retained observations; changed-view evidence counts remain
explicit. This aggregation was chosen after inspecting development results,
so it is an exploratory baseline, not a predeclared held-out outcome.

All three arms—unfiltered sequential, NCC-filtered sequential, and direct
matching—assign higher median error to the perturbed cameras in all 120
seed/physical-change comparisons. Thus the 11 per-track reversals did not cause
camera-level ranking reversals under this aggregate. Even the direct baseline
separates these particular candidates despite its lower changed-view coverage.
These are correlated comparisons on one scene/path, not 120 independent scenes.
The controls do not establish that a learned evaluator improves camera ranking.
They must not be used to claim such an advantage or validate a DNF cutoff.

Full paired bounds and coverage are retained in
`results/layered-observed-camera-ranking-v1.json`,
`layered-ncc-camera-ranking-v1.json`, and
`layered-direct-camera-ranking-v1.json`. Three tests check interval propagation,
unknown-score handling, strict separation, and invalid inputs. A meaningful
next camera-level validation needs real method outputs or a separate broader
scene/path set and a fixed scoring policy; more texture seeds of this control
would not supply that evidence.

### Continuous baseline on saved real camera methods

`probe_real_minimum_error.py` applies the same classical coordinate-reprojection
search to the saved DAGE/DA3 full-clip estimates on both TUM development clips.
It rechecks decoded RGB, prediction manifests/bytes, runner hashes, frame order,
and track provenance. Scoring uses the common published calibration, unchanged
RGB-derived tracks, and frames 1–89; it does not use reference-pose or depth
values. All twelve tracks enter median bounds, including unresolved scores.
The native-pixel calibration is explicit rather than borrowed from the renderer.

On the first static clip, DAGE's median bracket is [2.8125, 3.1875] pixels and
DA3's is [4.2227, 4.4609], with one unresolved track per method. On office,
DAGE's bracket is [2.0273, 2.0352] and DA3's is [3.3008, 3.3086], with no
unresolved tracks. All positive upper-bound witnesses pass independent
reprojection/depth checks. Complete traces are saved as `minimum-error-v1.json.gz`
in each clip's result directory. A test checks agreement with the frozen
synthetic solver and use of changed focal length/principal point.

Separately inspecting the existing first-clip reference metrics shows the same
preference: DAGE/DA3 Sim(3)-aligned ATE is .00643/.00796 m and relative rotation
error is .315/.358 degrees. These are prior validation metrics, not score inputs
or functional failure labels. No matching office reference-metric file was
found in that clip's result directory during this audit; its geometric preference
was not independently validated at that stage. Two development clips and two methods do
not establish general ranking accuracy, a novel advantage, or a DNF cutoff.

`validate_real_camera_baseline.py` subsequently validates both frozen scores
against reference poses in a separate step. It rechecks the ground-truth file
hash and every saved nearest-timestamp association, requires unambiguous reference
poses for all original scored frames 1–89, and revalidates RGB/prediction
provenance. Reference poses are not passed to the geometric scorer. Results are
saved as `minimum-error-reference-validation-v1.json` in each clip directory.
The first clip exactly reproduces the previously recorded metrics. On office,
DAGE/DA3 aligned ATE is .00700/.00856 m, relative translation error is
.00678/.00886 m, and relative rotation error is .239/.266 degrees. All three
metrics agree with the lower geometric score for DAGE. This is two development
comparisons, not general ranking validation or human usability evidence.
One test checks association verification and that missing reference poses are
not imputed. The score and validation artifacts remain separate and preserve
their input hashes and all per-method results.

### Sensitivity to the selected real-video tracks

`audit_real_track_selection.py` enumerates all 4,095 nonempty subsets of the
twelve existing tracks for each real clip. Every comparison uses identical
slots and observation indices for both methods. Unresolved scores retain their
[0, infinity] bounds. The complete-roster score is not replaced by a favorable
subset, and all subsets are saved in each clip's
`track-selection-sensitivity-v1.json.gz`.

On the first static clip, all subsets retaining at least ten tracks still
prefer DAGE. Removing three tracks can make the median intervals overlap;
no subset strictly prefers DA3. On office, every subset retaining at least five
tracks prefers DAGE. With four retained tracks, 27 of 495 subsets prefer DA3;
eight removals are therefore the minimum needed to reverse that comparison.
These are exhaustive deletion checks within the observed roster, not independent
samples, confidence intervals, evidence about unseen tracks, or human judgments.
The audit is exploratory and uses no reference poses. It does not establish
a learned evaluator's advantage or a functional failure threshold.

Three focused tests check exhaustive enumeration, a known reversal, unresolved
score accounting, and rejection of mismatched paired observations or duplicate
slots. The preceding reference-validation revision passed 1,972 non-weight
tests, with one unstaged-KITTI skip and six weight tests deselected; the three
new audit tests also pass separately.

### Fixed dynamic-scene method comparison

The next development comparison uses `rgbd_dataset_freiburg3_walking_xyz`.
`results/walking-xyz-rgbd-comparison-v1/protocol.json` fixes the sample, methods,
inference settings, geometric baseline, and separate reference validation before
new inference. This sequence was used previously for DAGE development and is
not held out. It adds moving-object footage to the two preceding comparisons;
it does not by itself test a learned evaluator or establish novelty.

The staging helper now explicitly supports this sequence while preserving the
same first-90-RGB, stride-three rule, chosen before reference/depth association.
Its record identifies the scene as dynamic. The inference runners use the same
settings and now describe their scope as fixed-RGB development, rather than
assuming a static scene. Eleven targeted staging/input/preservation tests pass,
including the added sequence. Outputs and conclusions will be recorded only
after their execution and verification; the saved protocol is not evidence of
a completed experiment.

Staging completed on Anima with all 90 planned RGB frames. The saved
`staging-manifest.json` records 89 reference associations (only frame 0 is
unavailable) and 83 depth associations. Thus all preselected scoring frames
1–89 have reference associations for the later validation step. Depth is not
an input to inference or geometric scoring. Camera inference is a separate
step; staging success does not establish successful method outputs.

Both fresh runs subsequently completed: DAGE in 185.2 seconds including model
load, and DA3 in 24.9 seconds under its recorded timing convention. Their
manifests, predictions, executed runners, and DA3 helper scripts are retained
under `dage-v1/` and `da3-v1/`. Source RGB bytes, decoded arrays, frame order,
prediction hashes, runner hashes, and all ninety rigid poses were verified.
The original fixed-grid tracker retained 215 of 1,068 planned scored
observations, with per-slot counts `[9,52,4,3,3,7,53,23,5,3,53,0]`.
No track provides evidence after frame 53: frames 54–89 are entirely uncovered.
The twelve slots remain in the aggregate, including the empty slot.

The frozen RGB-only geometric median bounds are [1.3008, 1.8203] pixels for
DAGE and [1.9688, 3.2070] for DA3. Both have one unresolved slot. The score
was saved before separate reference validation. DAGE/DA3 aligned ATE is
.01501/.02086 m, relative translation error .01266/.02042 m, and relative
rotation error .3197/.3370 degrees on the fixed original frames 1–89.
All three metrics agree with the geometric preference, but they do not
validate functional usability or the missing image evidence in the final
36 frames. In particular, a low geometric error is not evidence that those
unobserved frames are accurate. The next evaluator comparison needs a way
to obtain and validate evidence across that interval; it must not infer
whole-clip reliability from the surviving early tracks.

The full score traces and separate reference report are retained as
`minimum-error-v1.json.gz` and `minimum-error-reference-validation-v1.json`.
Only descriptive scope text in the scoring/validation helpers was updated
to avoid incorrectly describing this third comparison as two clips; the
score, calibration, frame roster, and reference metric rules did not change.
The two focused score/association tests pass. These remain development
results outside the manuscript and do not establish a novel learned method.

### Applying the frozen correspondence judge to camera projections

`build_camera_judge_corpus.py` connects the trained image judge to the actual
DAGE/DA3 estimates on all three RGB-D development clips. For each original
slot, it fits a 3D point using only frames 1–10, then projects that fixed point
into frames 30, 60, and 89. The source marker uses frame 1. Later tracked
positions are not read to create candidate markers or center their crops.
Panels use exactly the existing correspondence renderer and prompt, without
method names. Missing prefix observations, degenerate fits, nonpositive depth,
and off-image projections remain explicit outcomes, not failure labels.
No replacement point or favorable query is selected.

The complete 216-presentation roster yields 58 static, 57 office, and 14
walking panels. The other 87 presentations remain in the manifests. One
walking endpoint panel was inspected visually for display correctness; this
was not a correspondence label or camera-quality judgment. Prefix residuals
are recorded, not used to select a cutoff. The fitted point is not certified:
tracking, staticness, calibration, point fitting, and cameras can all affect
its apparent placement.

`results/camera-marker-judge-v1/protocol.json` fixes base-versus-adapted
inference with the existing twenty-pass adapter, original prompt, and decoding
settings. The three corpus manifests are retained alongside it; source-image
panels stay local and on Anima. `evaluate_tapvid_lora.py --partition camera`
requires the full presentation roster and frozen prompt, verifies image/export
hashes, rejects supplied targets, and preserves all unavailable outcomes.
It checks the original training manifest and adapter files without opening
calibration exports or images. Camera inference produces unlabeled predictions,
not an accuracy table, camera ranking, or functional DNF result. Execution is
separate from corpus construction; no judge result is claimed here yet.

Twenty-two targeted tests pass, including tests that later track loss does
not prevent projection, later observations cannot change the prefix fit,
invalid projections remain unavailable, the complete roster is required,
and targets or changed prompts are rejected. Existing external-evaluation,
training-membership, and analysis tests remain passing.

`analyze_camera_judge.py` audits completed camera-application reports against
their corpus, training-manifest, export, and executed-evaluator hashes. It
requires every renderable presentation exactly once for each arm, re-parses
the raw response, rejects non-null target labels, and checks paired processed
pixel hashes and image-token counts. Per-method reporting keeps all 36 planned
presentations, with unavailable reasons separate from predicted labels. It
also records base-to-adapted response changes. It deliberately reports no
accuracy, camera winner, or DNF rate without reference judgments. Six new
audit tests and the four corpus/selection tests pass. Running or partial
reports cannot be analyzed as complete results.

The first static-clip application completed and passed that audit. Both arms
saw identical processed pixels and image-token counts for all 58 renderable
panels. Each camera method has 29 presented and seven unavailable cases out
of 36 planned presentations. The base model answered `far` for every panel.
The adapted model predicted 14 close / 11 occluded / 4 far for DAGE and
13 close / 13 occluded / 3 far for DA3. These are unverified predictions:
the one-count difference in close responses is not a camera preference or
accuracy improvement. Full responses and audited accounting are retained in
`results/camera-marker-judge-v1/static-evaluation.json` and
`static-summary.json`. The remaining clip runs are separate and are not
represented by this completed result.

The office application also completed and passed the paired-input/provenance
audit. DAGE has 28 presented / eight off-image cases; DA3 has 29 presented /
seven off-image cases, each out of 36 planned. The base model again predicted
`far` on every presented panel. The adapted predictions are 11 close / 10
occluded / 7 far for DAGE and 12 close / 4 occluded / 13 far for DA3. Neither
close-count ordering nor the different occlusion frequencies establish a
camera-quality result: labels are unverified, availability differs, and
prefix-fit error can also affect placement. Raw and audited reports are
`office-evaluation.json` and `office-summary.json` in the same result directory.
The full non-weight suite passes: 1,986 tests, one skip for an unstaged KITTI
bundle, and six weight tests deselected. This validates the tested code paths,
not the correctness of the model's visual predictions.

The walking application then completed and passed the same audit. Each method
has seven presented cases out of 36 planned: 24 are unavailable because of
missing prefix observations, two because the query projection is off-image,
and three because the fitted point has nonpositive prefix depth. The base
predicts `far` on all fourteen panels. Adapted DAGE predictions are three
occluded and four far; adapted DA3 predictions are seven far. Reports are
`walking-evaluation.json` and `walking-summary.json`. These judgments do not
establish that either trajectory failed, nor that all marked details are
actually far or occluded.

The entire fixed application is complete: 216 planned presentations, 129
renderable panels, 87 unavailable presentations, and 258 paired-arm responses.
The base predicts far throughout. Adapted totals are 50 close, 41 occluded,
and 38 far. Neither arm abstains or emits an invalid label. Absence of
abstention is an observed behavior, not calibrated confidence. The integration
and provenance checks pass, but correspondence correctness on these panels
and camera-level usefulness remain unvalidated. No DNF rate, method ranking,
or fine-tuning accuracy improvement follows from these unlabeled counts.

### Camera-projection correspondence transfer with independent point labels

The twenty existing TAP-Vid test videos do not overlap the three DAVIS camera
development clips (`bear`, `boat`, `bmx-bumps`). They contain reviewed point
positions and visibility, which can validate a judge's answer about a projected
marker without treating that judge's answer as truth. They do not annotate
staticness, camera poses, or functional camera usability. This validation
therefore targets candidate-position correspondence, not a camera DNF rule.

`results/tapvid-camera-transfer-v1/protocol.json` fixes a forty-attempt camera
batch: DAGE and DA3 on every original frame of all twenty videos, with the same
pinned checkpoints and prior inference settings. `run_tapvid_cameras.py` reads
only the RGB array, verifies its staged byte and decoded hashes, and saves poses
and estimated calibration. It does not load point/visibility annotations.
`run_tapvid_camera_batch.py` runs attempts in separate processes, records process
failures, and continues the full roster. A zero process exit is not labeled as
an independently verified prediction result. Three tests check RGB-only access,
partition restrictions, attempt preservation, and complete accounting after a
failed process. The batch is separate from subsequent marker construction and
judge scoring; successful execution is not assumed here.

The later marker protocol uses each frozen track's first-visible query and ten
consecutive early RGB observations, then projects without later refitting at
lags 15/30/60. Only afterward may the annotated positions/visibility label the
candidate, using the unchanged <=2 / >=8 native-pixel rule and explicit
occluded/borderline outcomes. The original twenty-pass judge remains frozen.
These test videos were already used for prior correspondence comparisons, so
this is a declared transfer study, not a fresh untouched benchmark. Moving
points and uncertain prefix geometry remain limitations for camera-level use.

`tapvid_camera_markers.py` implements that geometry/reference separation.
`build_tapvid_camera_transfer.py` verifies the original RGB input and camera
outputs, freezes every planned candidate in `geometry.json`, and only then
opens the later point/visibility references for scoring. Source queries are
checked against the original first-visible annotations after geometry is
frozen. Later annotated positions never enter the fit or projection. The
reference label cannot change a candidate. Seven targeted tests cover this
path, including synthetic end-to-end construction, poisoned later track and
scoring metadata, missing prefixes, occlusion, and borderline labels.

The first completed video, bike-packing, retains all 150 planned method/track/
lag presentations: 38 far, nine occluded, 18 off-image query projections, 65
nonpositive-prefix-depth cases, and 20 queries beyond the video. There are no
close or borderline examples in this video. This class balance is retained;
the protocol does not replace the video or adjust the threshold. These are
candidate correspondence references, not camera-failure labels or judge
results. Both camera outputs and the frozen geometry/export hashes were
verified locally. The corresponding manifests and geometry are retained
under `results/tapvid-camera-transfer-v1/`; image panels remain on Anima.

The initial builder import on Anima stopped before creating output because
its installed package lacks `plumbline.image_evidence`. A pinned copy is loaded
from this experiment's isolated `geometry-support` directory through a
process-local package search path. No shared repository or installed package
was changed. The result records the actual geometry/conventions helper hashes.
The forty-attempt camera batch continues independently; this first corpus is
not a substitute for the remaining planned videos or for evaluator validation.

Five video corpora have now been prepared and hash-checked: bike-packing,
blackswan, bmx-trees, car-roundabout, and car-shadow. Their 600 planned
presentations contain 75 far and 34 occluded labeled candidates, 103 off-image
queries, 298 nonpositive-prefix-depth cases, and 90 beyond-video queries.
There are still no close candidates in this partial set. This is an availability
and label-distribution result, not judge accuracy; the full twenty-video
roster remains unchanged. Every per-video geometry, manifest, and export is
retained under the transfer result directory.

`combine_tapvid_camera_transfer.py` requires all twenty per-video corpora before
creating a combined test input. It checks hashes and the complete two-method,
all-track, three-lag roster, retains unavailable and borderline cases, and
copies verified panels without altering them. The existing evaluator's test
path additionally validates this transfer roster and describes the run as
camera-position correspondence transfer rather than camera-DNF validation.
Fourteen targeted transfer/partition/builder tests pass. Combined corpus
construction and judge inference have not yet run.

The combiner includes a simple source-patch NCC comparison, reusing the prior
17-by-17 patch and 0.8 threshold. It predicts close or abstains; rejection does
not imply far. This baseline was selected for the transfer comparison after
the first video's reference labels were available, not preregistered before
all label inspection. Its threshold is unchanged and is not tuned to this
transfer set. It uses only RGB and marker coordinates, not reference labels.

The combined-input path now has a synthetic end-to-end test spanning all
twenty videos and both methods through `select_evaluation`. It verifies copied
panel hashes, full membership, and an image-only baseline that does not change
with the reference label. Combination also compares every scored record's
source/query coordinates, frames, method, and track against the frozen geometry
record; changed candidates or unexpected availability changes are rejected.
This binding check passes on all seven currently prepared real video corpora.
Ten targeted combination/roster/builder tests pass.

Cows and dance-twirl extend the prepared set to 810 planned presentations:
176 far, 62 occluded, seven borderline/unlabeled, 135 off-image queries,
340 nonpositive-prefix-depth cases, and 90 beyond-video queries. There are
still no close candidates. All outcomes remain recorded without modifying
the class thresholds or replacing videos. The complete transfer study is
still pending; these seven videos are not reported as the full test set.

Dog, drift-chicane, gold-fish, and horsejump-high extend preparation to eleven
videos and 1,320 planned presentations. Counts are 18 close, 277 far, 118
occluded, 16 borderline/unlabeled, 154 off-image queries, 507 nonpositive-prefix
depth cases, and 230 beyond-video queries. Seventeen close cases come from
drift-chicane and one from gold-fish. Horsejump-high has no evaluable candidates.
These outcomes remain in the fixed roster. Frozen geometry bindings, exported
file hashes, and consistency of saved distances/visibility with the unchanged
label thresholds were checked for all eleven corpora. This does not yet report
judge performance or validate a camera-quality metric.

`analyze_camera_transfer.py` prepares the separate accuracy analysis. It checks
complete paired responses, input hashes, the reference labels and exact parser,
and identical processed pixels/image-token counts. It compares base/adapted
predictions, the training-only majority label, and the fixed source-patch NCC
baseline. It reports classification metrics and availability separately, keeps
all twenty video entries (including those without labels), and leaves balanced
accuracy undefined when a reference class is absent. Descriptive paired
bootstrap intervals resample whole videos with labeled examples, keeping both
methods and their correlated observations together; excluded unlabeled videos
remain listed. Results grouped by candidate camera method describe judge
classification, not camera quality. Seventeen targeted analysis/metric/input
tests pass. The analysis is ready but has not been run on transfer judge
responses; the full camera batch and subsequent judge inference remain pending.

India, lab-coat, libby, and motocross-jump extend preparation to fifteen
videos and 1,920 planned presentations. Totals are 20 close, 392 far, 155
occluded, 38 borderline/unlabeled, 209 off-image queries, 564 nonpositive-prefix
depth cases, 490 beyond-video queries, 34 nonpositive-query-depth cases, and
18 missing-prefix cases. Libby has no evaluable candidates. Behind-camera
prefix fits and behind-camera query projections remain separate unavailable
outcomes, not DNF labels. All fifteen geometry/export hashes and frozen
candidate bindings pass verification. The complete twenty-video judge
comparison has not started.

The full non-weight test suite passes at this stage: 2,009 tests, one skip for
an unstaged KITTI bundle, and six weight tests deselected. This includes the
transfer geometry, reference-label separation, combined-input selection, and
accuracy-accounting tests; it is not evidence that the judge's answers or
camera-quality assessments are correct.

All forty camera attempts have now completed across the fixed twenty-video
roster. Their prediction arrays and manifests are archived under
`results/tapvid-camera-transfer-v1/runs/`. Independent local checks verified
prediction-file hashes, rigid poses, finite calibration with positive focal
lengths, the runner hash, and input metadata against the staging manifest.
Each method pair records identical per-frame RGB hashes. Package source hashes
are consistent across all forty runs, and checkpoint hashes are consistent
within each method. These checks validate the recorded inputs and output
structure, not camera accuracy.

All twenty per-video corpora are prepared and their frozen candidate bindings
and geometry/export hashes pass verification. The full roster contains 2,610
planned presentations: 30 close, 498 far, 170 occluded, 51 borderline/unlabeled,
and 1,861 unavailable. Unavailable cases comprise 889 nonpositive-prefix-depth,
578 beyond-video, 313 off-image-query, 51 nonpositive-query-depth, and 30
missing-prefix cases. None is converted into a camera DNF label. There is no
replacement of videos or adjustment of thresholds to change these counts.

The combined corpus and its fixed source-patch NCC baseline are saved in
`results/tapvid-camera-transfer-v1/combined-v1/`. The frozen base/adapted judge
comparison has started on all 698 labeled presentations, with identical inputs
for both arms and no further training. Its executed evaluator is archived.
Responses and classification results remain pending. The low close-class
count and substantial unavailable fraction require class-specific metrics and
separate availability accounting; overall accuracy alone would be insufficient.
This is a correspondence-transfer test, not yet validation of camera-level
quality, staticness, or functional DNF. No new claims are added to the paper.

An analysis audit added planned/labeled counts and unavailable reasons by
video, candidate method, and query lag, alongside classification metrics by
lag. It changes reporting only, not the frozen experiment. Nine analysis
tests pass, including all-unavailable and missing-class cases. Such groups
report undefined accuracy, not perfect accuracy, and remain in availability
accounting.

The completed corpus has 319, 260, and 119 labeled presentations at lags 15,
30, and 60 respectively, out of 870 planned at each lag. Close labels number
28, two, and zero respectively. Car-shadow, horsejump-high, and libby have no
labeled presentations. DAGE-origin candidates have 302 labeled presentations
(30 close, 193 far, 79 occluded); DA3-origin candidates have 396 (zero close,
305 far, 91 occluded), out of 1,305 planned for each. These are reference and
availability counts, not judge results or a camera-method ranking. In
particular, balanced three-class accuracy is undefined for the DA3 subgroup
and the 60-frame subgroup. These differences must remain visible in the final
analysis rather than being hidden by aggregate accuracy. Judge inference is
still running; no parameters or thresholds have been changed.

### Exploratory RGB-track baseline on camera projections

`analyze_camera_transfer_lk.py` adds a separate comparison while the frozen
judge inference continues. This baseline was added after the reference counts
and partial base-model responses were observed; it is not preregistered. It
does not change the combined corpus, VLM inputs, or experiment thresholds.
It reuses the saved RGB-only LK observations and their existing checks:
forward/backward error at most one native pixel and source-patch NCC at least
0.8. For an accepted observation, distance from the camera-projected pixel to
the tracked pixel is classified as close at at most two pixels, far at at
least eight pixels, and uncertain otherwise. Missing or rejected observations
are uncertain, not far or occluded. No threshold sweep was performed.

The tracker is hash-bound to every per-video construction manifest. Only RGB
coordinates and checks enter its predictions; later annotation metadata is
excluded. All predictions are fixed before reference labels enter scoring.
The baseline shares early tracking evidence with the marker fit, but its later
image observations were not used to fit or refit those points. This dependence
and the possibility of tracking errors remain limitations.

The saved `exploratory-lk-baseline-v1.json` retains all 2,610 slots. On the 698
labeled presentations it answers 162 (23.2% coverage), with 160 correct and two
incorrect (98.8% accuracy conditional on answering). It abstains on 536. Of 16
close predictions, 15 are correct; it detects 15 of the 30 close references.
It makes no occlusion predictions. These are correspondence metrics, not
camera quality or DNF. The original patch-only gate answers 14 labeled cases,
nine correctly, while the training-majority far baseline scores 498/698 but
detects no close or occluded cases. These comparisons require both coverage
and class-specific reporting. Nine new tests check reference-label separation,
threshold boundaries, missing-evidence abstention, source binding, and duplicate
frame rejection. VLM transfer results remain pending; no paper claims are added.

The final transfer analysis accepts `--tracker` to recompute this exploratory
baseline from its hash-bound input, rather than trusting saved aggregate scores.
It reports each arm on the same two labeled subsets: RGB-track answered and
RGB-track abstained. Missing and borderline reference labels remain outside
accuracy scoring and inside the planned roster. This descriptive stratification
was added before completed adapted-model responses; it is not a trained fallback
policy or evidence of camera-level failure detection. Nineteen targeted analysis
and baseline tests pass, including paired subset membership and rejection of an
incomplete baseline roster. The running VLM experiment is unchanged.

### MOVi-E geometry feasibility pilot

The next unresolved issue is distinguishing independent scene motion from
camera error. The [official MOVi documentation](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md)
describes rendered RGB, camera and object poses, depth, and instance masks.
Its `is_dynamic` field describes the initial setup, not whether an object stays
stationary throughout the clip. A future static-support reference must use
actual motion and validated projection conventions, not this flag alone.

`stage_movi_geometry_pilot.py` stages only the first validation shard of
MOVi-E/256x256/1.0.0, selected before decoding scenes. The 115,302,799-byte shard
is on Anima under `/data3/kmatzen/plumbline/research/movi-geometry-pilot-v1/`.
Publisher sizes and MD5 checks pass for all four downloaded objects; generation
IDs, URLs, and SHA-256 hashes are archived with dataset metadata under
`results/movi-geometry-pilot-v1/`. The script preserves existing directories and
failed attempts. Two tests cover checksum/size rejection and no overwrite.

This is parser and geometry development, not held-out evaluation, a replacement
for real-video validation, or evidence of evaluator performance. No scenes have
yet been decoded, no camera methods run, and no DNF labels inferred. Downstream
work must retain every scene in this shard and keep reference annotations out
of camera/evaluator inputs. The existing DAVIS judge comparison continues
unchanged on the GPU; staging this small pilot used CPU/network only.

All four scenes in the selected shard are now decoded: 3835, 5397, 4385, and
4970, each with 24 frames and respectively 13, 8, 10, and 16 objects. RGB-only
NPZs and separate reference NPZs remain on Anima in `decoded-v1/`. Their hashes,
raw record hashes, scene membership, and decoder dependency versions are
archived in `decoded-manifest.json`. Reader dependencies were installed into a
separate pilot directory, without changing the shared environment or running
judge. Integer depth/object-coordinate precision is retained; PNG channel
order and shape checks are tested.

`audit_movi_camera_metadata.py` independently projects the released object
positions using the camera-to-world wxyz quaternions and the camera convention
in the [upstream implementation](https://github.com/google-research/kubric/blob/main/kubric/core/cameras.py).
All 1,128 object-center/frame comparisons agree with released image positions
within 0.000021 native pixels. Per-comparison residuals and input hashes are
saved in `metadata-projection-audit-v1.json`. Seven pilot tests pass. This is
metadata consistency, not validation of rendered surface coordinates, depth
decoding, visibility, static support, or a camera-evaluation method. Those
checks are still required before this pilot can supply task-level references.

`audit_movi_surfaces.py` now checks the rendered foreground surface data on a
fixed eight-pixel grid across every frame. Object-local coordinates are mapped
through the ordered world-space bounding-box corners and projected using the
reference camera. Decoded radial depth is compared with distance from that
world point to the camera. The TFDS builder's uint16 depth conversion is used;
the current README's raw-TIFF description is not applied to these encoded
TFDS samples. No thresholds were used to select or discard foreground samples.

There are 98,304 planned grid locations and 24,437 foreground samples across the
four scenes. All foreground samples have positive camera depth. Maximum
same-frame reprojection error is 0.004507 pixels; maximum radial-depth
disagreement is 0.000391 scene units. Per-scene medians, 95th percentiles, maxima,
and every sampled residual are retained in `surface-audit-v1/`. Same-instance
3x3 interior statistics are reported separately without dropping boundaries.
Two additional tests cover corner ordering, rigid transforms, boundary
retention, and radial versus axial depth. This checks same-frame surface
consistency only. Cross-frame visibility, static-support selection, and
camera-method evaluation remain pending; no functional-DNF labels result.

`audit_movi_temporal.py` transports every foreground point on the fixed
frame-zero grid through each object's released bounding-box trajectory. The
1,058 source points yield 25,392 point-frame records; 24,405 project within the
image with positive depth. All records remain in `temporal-audit-v1/`, including
off-image points. It saves world displacement, projection difference between
the actual point and a hypothetical source-frozen world point under the same
reference camera, four-neighbor radial-depth ranges, and mask-match counts.
Unobservable neighborhoods are explicitly unavailable. No visibility threshold,
staticness label, or DNF verdict is produced.

The largest observed displacement among sampled points on initially still
objects is 3.842 scene units. Thus the initial setup flag cannot substitute for
actual motion evidence in this pilot. The same-frame identity checks stay below
0.003508 pixels at the sampled source locations. This does not validate later
visibility: depth-neighborhood agreement and same-instance masks alone can miss
self-occlusion or nearby surfaces. The upstream point-tracking implementation
also uses surface-normal checks; this diagnostic is not claimed to reproduce
its full visibility procedure. Two new tests separate object motion from camera
motion, preserve off-image records, and ensure the initial flag does not override
geometry. No candidate methods or evaluator accuracy have been measured here.

The released normal PNGs are now extracted separately in `normals-v1/`, with
the exact scene order checked against the decoded pilot. Existing RGB and
reference arrays were not changed. `normals-manifest.json` archives hashes.
`audit_movi_visibility.py` combines normal orientation with the saved depth
neighborhoods and instance-mask evidence. It follows the depth, mask, and normal
checks described in the
[upstream point-tracking code](https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py),
but is not an exact reproduction: it uses the fixed frame-zero grid, conservative
image-center bounds, and leaves unreliable normals unresolved rather than
ignoring them. All component flags remain available, including overlapping
reasons for rejection.

The audit yields 9,706 evidence-supported, 4,105 rejected, and 11,581 unresolved
point-frame records out of 25,392. Rejection components overlap: depth 2,967,
mask 954, normals 1,017, and outside/behind 987. These are algorithmic visibility
diagnostics, not independently validated labels or camera outcomes. Results and
all per-point flags are archived in `visibility-audit-v1/`. Three tests verify
normal transport, uncertainty handling, and preservation of off-image cases.
Rendered surface orientation and conservative visibility still need independent
checks before these records can serve as evaluation references.

### Completed frozen camera-position transfer comparison

The full comparison is complete. `results/tapvid-camera-transfer-v1/` now
contains `judge-evaluation-v1.json` and `judge-summary-v1.json`. Both arms have
all 698 labeled presentations with matching processed-pixel hashes and token
counts. Input, training, evaluator, and corpus hashes pass analysis checks.
An independent direct confusion-matrix recount matches the summary. Neither arm
abstains or produces an invalid response. The 1,861 unavailable and 51 borderline
presentations remain outside classification accuracy and inside the full roster.

The base model answers far on all 698 cases: 498 correct (71.35% accuracy), with
33.33% balanced accuracy and no detected close or occluded cases. The adapted
model is correct on 334/698 (47.85%); recalls are 20/30 close, 244/498 far, and
70/170 occluded, giving 52.28% balanced accuracy. However, only 20 of its 181 close
predictions are correct (11.05% close precision): 149 far and 12 occluded cases
are incorrectly called close. This does not support an automated quality gate.
Improved balanced accuracy must not obscure these false acceptances.

The exploratory RGB-track baseline answers 162 cases, 160 correctly. On exactly
those cases the adapted judge is correct on 55/162. On the remaining 536 labeled
cases where tracking abstains, the adapted judge is correct on 279/536 (52.05%),
versus 352/536 (65.67%) for the training-majority far answer. Its close precision
on that abstention subset is 7.69%. Thus this result does not establish a useful
automatic fallback to the learned judge. Abstention is not treated as a wrong
class label when discussing the baseline's conditional accuracy.

Across the seventeen videos with labeled presentations, macro accuracy is
73.38% base and 51.88% adapted. The descriptive paired-video bootstrap interval
for adapted-minus-base macro accuracy is [-0.333, -0.102], with mean -0.215.
The other three videos remain explicitly listed as unlabeled. These are not
independent-frame confidence claims, a fresh untouched benchmark, camera-method
rankings, or functional-DNF validation. This development result is not added to
the manuscript. No failed cases are erased and no post-result thresholds or
training changes are substituted into the frozen experiment.

The full non-weight suite after the MOVi audit additions passes: 2,036 tests,
one skip for an unstaged KITTI bundle, and six weight tests deselected. This is
software verification, not validation of the proposed research contribution.

### Fixed RGB-only camera pilot on MOVi-E

`run_movi_cameras.py` and `run_movi_camera_batch.py` start eight fresh camera
attempts on the existing four development scenes, two methods per scene. The
protocol is archived in `results/movi-camera-pilot-v1/protocol.json` and binds
the exact decoded input manifest. No new scenes are selected. DAGE and DA3 use
the same pinned checkpoints and 252-resolution settings as the completed DAVIS
camera runs. Every attempt remains recorded; the batch continues after process
failure and refuses to overwrite previous runs.

The loader accepts only the RGB-only NPZ and checks its file and decoded-array
hashes. It does not open reference NPZs, and it rejects archives carrying extra
arrays. All 24 frames enter inference, with intrinsics predicted by each method.
Four tests cover RGB/reference separation, frame retention, hash rejection,
preservation of prior attempts, and continuation across all eight attempts.
The GPU was confirmed free before launch. Outputs are under Anima's
`/data3/kmatzen/plumbline/research/movi-camera-pilot-v1/runs/`.

This pilot begins after the original judge transfer comparison is complete and
does not change it. It is not held-out validation. The purpose is to compare
predicted cameras against scene-motion-aware reference measurements in a
controlled setting; no camera-quality threshold or functional-DNF label has
been established. All eight inference attempts have now completed.

`evaluate_movi_cameras.py` separately verifies saved prediction and reference
hashes and evaluates all 24 poses and 23 adjacent pairs per attempt. The
reference conversion changes Blender camera axes to OpenCV axes. Position
alignment uses Sim3; rank-deficient trajectories remain unresolved. Rotation
matrices are normalized before angle calculations, with the correction recorded.
Four tests cover camera axes, quaternion signs, missing frames, static-camera
alignment, similarity invariance, and finite-precision rotation errors.

`reference-evaluation-v2.json` is the current evaluation. Version 1 is retained
as a development record: it computed direct rotation angles before matrix
normalization and is superseded. ATE favors DAGE on scenes 3835 and 4970 and
DA3 on 5397 and 4385. Adjacent translation error favors DAGE on three scenes
and DA3 on 5397. These conventional pose measurements provide comparison
targets; they do not validate the new evaluator or define functional DNF.
No result from this pilot has been added to the manuscript.

### Frozen MOVi marker proposals and separate motion audit

`build_movi_marker_geometry.py` uses RGB only and the saved cameras. It starts
64 fixed pixel slots at coordinates 12,44,...,236 on each axis, retains lost
tracks, and fits each marker from frames 0..9. Query frames are 12,18,23.
Later RGB tracks are saved but cannot change the fit. No reference mask or
motion annotation selects points. All four scenes and both camera methods
produce 1,536 planned proposals: 944 are inside the image, 546 lack a complete
fitting track, 43 project outside, and 3 have nonpositive fitting depth.
The four `marker-geometry-*-v1.json` files preserve every slot and fit.

The Anima run loaded the private copy of `plumbline.image_evidence`, without
changing the shared installation. Its SHA256 matches the local source:
`9fc932d5f4cea4008a9dc43106fe4075e67dff17274000853cd511bff17414f4`.

Only after those files were saved, `audit_movi_marker_motion.py` joined them
to the existing temporal and visibility diagnostic records. The raster-center
conversion is explicit: reference coordinates subtract 0.5 before comparison
with LK coordinates. All 1,536 proposals remain in `marker-motion-audit-v1.json`.
There are 366 foreground-reference proposals and 1,170 without a foreground
reference. Foreground visibility evidence is supported for 120, unresolved for
158, and rejected for 88, including unavailable proposals. These are diagnostic
cues, not independently verified visibility labels. No binary staticness or
DNF threshold is introduced, and no accuracy or camera ranking is inferred.
Three tests check fixed-slot retention, exclusion of later observations from
marker fits, coordinate conversion, and preservation of unresolved references.

### Background references for the same MOVi proposals

`audit_movi_background.py` follows the released point-tracking convention for
static background: unproject frame-zero pixels with radial depth and retain
their world positions. This is a separate reference operation, never an input
to camera inference or marker construction. The 195 background points cover
the 1,170 previously unreferenced proposals. All 24 frames remain stored with
depth/mask rejection diagnostics; non-rejection is not certified visibility.

An additional first-step consistency check compares these projections with
the released rendered forward flow, decoding its quantized range and row/column
order. All 195 points agree within 0.000879 pixels. The depth and flow share
the renderer, so this is a convention check, not independent physical truth
or a long-range visibility test. Raw first-flow arrays and all residuals are
archived in `results/movi-geometry-pilot-v1/background-audit-v1`.

`audit_movi_background_markers.py` joins those references to the already saved
RGB proposals and later RGB tracks. It retains all 1,536 proposals and reports
paired method summaries only on common renderable background slots not rejected
by reference depth/masks. Marker errors also contain prefix-tracking and
intrinsics errors; they are not pure camera errors or functional-DNF labels.
Tests cover radial depth off axis, rotated-camera round trips, retained
occlusions, flow channel order/quantization, and exact replay of the archived
marker audit. The manuscript is unchanged.

Reference implementations:
[Kubric point tracking](https://raw.githubusercontent.com/google-research/kubric/main/challenges/point_tracking/dataset.py)
and [MOVi-E format](https://raw.githubusercontent.com/google-research/kubric/main/challenges/movi/movi_e.py).

### Frozen visual-judge comparison on MOVi markers

`build_movi_judge_corpus.py` renders all 944 available proposals from the fixed
1,536-slot roster. It opens only verified RGB archives and frozen geometry,
not reference arrays. The exported inference rows contain only ID, image path,
and the original correspondence prompt. Panels stay on Anima; their hashes,
the complete roster, and the export are archived under
`results/movi-camera-pilot-v1/judge-corpus-v1`.

The RGB baseline uses the saved later LK track, forward-backward error <=1
pixel, and source-patch NCC >=0.8. It answers close at <=2 pixels and far at
>=8 pixels; otherwise it abstains. This gives 337 close, 6 far, and 1,193
uncertain responses across all planned slots. Those are predictions, not
accuracy figures. Thresholds are unchanged from the earlier comparison.

The evaluator's unlabeled-camera mode now also checks the exact four-scene,
two-method, 64-slot, three-query MOVi roster and fixed source grid. The prior
72-example camera mode remains supported. Ten targeted tests pass, including
rejection of missing proposals and targets in inference rows.

A fresh base/adapted run was launched after checking GPU availability and
confirmed live with initial responses. It uses the completed twenty-pass
adapter, unchanged prompt, original quantized arithmetic, and both arms on
identical images. There are no updates, selection, or calibration reads.
Outputs are under Anima's
`/data3/kmatzen/plumbline/research/movi-camera-pilot-v1/judge-v1`.
The comparison is developmental and was added after reference audits; it is
not a held-out validation claim. Reference visibility remains diagnostic,
and correspondence agreement cannot establish functional camera DNF.

`analyze_movi_judge.py` checks exact proposal identity against both separate
reference audits. It reports diagnostic agreement only for the unchanged <=2
and >=8 pixel ranges, with depth/mask non-rejected backgrounds and
normal-supported foregrounds. It does not assign occluded labels from the
rejection cues. Unavailable geometry, uncertain/rejected visibility, and
borderline distances remain separate. Complete model comparisons additionally
require all 944 IDs in both arms, matching processed image hashes/token counts,
the original parser, and absent targets. Partial results cannot enter the
comparison. Results are split by scene, foreground/background, and the same
RGB-baseline answered/abstained subsets for every arm.

The baseline-only diagnostic is archived before the judge finishes. Among all
1,536 planned proposals, 389 have diagnostic labels, 393 are borderline, 162
have unresolved/rejected reference visibility, and 592 lack renderable geometry.
The baseline answers 247 of the 389 labeled cases and agrees on 244; it abstains
on the other 142. There are three false-close answers under these reference
cues. These figures measure agreement with diagnostic references, not verified
camera accuracy. The first fixed panel was visually inspected for layout;
that inspection is not a human judgment dataset, and no labels were added.

### All-step background-flow reference check

`audit_movi_background_flow.py` extends the first-step convention check to
all 195 fixed background sources and 23 adjacent pairs (4,485 planned pairs).
It samples released rendered forward flow bilinearly at the frozen projected
reference points. All points remain stored: 4,438 pairs permit image sampling,
and 3,847 have all four neighboring pixels labeled background and pass the
existing depth/mask checks at both endpoints. The latter subset was specified
before running this audit, not selected by measured flow error.

On that subset, the maximum displacement disagreement is 0.000974 pixels.
Outside it, disagreements reach 30.02 pixels, illustrating why flow from a
different visible surface must not be treated as the tracked background's
motion. The archive retains these cases and out-of-image pairs. This supports
the temporal coordinate conversion, not independent visibility truth or the
visual judge's accuracy. It does not change the current judge inputs or the
baseline diagnostic's reference policy.

Records and hashes are under
`results/movi-geometry-pilot-v1/background-flow-audit-v1`. Two new tests check
fractional-pixel interpolation, no extrapolation, temporal indexing, channel
order, and retention of rejected cases; all 37 MOVi and analysis tests pass.

### Reference-registered virtual-anchor task outcome

`evaluate_movi_anchor_task.py` adds a separate camera-level development target.
It registers each predicted trajectory to the reference using frame-zero
orientation and origin, then a positive least-squares scale from frames 0..9.
Later reference poses cannot change registration. It projects the 195 known
static background points through the candidate cameras over frames 10..23.
This differs from the RGB marker test: the world points are known, so errors
from fitting points to image tracks do not enter this outcome.

The primary arm uses predicted intrinsics; a reference-intrinsics ablation
helps identify calibration effects. All eight attempts and 5,460 planned
later point-frames per calibration arm remain recorded. There are 4,804
reference-eligible point-frames. Behind-camera candidates, off-image
projections, and unresolved registration have separate counts; positive-depth
off-image errors remain in summaries. A stationary prefix can leave scale
unresolved, which is not automatically a DNF.

On these four development scenes, mean anchor drift with predicted intrinsics
favors DAGE in all four, unlike ATE's two/two split. On scene 5397, DAGE/DA3
mean drift is 3.45/17.59 pixels with predicted intrinsics and 3.96/1.89 with
reference intrinsics. This shows that the chosen application outcome and
calibration can change the comparison; it does not validate the proposed
RGB-only evaluator. The result is specific to this registration protocol and
four scenes, defined after earlier audits, not a held-out claim.

An oracle check through the same registration/projection functions, using
reference cameras and reference intrinsics, gives maximum residual below
2.35e-13 pixels on every scene. Four tests cover gauge invariance, exclusion of
later reference poses from registration, unresolved scale, complete candidate
accounting, and replay of the saved array summaries and hashes. Artifacts are
in `results/movi-camera-pilot-v1/anchor-task-v1`. No binary DNF threshold is
chosen and the manuscript is unchanged.

### Calibration identifiability control

`probe_focal_depth_ambiguity.py` checks whether correspondence consistency can
certify the known-world anchor task in every setting. It cannot. For parallel
cameras moving sideways, replacing focal length f by a*f and every scene depth
Z by a*Z preserves u=f*(X-Cx)/Z+cx and v=f*Y/Z+cy. The control uses four distinct
depths, 24 poses, and factors 0.5,0.8,1,1.25,2. Each alternative fits the first
ten observations and forecasts later points to within 1.46e-11 pixels, yet
mean known-world anchor error reaches 33.11 pixels when the anchor coordinates
are not changed. Both calculations use the existing projection/fitting code.

This is an instance of the established self-calibration critical-motion
problem, not a new theorem or failed model experiment; see
[Sturm, BMVC 1999](https://www.bmva-archive.org.uk/bmvc/1999/papers/07.html).
It does not show that a particular MOVi clip is exactly degenerate. It does
show why the newly added reference task must not be treated as an outcome
that RGB correspondence alone can always identify. Physical calibration,
known scene constraints, or explicit uncertainty are required in ambiguous
cases. Learned appearance priors do not constitute a geometric certificate.
The fixed control and its regression test are retained under
`results/movi-camera-pilot-v1/focal-depth-ambiguity-control-v1.json`.
No current judge input, reference label, or manuscript claim is changed.

### Constant-answer controls for the pending judge comparison

The 389 diagnostic labels contain 352 close and only 37 far cases. Therefore,
an image-free always-close response attains 90.49% diagnostic agreement while
detecting none of the far cases. `analyze_movi_judge.py` now reports all-close,
all-far, and all-uncertain controls without choosing one from the evaluation
results. All use the same renderable roster and paired reference subsets.
These controls were added after label counts were known and before the judge
finished; they do not change model inputs, thresholds, or weights.

Every arm also reports close and far recall with abstentions in the denominator,
and their mean only when both classes are present. This balanced agreement
is 0.5 for either constant decisive answer. The selective RGB baseline retains
98.79% conditional agreement, but its close/far recall is 68.47%/8.11% and its
balanced agreement is 38.29%. These are different aspects of its performance:
high agreement among answered cases does not establish broad failure coverage.
The expanded baseline report is `rgb-baseline-diagnostic-v2.json`; version 1
remains unchanged. Tests check the absent-class case and replay all controls.

### Completed MOVi judge comparison (development only)

Both frozen arms completed all 944 renderable examples. The saved evaluation
and analysis are `results/movi-camera-pilot-v1/judge-evaluation-v1.json` and
`judge-diagnostic-v1.json`. Exact IDs, paired processed-pixel hashes and token
counts, absent scoring targets, corpus/export hashes, evaluator hash, training
manifest hash, and development manifest hash were checked. An independent
recount matches the saved confusion matrices. The evaluator verifies checkpoint
and adapter hashes before inference. No calibration exports were opened.

Among the 389 diagnostic labels (352 close, 37 far), the base model answers
far throughout: 37 agreements and 50% balanced agreement. The adapted model
agrees on 143/389 (36.76%); close/far recalls are 37.78%/27.03%, averaging
32.41%. Its 151 close predictions include 18 false-close cases. It also predicts
occluded on 65 diagnostically visible cases; these are disagreements, not
abstentions. By comparison, the RGB baseline agrees on 244/247 answered cases
and abstains on 142. On those 142, the adapted model agrees on 59 (41.55%),
below the image-free always-close control's 111/142 (78.17%). Its close
precision on that subset is 51/66 (77.27%).

This comparison does not establish a useful learned fallback or a camera-DNF
metric. Reference visibility is still diagnostic, scenes are developmental,
and repeated frames/methods are dependent. The unsuccessful experiment stays
in the development record, not the manuscript. Do not adjust thresholds on
these results and report them as held-out validation. The next method must
address functional camera failure rather than equating point-match judgments
with camera usability, and must be validated on fresh scenes after development.

### RGB-only focal sensitivity after judge completion

`probe_movi_focal_sensitivity.py` varies both focal axes by fixed factors
0.5, 0.8, 1, 1.25, and 2 while leaving predicted poses unchanged, then refits
each point using only frames 0..9. All 512 scene/method/track slots are retained.
It compares forecasts at frames 12, 18, and 23 without reading reference
arrays. The unchanged 2-pixel prefix tolerance and 8-pixel disagreement
threshold are diagnostic thresholds, not calibrated uncertainty limits.

The numbers of tracks with both original and modified prefix fits within
2 pixels are respectively 185, 213, 226, 214, and 179. Their maximum later
forecast differences are 8.469, 4.809, 0, 4.431, and 6.462 pixels. Only one
query pair exceeds 8 pixels, at focal factor 0.5. All compatible query
pairs have finite positive-depth forecasts; off-image forecasts are retained.
The complete record is `results/movi-camera-pilot-v1/focal-sensitivity-v1.json`.

These are per-track alternatives, not jointly supported whole-scene models.
Tracks may move or be incorrect. This restricted, unoptimized search cannot
certify uniqueness when it finds no disagreement, and does not establish a
dominant ambiguity explanation for the judge's errors. Tests reproduce the
parallel-motion ambiguity, show later forward motion can separate forecasts,
and verify missing-prefix accounting and independence from later observations.

### Correspondence image-input dependence audit

`audit_correspondence_image_dependence.py` uses the first three training-export
rows without selecting by labels or responses. It retains the frozen prompt,
checkpoint, adapter, arithmetic, token layout, and image grid, and compares
the original normalized pixel payload with an all-zero payload. Both arms
generate eight tokens greedily; no training or calibration reads occur.
This artificial intervention tests input sensitivity, not natural-image
accuracy. The record is `results/correspondence-image-dependence-v1/manifest.json`.

All six pairs completed. Base answers remain `far`, but the first-token
probability distributions change: total variation distances are 0.2248,
0.2116, and 0.2337. Adapted answers change from `close` to `far` on all three,
with distances about 0.99967. Real/zero pixel hashes differ, and matched arms
have identical payload hashes and token counts. Thus the input image does
influence this configuration on these examples; constant generated labels
alone do not imply a disconnected image path. This does not establish correct
visual reasoning, isolate the cause of transfer failure, or justify DNF labels.
The controls are previously trained examples and cannot measure generalization.

### Exact-prefix marker-placement ablation

`audit_movi_prefix_tracking.py` refits background markers using exact reference
coordinates for frames 0..9, keeping the saved predicted cameras, intrinsics,
and fitting helper unchanged. Later coordinates are used only for scoring.
All 1,536 original slots are retained, including foreground and unavailable
cases; off-image projections remain measurable. The paired diagnostic requires
reference depth/mask non-rejection throughout the prefix and at the query, plus
finite original and oracle-fit projections. This is a reference-assisted
development ablation, never an input to a reference-free evaluator.

For scenes 3835, 5397, 4385, and 4970, the paired DAGE counts are 97, 83, 112,
and 52; DA3 counts are 97, 81, 112, and 52. RGB-fit/oracle-fit median later
errors in pixels are DAGE 2.208/1.639, 3.429/2.932, 2.489/2.333, 2.455/1.575;
DA3 2.245/1.966, 1.745/1.389, 1.559/1.440, 3.166/2.736. These per-method
subsets are not identical, so these summaries do not establish a paired method
ranking. Differences are not an additive causal decomposition: refitting can
compensate for camera error, and even the median paired improvement can be
negative (DAGE scene 4385: -0.021 pixels).

The result preserves a distinction between camera output, marker initialization,
and later visual checking. It does not validate a new evaluation method or DNF
threshold. The record is `results/movi-camera-pilot-v1/prefix-tracking-ablation-v1.json`.
Tests check exact synthetic recovery, independence from later coordinates,
full-roster accounting, and replay of the saved analysis.

### Matched task-success curves without a selected tolerance

`analyze_movi_task_tolerances.py` evaluates both methods on exactly the same
reference-eligible slots within each task. It retains unavailable placements
in the denominator without calling them camera failures. It computes every
empirical error-curve breakpoint and also reports 1, 2, 4, 8, and 16 pixels;
no tolerance is selected to maximize a method's advantage. These are classical
empirical success curves, not a novel metric or a calibrated DNF rule.

With predicted intrinsics and known-world anchors, DAGE weakly dominates DA3
at every tolerance in three scenes. Scene 4385 crosses: at 2 pixels, DA3 has
46/715 successful placements and DAGE 23/715; at 8 pixels, DAGE has 372/715
and DA3 246/715. Thus DAGE's lower mean error on all four scenes does not mean
it wins at every required precision. With reference intrinsics, scene 5397
instead favors DA3 at every tolerance.

For RGB-initialized surface markers, scene 5397 favors DA3 at every tolerance
on the shared 111-slot roster; the other three scenes have crossing curves.
The oracle-prefix surface-marker curves have the same crossing/noncrossing
pattern. These tasks differ in initialization, support, and calibration, so
their contrast is not an isolated causal effect of one component. It shows
why a usable evaluation must state the task and precision requirement rather
than treat one average as a universal camera ranking. All results remain
four-scene development evidence, not validation of a reference-free evaluator.
The complete paired curves are `results/movi-camera-pilot-v1/task-tolerances-v1.json`.
Tests cover crossings, unavailable denominators, dominance, monotonicity, and
exact replay with source hashes checked.

### Research decision after the completed judge experiments

The literature check on 2026-10-17 adds a relevant comparison:
[PathEval, CVPR 2025](https://arxiv.org/html/2411.18711v4), by Aghzal, Yue,
Plaku, and Yao, separates task interpretation from fine visual perception in
path evaluation. Its experiments motivate task-specific discriminative
adaptation of visual representations. This is adjacent evidence, not a result
about camera tracking or proof of the cause of our errors.

The saved twenty-pass training manifest contains 144 trainable tensors and
1,843,200 parameters, all in language-model q/v attention LoRA layers. There
are no trainable visual tensors. Consequently, the completed experiment tests
language-side adaptation of a frozen visual representation; it does not test
whether training a visual verifier on geometric supervision can work. The
image-zeroing audit shows input dependence, not adequate spatial resolution.

The [3DCodeVerse harness](https://3dcodeverse.com/) combines visual judgments
with deterministic measurements that constrain the judgments, and separates
judge development from frozen harness comparisons. That supports using a
measurement-constrained verifier here, not treating fluent judgment text as
ground truth. [ORE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb206443c93d07da8b1974b768d8a0d4-Abstract-Datasets_and_Benchmarks.html)
already evaluates cameras without ground-truth trajectories using static
object tracklets. Reprojection checks, task-dependent curves, and adding a VLM
are not sufficient novelty claims by themselves.

The next learned-method hypothesis is a visually trained, measurement-constrained
verifier of task failure. Before another language-model training run, test
whether visual features can distinguish controlled marker displacement from
occlusion and unresolvable evidence on development-only data. Compare frozen
features, discriminatively adapted visual features, and the existing RGB
tracker at matched answer coverage. Keep marker initialization, camera output,
static-surface support, and visibility as separate inputs/claims. A verifier
must not turn missing tracks or moving objects into camera failures.

Advancement requires reliable improvements on cases the classical verifier
cannot answer, followed by fresh scene- and method-disjoint functional tests.
Model and threshold selection may use development data only; the already
inspected DAVIS and MOVi examples cannot become fresh validation by renaming
their split. The existing calibration exports remain unopened. A learned
verifier that only reproduces classical decisions does not establish the
requested contribution. BO/RL should wait until this evaluator is validated,
since optimizing an unreliable score would not demonstrate better tracking.

### Paired-crop visual adaptation diagnostic

`train_visual_correspondence_probe.py` completed a small comparison on Anima.
It uses the two native 161x161 bottom crops from the original panels, retaining
their marker rings and omitting context. An ImageNet ResNet18 encodes each
crop; a shared head receives both features, their absolute difference, and
their elementwise product. Frozen-backbone and trainable-layer4 arms start
with the same pretrained weights and seed-0 head. Batch-normalization statistics
remain frozen in both. Both use 20 epochs, Adam 1e-4, batch size 8, training
class weights, and the same per-epoch example orders. There is no augmentation,
early stopping, or best-checkpoint selection. Validation images/labels are
loaded only after both arms finish; calibration exports are never opened.

The 188 training examples are followed by all 128 development validation
examples. The frozen arm gets 60/128 correct (46.88%), with close/far/occluded
recall 63.04%/72.97%/8.89%, balanced accuracy 48.30%. The adapted arm gets
53/128 (41.41%), with recalls 84.78%/37.84%/0%, balanced accuracy 40.87%.
Its final training loss is about 0.0000032, which does not establish
generalization. This setup provides no positive visual-adaptation result.
It is a generic visual-backbone diagnostic, not an isolated test of adapting
Qwen's vision encoder. Neither arm supports DNF judgments, and the two-video
validation set cannot support a broad statistical claim.

The full protocol, epoch losses, checkpoint hashes, and validation predictions
are in `results/visual-correspondence-probe-v1/manifest.json`. Weights remain
under Anima's private `/data3/kmatzen/plumbline/research/visual-correspondence-probe-v1`.
No shared repository or environment was modified. Loader tests reject
calibration access and mismatched targets; artifact tests verify lineage,
complete pairing, disjoint IDs, and prediction counts. This unsuccessful
development experiment is not added to the manuscript. More training on this
same small corpus is not justified by the result; visibility/ambiguity
supervision and evidence representation need to be addressed first.

### Crop evidence support audit

`audit_correspondence_crop_support.py` checks source annotations only for the
six training and two validation videos. It uses the renderer's rounded crop
center and 161x161 native-pixel extent, and does not treat coordinates of
occluded points as visible evidence. Calibration source files and exports
remain unopened. The saved result is
`results/visual-correspondence-probe-v1/crop-support-v1.json`.

Among visible far examples, 29/84 training cases and 22/37 validation cases
place the annotated source detail outside the candidate crop. Thus absence
from that local crop does not distinguish a visible detail elsewhere from
global occlusion. Containment is only a geometric check, not proof that a
detail can be identified visually. This concerns the crop-only ResNet probe;
the original VLM panels also contain reduced full-frame context.

For the 22 outside-crop validation cases, frozen/adapted models correctly say
far on 15/9. For the 15 inside-crop cases, they do so on 12/5. Hence missing
crop context is not a complete explanation for the adapted model's errors.
The next representation must separate local correspondence from full-frame
visibility, with temporal evidence or abstention when visibility is unresolved.
Do not relabel outside-crop visible points as occluded or claim that these
counts validate a new metric.

### Frozen temporal correspondence and visibility baseline

`probe_tapvid_cotracker_verifier.py` runs the already pinned CoTracker3 offline
predictor on all 316 labeled training/validation cases. Queries are the same
supplied initial points, grouped by video and source frame. Each RGB window
ends at its last requested observation and spans at most 61 frames. Checkpoint
and all predictor-source hashes match the earlier retained CoTracker run.
No new training, threshold selection, camera outputs, calibration source files,
or calibration exports are involved. Source annotations beyond the supplied
queries do not enter tracking; targets are attached only for scoring.

Upstream visibility false produces an `occluded` prediction. Otherwise, distance
to the original candidate determines close <=2 pixels, far >=8 pixels, or
uncertain between them. All 316 cases completed. Training agreement is 167/188.
Validation agreement is 99/128, with 12 uncertain answers. The validation
confusion is: close -> 36 close and 10 uncertain; far -> 21 far, 14 occluded,
and 2 uncertain; occluded -> 42 occluded and 3 far. Thus temporal evidence is
useful in this small comparison, but a tracker visibility rejection is not
proof of occlusion. No validation close prediction is false in this sample;
this is not a calibrated precision guarantee or camera-success certificate.

This is an existing-method baseline, not our novel contribution. Windows,
inputs, and pretraining differ from the visual probes, so this is not an
isolated temporal-context ablation. These videos have already been inspected;
pretraining overlap has not been excluded, and no fresh-test claim is made.
Future evaluator comparisons must include this stronger temporal baseline and
report its visibility errors, coverage, and task limitations. The complete
record is `results/tapvid-cotracker-verifier-v1/manifest.json`; an artifact test
checks exact membership, provenance, decisions, window limits, and counts.

### Matched temporal verifier on MOVi camera markers

`probe_movi_temporal_verifier.py` runs the same frozen CoTracker3 checkpoint on
all four MOVi videos, with all 64 original fixed frame-zero queries. The tracker
sees only the 24 RGB frames and these query coordinates, not camera predictions,
reference geometry, or diagnostic labels. The saved camera proposals are compared
with its tracks afterwards. All 1,536 slots remain; unavailable proposals do
not become camera failures. `analyze_movi_temporal_verifier.py` replays the exact
389 diagnostic labels and source hashes from the earlier VLM analysis.

The temporal verifier agrees on 366/389 (94.09%), with 17 uncertain answers.
It agrees on 366/372 decisive answers (98.39%), at 95.63% coverage. Close/far
recall is 337/352 and 29/37, giving 87.06% balanced agreement. Six diagnostically
visible cases are predicted occluded; none of the 337 close predictions is
false under these reference cues. On the 142 RGB-baseline-abstained labeled
cases, it agrees on 121, predicts occluded incorrectly on six, and abstains on
15. These are diagnostic agreements, not calibrated guarantees or independent
visibility truth. Four developmental scenes cannot establish generalization.

The complete inference and diagnostic records are under
`results/movi-temporal-verifier-v1`. An artifact test verifies script lineage,
complete paired membership, original decision thresholds, and exact replay.
This existing-method result is substantially stronger than the current learned
panel judge on the same correspondence task. Further correspondence-only
training is not justified as the main contribution without evidence of an
advantage over this baseline. The remaining research target is task-level
camera failure with staticness and evidence sufficiency handled explicitly;
accurate tracking of a moving object is not evidence of a correct camera.

### Prefix-only temporal marker initialization

`build_movi_temporal_markers.py` replaces the early LK observations with an
independent CoTracker run on frames 0..9 only. The fixed 64 source pixels,
predicted cameras/intrinsics, fitting helper, and query frames are unchanged.
Visibility-false prefix observations remain missing. All 1,536 original IDs
are retained for pairing, but these are new proposals and must never be joined
to old VLM answers as if their image inputs were unchanged.

The alternative produces 1,143 renderable proposals versus the original 944.
Per-scene renderable counts are 302, 318, 324, and 199. This does not establish
better accuracy or coverage of the needed motion cases. Repeating the separate
reference-motion audit shows that of the same 26 supported foreground cases
with at least 8 pixels of counterfactual static displacement, only eight are
renderable, unchanged from before. Twelve project outside the image, four have
nonpositive fitted depth, and two lack prefix observations. Thus the coverage
increase does not solve this pilot's weak support for the motion-attribution
test. Off-image and fitting failures remain explicit, not camera-DNF labels.

All proposals and the new motion audit are under `results/movi-temporal-markers-v1`.
The replay test checks all 1,536 projections against local fitting code and
camera arrays, source hashes, original IDs, and the ten-frame observation limit.
Anima uses the existing private geometry helper via explicit import; the
shared repository is unchanged. This is a development harness comparison,
not a claimed contribution or a manuscript experiment.

### Camera-independent rigid-motion support baseline

`probe_movi_rigid_support.py` fits a fundamental matrix to temporal tracker
correspondences without using either evaluated camera method. It uses all
64 fixed points at frames 12, 18, and 23, gated by predicted visibility;
OpenCV RANSAC uses 1 pixel, confidence 0.99, 2,000 iterations, and seed 0.
Finite Sampson distances <=1 pixel are compatible; missing support stays
unresolved. Duplicate method observations are checked for equality and removed,
leaving 768 scene/frame/point cases. Reference motion and visibility cues enter
only after fitting, for a separate diagnostic comparison.

Among 513 depth/mask-nonrejected static background cases, 489 are compatible,
nine incompatible, and 15 unresolved. Among 13 supported foreground cases
with at least 8 pixels of counterfactual static displacement, eight are
incompatible and five unresolved. The other 242 reference cases are retained
as unresolved ground truth, not scored as static or moving. This is a small,
developmental check of an existing geometric baseline, not a new motion
detector or a claim of generalization.

An incompatible observation is not proof of object motion: tracking errors
and degenerate geometry remain alternatives. Conversely, epipolar compatibility
does not certify a static surface. A camera evaluator must preserve these
distinctions rather than silently drop all outliers or call them camera errors.
The record is `results/movi-temporal-verifier-v1/rigid-support-v1.json`; tests
check distance arithmetic, insufficient support, deduplication, and exact replay.

### Held-out-point check of rigid support

`probe_movi_crossfit_support.py` divides the fixed 8x8 source grid into four
interleaved groups using row/column parity. Each group's judgments use a matrix
fit only to visible points in the other three groups. RANSAC settings and the
1-pixel Sampson threshold are unchanged. This removes direct own-point fitting,
not dependence between tracks or scenes, and is not independent validation.

The held-out check changes 25/768 decisions. Among the 513 supported static
background cases, 485 pass, 13 fail, and 15 remain unresolved. Among the 13
supported moving foreground cases, seven fail, one passes, and five remain
unresolved. The in-sample check had nine static rejections and eight moving
rejections. Neither version certifies staticness or justifies treating all
rejected points as camera failures. The record is
`results/movi-temporal-verifier-v1/crossfit-support-v1.json`. A direct intervention
test changes a held-out group's coordinates and verifies that its fitted
matrix remains unchanged; all fitting and held-out slot lists are retained.

### Joint placement and motion evidence

`audit_movi_joint_evidence.py` joins the frozen temporal placement check to the
held-out-point epipolar support check, using their unchanged thresholds. The
ungated arm retains close/far answers; the gated arm additionally requires
epipolar compatibility. Occluded, unavailable, and uncertain answers remain
unresolved in both arms. Neither arm assigns camera DNF. Reference labels are
joined after the decisions, and all 1,536 method-point-frame slots remain.

The ungated arm has 402 close, 44 far, and 1,090 unresolved answers. The gated
arm has 390 close, 39 far, and 1,107 unresolved answers. All 334 determinate
answers on reference-supported static backgrounds survive the gate (305 close,
29 far). Among static-background diagnostic labels, 290/302 close cases and
28/29 far cases are correctly answered; the remaining 13 abstain. Borderline
and unresolved references are reported separately, not counted as correct.

Only one of the 26 method-point-frame slots on supported moving foregrounds
has a determinate placement answer: an ungated far warning. The gate abstains
on it. This is too little motion coverage to establish a useful attribution
advantage. The other 16 removed answers have unresolved motion references;
their removal cannot be scored as an improvement. The four development scenes,
repeated frames, and two methods are not independent samples.

This joint audit identifies the next experimental requirement: obtain
renderable moving-surface placements as well as static-surface placements,
without choosing examples based on which evaluator wins. Correspondence
agreement alone cannot validate camera failure attribution. The complete
record is `results/movi-temporal-verifier-v1/joint-evidence-v1.json`, including
per-scene/per-method accounting and every decision. Replay tests check the
full roster, exact retained output, and missing-evidence behavior. These are
development findings, not new manuscript claims.

### Correct-camera motion control with oracle observations

`audit_movi_oracle_motion_control.py` removes point-fitting and image-matching
errors from the placement arm. It uses correct cameras, calibration, initial
world points, and surface correspondences. For foregrounds, the retained
reference audit supplies the discrepancy between a source-frozen point and
the moving surface under the same correct camera. Static-background discrepancy
is zero by definition, not a new numerical projection test. Only the motion
gate still uses predicted image tracks. This is an oracle control, not an
implementable reference-free metric, and changes initialization and matching
together rather than isolating either effect.

The complete roster contains 768 unique point/frame slots, without counting
the two original methods twice: 513 supported static-background observations,
60 supported foreground observations, and 195 unresolved references. With the
unchanged 2/8-pixel thresholds, foregrounds yield 43 close, 13 far, and four
borderline observations despite correct cameras. The crossfit motion gate
retains 40 close and one far foreground observation, abstaining on the other
19. It retains 485/513 static-background observations and abstains on 28.
Unresolved references remain unresolved throughout.

The surviving far observation is scene 4970, frame 12, source-grid slot 25:
178.425 pixels of reference motion discrepancy but 0.769 pixels of held-out
Sampson distance. This is a concrete counterexample to treating epipolar
compatibility as a static-surface certificate. Removing 12/13 motion warnings
in this small development control is not proof of camera-failure detection:
the surviving warning is unsupported as a camera accusation, static coverage
is lost, visibility references remain diagnostic, and no independent scene
validation has occurred. The artifact is
`results/movi-temporal-verifier-v1/oracle-motion-control-v1.json`; tests verify
reference deduplication, exact replay, and full accounting. It remains outside
the manuscript.

### Frozen-matrix correspondence intervention

`audit_movi_motion_track_error.py` tests whether correspondence error explains
the accepted moving point in the oracle control. It holds each crossfit matrix,
source coordinate, and 1-pixel threshold fixed, replacing only the held-out
target coordinate with its reference projection. All 768 unique planned
point/frame slots are retained. There are 573 supported reference projections
and 195 unresolved references. Predictor-invisible cases are reported separately
and are not treated as deployed motion-gate answers.

For the accepted moving point (4970/frame 12/slot 25), the predicted coordinate
is 8.207 pixels from the reference projection. Its Sampson residual changes
from 0.769 to 4.785 pixels, changing compatibility to incompatibility under the
same matrix. Thus the observed pass is not evidence that this exact surface
motion lies within the fitted epipolar tolerance: the predicted correspondence
causes the pass. This does not establish that the fitted matrix is correct.

Among the eight predictor-visible supported moving observations, seven remain
incompatible and this one changes to incompatible. Among 498 predictor-visible
static-background observations, 476 remain compatible, five remain incompatible,
eight change from incompatible to compatible, and nine change in the opposite
direction. Exact held-out coordinates therefore do not make the fitted matrices
a staticness certificate: errors in their other input tracks, geometric
degeneracy, and reference uncertainty remain. The intervention does not refit
matrices with ground truth or claim an improved deployed evaluator.

The retained record is `results/movi-temporal-verifier-v1/motion-track-error-v1.json`.
Tests check fixed-matrix residual arithmetic, full visibility accounting, and
exact replay. The development implication is to evaluate correspondence
reliability and fitted-geometry uncertainty separately before assigning camera
failure; this result is not added to the manuscript.
