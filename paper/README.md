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
No fine-tuning or RL has begun, and the currently available labels are
insufficient for a defensible training experiment.

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
records are staged for the next manuscript revision; the current PDF does not
yet contain these two citations.

[3D-DefectBench](https://arxiv.org/abs/2607.10826v2) studies how the model,
rendering, evidence, and prompt affect human agreement for 3D-asset defects;
[Spotlight](https://spotlight-video.github.io/) studies localized error detection
in generated video. Neither establishes the proposed camera criterion. Merely
using a VLM for failure detection is not a sufficient novelty claim. Full
related-work coverage and actual camera experiments remain outstanding.

### Fresh dense development runs

#### RGB-D evidence validation (in preparation)

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
RGB-reference timing offsets must both be reported. No real-data result from
this audit is available yet, and it is not included in the paper's claims.

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

After staging, the development command is:

```sh
uv run python -m paper.rgbd_track_audit \
  paper/results/dense/dage-walking-xyz-v2 \
  paper/results/dense/tum-anchor-xyz-v1.json \
  /path/to/staged-depth \
  /path/to/new-audit.json
```

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
