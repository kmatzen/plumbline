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
[3D-DefectBench](https://arxiv.org/abs/2607.10826v2) studies how the model,
rendering, evidence, and prompt affect human agreement for 3D-asset defects;
[Spotlight](https://spotlight-video.github.io/) studies localized error detection
in generated video. Neither establishes the proposed camera criterion. Merely
using a VLM for failure detection is not a sufficient novelty claim. Full
related-work coverage and actual camera experiments remain outstanding.

### Fresh dense development runs

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

Chen et al., [Object Reprojection Error (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/file/eb206443c93d07da8b1974b768d8a0d4-Paper-Datasets_and_Benchmarks.pdf)
already evaluate trajectories without reference camera poses. This is a required
comparison, not a novelty claim for this project.

`src/plumbline/object_reprojection.py` is a development ORE-style kernel, not a
verified reproduction. It uses a fixed first-box center, a supplied finite depth
grid, normalized L1 distance outside boxes, and rejects depth hypotheses behind
any observed camera. It reports depth-boundary solutions and unavailable cases.
Synthetic tests check projection direction, depth recovery, coordinate/scale
changes, and input rejection; they do not validate real-data ranking quality.
The upstream implementation and its sampling, depth range, visibility, and
aggregation rules must be checked before reporting published ORE reproduction.

DAVIS segmentation objects cannot be assumed static. Intrinsics are also absent
from our DAVIS inventory. Do not use arbitrary focal lengths or moving masks as
ORE ground truth. An annotated static-track subset and an explicit calibration
policy are needed. Predicted calibration would define an estimated-intrinsics
variant and must be reported separately from known-intrinsics evaluation.
