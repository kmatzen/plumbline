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

## Human diagnostic-review pilot

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
