# Camera-tracking review harness

Status: artifact generation, validity gates, blinded pairs, human and external
judge votes, calibration, ranking, and Bayesian proposals are implemented in
`plumbline.tracking_review`, `plumbline.design_search`, and
`plumbline tracking-review`.

## Decision

Do **not** run Plumbline camera trackers through 3DCodeBench as if they were
procedural object generators. Its task contract is Blender source → GLB → orbit
renders, while Plumbline produces camera trajectories, depth, and point maps.
The adapters would be mostly glue and the resulting scores would measure object
rendering rather than tracking.

Reuse the harness pattern instead:

- independent, resumable scorers over a stable result tree;
- deterministic, blinded A/B presentation order;
- reference + two candidate artifacts in one judgment;
- explicit validity gates before expensive judging;
- balanced round-robin pair selection;
- Bradley–Terry/Elo-style ratings with task-level bootstrap intervals;
- calibration against human preferences before treating a VLM judge as a
  measurement instrument.

These patterns come from the Apache-2.0
[3DCodeBench](https://github.com/gaoypeng/3dcodebench) judge harness and the
published design of
[4DCodeBench](https://github.com/4DCodeBench/4DCodeBench). 4DCodeBench currently
has no repository license file, so Plumbline must not copy or vendor its code;
the implementation here is independent.

## Why metrics alone are insufficient

ATE, RPE, pairwise rotation/translation accuracy, and AUC compress different
failure modes into means. Two trackers with similar aggregates can differ in:

- slow global drift versus one catastrophic relocalization;
- rotational jitter versus translational scale error;
- loop closure and return-to-origin behavior;
- short dropouts, frozen poses, or discontinuities;
- performance on static, dynamic, low-texture, and fast-motion segments;
- latency, memory, and usable-track coverage.

There is no honest universal scalarization of those preferences. The harness
therefore keeps the metric vector, exposes Pareto tradeoffs, and adds pairwise
human/VLM preference as a separate signal rather than declaring the judge to be
ground truth.

## Stable artifact contract

Each `(dataset, sequence, system, config_hash)` produces:

```text
tracking-review/<dataset>/<sequence>/<system>/<config_hash>/
├── manifest.json
├── poses.npz
├── metrics.json
├── diagnostic.png
└── diagnostic.mp4          # when RGB frames are available
```

`manifest.json` records source report/cache fingerprints, frame indices,
coordinate convention, alignment, renderer version, and failure/gate status.
`poses.npz` stores unaligned and evaluation-aligned GT/predicted
`world_from_camera` trajectories so a rendering change never requires model
inference again.

The diagnostic image/video must use a fixed layout and scales derived from GT,
never from the candidate:

1. input RGB with frame number and failure markers;
2. top, side, and perspective trajectory views (GT plus one candidate);
3. translation, rotation, step-motion, and jerk errors over time;
4. loop-closure and accumulated-drift indicators;
5. a reprojection/stabilization panel when depth or a point map is available;
6. runtime, peak memory, coverage, and existing scalar metrics in a footer.

Candidate names are replaced by deterministic opaque labels during review.
Identical frame sampling, plot bounds, colors, and encoding are mandatory.

## Gates

Judge calls are skipped when an artifact is invalid. Gates are deterministic
and reported separately:

- missing or non-finite poses;
- wrong frame count or timestamps;
- invalid SE(3) transforms;
- insufficient tracked-frame coverage;
- static/frozen trajectory on a moving-GT segment;
- missing diagnostic frames or decode failure.

One failed candidate loses to one valid candidate; two failed candidates are a
tie tagged `both_invalid`. This avoids paying a VLM to discover mechanical
failures and keeps failure rate visible instead of hiding it in preference.

## Review and ranking protocol

For each sequence, select the least-compared system pairs. Randomize A/B order
deterministically from the comparison ID, and record the swap. A reviewer sees
GT context, candidate A, candidate B, and the fixed rubric:

1. path geometry and endpoint/loop consistency;
2. local motion fidelity and absence of jitter;
3. recovery from occlusion or fast motion;
4. coverage and catastrophic failures;
5. resource cost only as a separately labeled tie-breaker.

Allowed verdicts are A, B, tie, and both bad. `both bad` is retained for
diagnostics but is not silently converted into a win. Ratings use a
Bradley–Terry model; ties are half a win each. Confidence intervals resample
whole sequences, not votes, because comparisons sharing a sequence are
correlated. `plumbline.tracking_review` already implements this statistical
core and rejects disconnected comparison graphs.

Before a VLM judge is used at scale, collect a stratified human calibration set
covering datasets and failure modes. Report judge coverage, accuracy, confusion
matrix, and Cohen's kappa on decisive human labels. If calibration is weak, the
VLM remains a triage tool and human ratings remain authoritative.

### Current CLI

```bash
# Emit one artifact per evaluated sequence directly from a normal pose run.
plumbline run --model vggt --dataset sintel --tasks pose \
  --tracking-review-dir review

# Canonical world_from_camera arrays, .npy or keyed .npz.
plumbline tracking-review render \
  --predicted pred.npy --reference gt.npy --output-root review \
  --dataset tum-dynamics --sequence walking_xyz \
  --system cut3r --config-hash "$CONFIG_HASH"

# Deterministically randomizes which system is displayed as A/B.
plumbline tracking-review pair \
  --artifact-a review/.../cut3r/... --artifact-b review/.../dage/... \
  --task-id tum-dynamics/walking_xyz --comparison-id tum-walk-001 \
  --output pairs/tum-walk-001

# Open review.png, then append one verdict. Gated pairs state their required
# automatic verdict in pair.json and do not produce a review image.
plumbline tracking-review vote \
  --pair pairs/tum-walk-001/pair.json --winner a \
  --comparisons comparisons.jsonl --reviewer human-1

plumbline tracking-review rank \
  --comparisons comparisons.jsonl --bootstrap 1000 --output ratings.json

# The adapter command reads one JSON request from stdin and writes one JSON
# response to stdout. It may call a local model or a hosted service.
plumbline tracking-review judge \
  --pair pairs/tum-walk-001/pair.json \
  --command "python my_vlm_adapter.py" \
  --comparisons vlm.jsonl --output judge-result.json

# A judge is not accepted until it agrees with a separate human log.
plumbline tracking-review calibrate \
  --human human.jsonl --judge vlm.jsonl --output calibration.json
```

The judge request contains the absolute path to `review.png`, the comparison
ID, the task ID, allowed verdicts, and the fixed rubric. The adapter returns:

```json
{"winner": "a", "reason": "lower drift and less jitter", "provider": "local-vlm"}
```

Plumbline runs the command without a shell, applies a timeout, validates the
response, and records the vote in the append-only log. API keys and provider
libraries remain in the adapter.

## Design search

### Bayesian optimization

Camera trackers expose expensive mixed parameters (resolution, window length,
confidence thresholds, global-alignment iterations, precision, and keyframe
policy). Constrained multi-objective Bayesian optimization is a good fit:

- objectives: preference/rating, ATE/RPE/AUC vector, coverage, latency, VRAM;
- hard constraints: no gate failure, minimum coverage, maximum memory/runtime;
- fidelity levels: short/cheap sequence subset → full benchmark;
- observations keyed by the existing `config_hash` and prediction cache;
- seeded random scalarizations across objectives, so repeated proposals sample
  different parts of the Pareto surface;
- Gaussian-process feasibility estimates for declared coverage, memory, or
  runtime constraints.

The optimizer must hold out sequences when reporting the final winner. Using
the evaluation set for both search and selection would merely automate
benchmark overfitting.

The search specification is plain JSON:

```json
{
  "parameters": {
    "precision": {"type": "choice", "values": ["fp16", "fp32"]},
    "window": {"type": "int", "low": 2, "high": 16},
    "confidence": {"type": "float", "low": 0.001, "high": 0.5, "log": true}
  },
  "objectives": [
    {"name": "preference", "direction": "maximize"},
    {"name": "latency_ms", "direction": "minimize"}
  ],
  "constraints": [
    {"name": "coverage", "op": ">=", "value": 0.95},
    {"name": "peak_vram_gb", "op": "<=", "value": 10.5}
  ]
}
```

Each line in `observations.jsonl` contains `parameters`, `metrics`, and an
optional `status`. Failed runs may use `"status": "failed"`. Propose the next
configuration with:

```bash
plumbline tracking-review propose \
  --spec tracking-search.json \
  --observations observations.jsonl \
  --seed 4 --output next-config.json
```

The first proposals use space-filling search. After three successful
observations, the command uses Gaussian-process expected improvement and
constraint probabilities. The result includes a stable `config_hash` for the
prediction cache.

### RL: not the first tool

RL is unjustified for choosing a static adapter configuration: Bayesian
optimization or a preference-aware bandit uses expensive evaluations far more
efficiently. RL becomes relevant only if the policy makes sequential decisions
inside a run—keyframe selection, window reset, relocalization, or adaptive
compute allocation. That would require a simulator or a large offline log with
well-defined state/action/reward and an untouched test set. Until then, calling
the search “RL” adds complexity without a defensible advantage.

## Delivery sequence

1. ✅ Implement deterministic trajectory diagnostic PNGs and manifests from
   canonical prediction/GT pose arrays, including direct emission from normal
   `plumbline run` pose evaluations.
2. ✅ Add validity gates and deterministically blinded pair generation.
3. ✅ Add a local human-review CLI that writes append-only JSONL verdicts.
4. ✅ Fit ratings and task-bootstrap intervals using `tracking_review`.
5. ✅ Add a provider-neutral VLM command contract and calibration report. A
   specific judge still needs human labels before its results can be trusted.
6. ✅ Add constrained multi-objective Bayesian proposals over declared mixed
   parameters.

No GPU is needed for review and proposal generation. `anima`'s 1080 Ti is sufficient for cached
tracker inference and diagnostic generation; a larger rented GPU is justified
only for methods that cannot fit or for high-throughput search after the
evaluation instrument is validated.
