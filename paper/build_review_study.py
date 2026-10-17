"""Build a blinded diagnostic-review pilot from the real camera-pose export.

Only serve the generated public/ directory. Private identity mappings and the
artifact manifests are kept beside it. This study asks about reference-based
diagnostics; ten sparse views cannot support a dense-video perceptual claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np

from plumbline.tracking_review import build_review_pair, write_tracking_review_artifact

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Camera diagnostic review</title>
<style>
body{font:17px system-ui,sans-serif;margin:24px auto;max-width:1500px;padding:0 20px;color:#18222d;background:#faf9f5}
h1{font-size:26px}button,input{font:inherit;padding:10px;margin:5px;border:1px solid #8795a3;border-radius:5px}
button{background:white;cursor:pointer}button.chosen{background:#cde6f3;border:2px solid #07597e}
img{width:100%;display:block;border:1px solid #d7dadd;margin:16px 0}p{max-width:1000px;line-height:1.5}
#status{font-weight:600}#actions{display:flex;flex-wrap:wrap}small{display:block;color:#4e5c68}
</style>
<h1>Camera diagnostic review</h1>
<p>Choose the candidate whose camera estimates more closely follow the blue reference path
and reference orientations. Consider path shape, large local errors, and rotation errors.
Both candidates use the same reference scales. These are ten sampled views, so do not
infer video smoothness or speed from the line segments.</p>
<p>Choose “Tie” when the visible quality is equivalent, “Both poor” when neither result is useful,
or “Cannot judge” when the evidence is insufficient. You can stop and export a partial session.
Responses stay in this browser until you download them. Use an anonymous reviewer code.</p>
<label>Reviewer code <input id="reviewer" autocomplete="off" placeholder="e.g. reviewer-01"></label>
<button id="download">Download responses</button><div id="status"></div>
<img id="pair" alt="Candidate A and Candidate B camera diagnostic plots">
<div id="actions"></div><button id="back">Previous</button><button id="next">Next</button>
<small>Study: __STUDY_ID__. Pilot diagnostic judgments; no method names or leaderboard scores are shown.</small>
<script>
const trials=__TRIALS__, studyId='__STUDY_ID__', storageKey='camera-study-'+studyId;
let state=JSON.parse(localStorage.getItem(storageKey)||'null')||{study_id:studyId,reviewer:'',index:0,responses:{}};
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer;
function save(){state.reviewer=reviewer.value.trim();localStorage.setItem(storageKey,JSON.stringify(state));}
function show(){const t=trials[state.index];document.getElementById('pair').src=t.image;
document.getElementById('status').textContent=`Comparison ${state.index+1} of ${trials.length}; ${Object.keys(state.responses).length} answered`;
const actions=document.getElementById('actions');actions.replaceChildren();
for(const [value,label] of [['a','A is better'],['b','B is better'],['tie','Tie'],['both_bad','Both poor'],['abstain','Cannot judge']]){
const b=document.createElement('button');b.textContent=label;
if(state.responses[t.id]?.winner===value)b.className='chosen';
b.onclick=()=>{if(!reviewer.value.trim()){reviewer.focus();return;}
state.responses[t.id]={presentation_id:t.id,winner:value,recorded_at:new Date().toISOString()};save();show();};actions.append(b);}
document.getElementById('back').disabled=state.index===0;document.getElementById('next').disabled=state.index===trials.length-1;}
reviewer.onchange=save;
document.getElementById('back').onclick=()=>{state.index=Math.max(0,state.index-1);save();show();};
document.getElementById('next').onclick=()=>{state.index=Math.min(trials.length-1,state.index+1);save();show();};
document.getElementById('download').onclick=()=>{save();if(!state.reviewer){reviewer.focus();return;}
const result={schema_version:'1',study_id:studyId,reviewer:state.reviewer,responses:Object.values(state.responses)};
const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
const a=document.createElement('a');a.href=url;a.download='camera-review-'+studyId+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
show();
</script></html>
"""


def build(source: Path, output: Path, seed: int = 441) -> dict:
    if output.exists():
        raise ValueError("choose a new output directory to preserve existing reviewer sessions")
    data = json.loads(source.read_text())
    study_id = hashlib.sha256(source.read_bytes() + str(seed).encode()).hexdigest()[:16]
    public = output / "public"
    public.mkdir(parents=True)
    private = output / "private"
    private.mkdir()
    artifacts = {}
    tasks = []
    for variant in data["variants"]:
        if variant["sigma"] != 0:
            continue
        sample_id = variant["sample_id"]
        reference = np.asarray(variant["reference"])
        for method, prediction in variant["predictions"].items():
            artifacts[sample_id, method] = write_tracking_review_artifact(
                np.asarray(prediction["poses"]),
                reference,
                private / "artifacts",
                dataset="realestate-pilot",
                sequence=sample_id,
                system=method,
                config_hash=prediction["pose_sha256"][:16],
            )
        tasks.extend(
            (sample_id, a, b, False) for a, b in combinations(sorted(variant["predictions"]), 2)
        )
    rng = random.Random(seed)
    # Repeats are reliability checks and must not count as additional independent votes.
    repeats = rng.sample(tasks, min(15, len(tasks)))
    tasks.extend((sample, a, b, True) for sample, a, b, _ in repeats)
    rng.shuffle(tasks)
    trials, mapping = [], []
    for index, (sample, a, b, repeat) in enumerate(tasks):
        pid = hashlib.sha256(f"{study_id}:{index}".encode()).hexdigest()[:20]
        pair_path = build_review_pair(
            artifacts[sample, a],
            artifacts[sample, b],
            private / "pairs" / pid,
            task_id=sample,
            comparison_id=pid,
        )
        pair = json.loads(pair_path.read_text())
        if pair["automatic_winner"] is not None:
            raise ValueError("study construction encountered a gated pair")
        shutil.copyfile(pair_path.parent / "review.png", public / f"{pid}.png")
        trials.append({"id": pid, "image": f"{pid}.png"})
        mapping.append(
            {
                "presentation_id": pid,
                "sample_id": sample,
                "system_a": pair["system_a"],
                "system_b": pair["system_b"],
                "repeat": repeat,
                "pair_key": f"{sample}:{a}:{b}",
            }
        )
    html = PAGE.replace("__STUDY_ID__", study_id).replace("__TRIALS__", json.dumps(trials))
    (public / "index.html").write_text(html)
    manifest = {
        "study_id": study_id,
        "seed": seed,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "unique_pairs": len(tasks) - len(repeats),
        "repeat_pairs": len(repeats),
        "presentations": mapping,
        "scope": "Reference-based diagnostic judgments on sparse views, not video preference.",
    }
    (private / "study.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "study_id": study_id,
        "public": str(public),
        "unique_pairs": manifest["unique_pairs"],
        "repeat_pairs": len(repeats),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "results" / "realestate_pose_export.json",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
