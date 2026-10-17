"""Small blinded usability/evidence-sufficiency pilot; serve public/ only.

No ground-truth usability labels exist. Project-author reviews are developmental
and must not be reported as independent final validation. Playback speed is an
explicit presentation choice, not an inferred capture rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

from paper.build_reference_free_review import build as render
from paper.build_reference_free_review import digest, load_verified

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Camera usability pilot</title>
<style>body{font:17px system-ui;max-width:1300px;margin:24px auto;padding:0 20px;background:#faf9f5;color:#17232b}
p{max-width:1000px;line-height:1.5}video{width:min(100%,854px);display:block}img{width:100%;display:block}
button,select,input,textarea{font:inherit;padding:10px;margin:6px 0}textarea{display:block;width:95%}
fieldset{margin:20px 0}#status{font-weight:600}.note{color:#6b3417}button{cursor:pointer}</style>
<h1>Camera usability pilot</h1>
<p>Judge whether the shown camera estimate could support stable placement of static 3D content
in this clip. Look for a mismatch between visible camera movement and estimated movement.
Moving people or objects need not follow the camera. A jagged path alone does not prove failure.
There are no reference camera poses, and predicted translation units are not independently calibrated.</p>
<p class="note">This is a test of the evidence as well as the estimate. Choose “Not enough evidence”
unless you can justify a usability judgment from what is shown. The video plays at 10 displayed
frames per second for inspection only; capture timing is unknown. Do not infer speed errors.
The trajectory plots show all supplied frames; the image strip shows six samples.</p>
<p>Use an anonymous reviewer code. Answers stay in this browser until you download them.
Partial sessions are supported. Usability is not a preference between methods.</p>
<label>Reviewer code <input id="reviewer" autocomplete="off"></label>
<button id="download">Download responses</button><p id="status"></p>
<video id="source" controls muted loop playsinline></video>
<img id="diagnostic" alt="Source frames and estimated camera motion without reference poses">
<fieldset><legend>Your assessment</legend>
<select id="verdict"><option value="">Choose an answer</option><option value="usable">Usable for the stated task</option>
<option value="unusable">Unusable for the stated task</option><option value="abstain">Not enough evidence</option></select>
<label>Relevant frame indices, if any <input id="frames" placeholder="e.g. 20–35"></label>
<textarea id="reason" rows="3" placeholder="Describe the visible evidence, or what is missing."></textarea>
<button id="save">Save answer</button><span id="message" role="status"></span></fieldset>
<button id="back">Previous</button><button id="next">Next</button>
<script>
const trials=__TRIALS__, study='__STUDY__', key='davis-review-'+study;
let state=JSON.parse(localStorage.getItem(key)||'null')||{reviewer:'',index:0,responses:{}};
const el=id=>document.getElementById(id);el('reviewer').value=state.reviewer;
function persist(){state.reviewer=el('reviewer').value.trim();localStorage.setItem(key,JSON.stringify(state));}
function show(){const t=trials[state.index],r=state.responses[t.id]||{};
el('source').src=t.video;el('diagnostic').src=t.image;el('verdict').value=r.verdict||'';
el('reason').value=r.reason||'';el('frames').value=r.frame_indices_text||'';
el('status').textContent=`Presentation ${state.index+1} of ${trials.length}; ${Object.keys(state.responses).length} saved`;
el('back').disabled=state.index===0;el('next').disabled=state.index===trials.length-1;el('message').textContent='';}
el('save').onclick=()=>{if(!el('reviewer').value.trim()||!el('verdict').value||!el('reason').value.trim()){
el('message').textContent='Enter a reviewer code, an answer, and a reason.';return;}
const t=trials[state.index];state.responses[t.id]={presentation_id:t.id,verdict:el('verdict').value,
reason:el('reason').value.trim(),frame_indices_text:el('frames').value.trim(),recorded_at:new Date().toISOString()};
persist();el('message').textContent='Saved.';el('status').textContent=`Presentation ${state.index+1} of ${trials.length}; ${Object.keys(state.responses).length} saved`;};
el('back').onclick=()=>{state.index--;persist();show();};el('next').onclick=()=>{state.index++;persist();show();};
el('reviewer').onchange=persist;
el('download').onclick=()=>{persist();if(!state.reviewer){el('reviewer').focus();return;}
const data={schema:'davis-usability-pilot-v1',study_id:study,reviewer:state.reviewer,responses:Object.values(state.responses)};
const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
const a=document.createElement('a');a.href=url;a.download='davis-review-'+study+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};show();
</script></html>"""


def build(runs: list[Path], root: Path, output: Path, seed: int = 914):
    if output.exists():
        raise ValueError("preserve previous reviewer sessions")
    hashes = [digest(p / "manifest.json") for p in runs]
    if len(hashes) != len(set(hashes)) or len(runs) < 2:
        raise ValueError("need distinct run manifests")
    study = hashlib.sha256(
        (str(seed) + ":".join(hashes) + digest(Path(__file__))).encode()
    ).hexdigest()[:20]
    public, private = output / "public", output / "private"
    public.mkdir(parents=True)
    private.mkdir()
    assets = []
    for index, run in enumerate(runs):
        manifest, _, paths = load_verified(run, root)
        asset = hashlib.sha256(f"{study}:asset:{index}".encode()).hexdigest()[:20]
        artifact = private / asset
        render(run, root, artifact)
        shutil.copyfile(artifact / "review.png", public / f"{asset}.png")
        frames = artifact / "source"
        frames.mkdir()
        for frame_index, path in enumerate(paths):
            shutil.copyfile(path, frames / f"{frame_index:06d}.jpg")
        command = [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(frames / "%06d.jpg"),
            "-frames:v",
            str(len(paths)),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(public / f"{asset}.mp4"),
        ]
        subprocess.run(command, check=True)
        assets.append(
            {
                "asset_id": asset,
                "run": str(run),
                "sequence": manifest["sequence"],
                "manifest_sha256": hashes[index],
                "image_sha256": digest(public / f"{asset}.png"),
                "video_sha256": digest(public / f"{asset}.mp4"),
                "ffmpeg_command": command,
            }
        )
    rng = random.Random(seed)
    order = [(i, False) for i in range(len(runs))]
    order += [(i, True) for i in rng.sample(range(len(runs)), 2)]
    rng.shuffle(order)
    trials, mapping = [], []
    for position, (index, repeat) in enumerate(order):
        pid = hashlib.sha256(f"{study}:presentation:{position}".encode()).hexdigest()[:20]
        asset = assets[index]["asset_id"]
        trials.append({"id": pid, "image": f"{asset}.png", "video": f"{asset}.mp4"})
        mapping.append({"presentation_id": pid, "asset_id": asset, "repeat": repeat})
    (public / "index.html").write_text(
        PAGE.replace("__TRIALS__", json.dumps(trials)).replace("__STUDY__", study)
    )
    record = {
        "study_id": study,
        "seed": seed,
        "assets": assets,
        "presentations": mapping,
        "html_sha256": digest(public / "index.html"),
        "display_fps": 10,
        "capture_timestamps": None,
        "scope": "project-author development pilot; not independent validation",
        "repeat_policy": "repeat presentations are reliability checks, not independent labels",
    }
    (private / "study.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    result = build(args.runs, args.root, args.output)
    print(
        json.dumps({"study_id": result["study_id"], "presentations": len(result["presentations"])})
    )
