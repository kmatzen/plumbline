import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from paper.build_tapvid_judge_corpus import PROMPT
from paper.combine_tapvid_camera_transfer import combine, validate_frozen_proposals
from paper.evaluate_tapvid_lora import select_evaluation
from paper.run_tapvid_cameras import digest


def test_twenty_video_combination_and_evaluation_input_selection(tmp_path):
    source, corpora = tmp_path/"source", tmp_path/"corpora"
    source.mkdir()
    names = sorted(f"video{i}" for i in range(20))
    frame = np.random.default_rng(0).integers(0, 256, (24, 24, 3), dtype=np.uint8)
    video = np.repeat(frame[None], 61, axis=0)
    entries = {}
    for name in names:
        path = source/f"{name}.npz"
        np.savez(path, video=video)  # Combiner must not need annotation arrays.
        entries[name] = dict(split="held_out", file=path.name, file_sha256=digest(path),
                             video_shape=list(video.shape), points_shape=[1, 61, 2],
                             decoded_array_sha256=dict(video=hashlib.sha256(video.tobytes()).hexdigest()))
    staging = source/"manifest.json"
    staging.write_text(json.dumps(dict(sequences=entries)))
    for name in names:
        directory = corpora/f"{name}-v1"
        directory.mkdir(parents=True)
        image = directory/"panel.png"
        cv2.imwrite(str(image), frame)
        frozen = [dict(id=f"{name}:{method}:{lag}", sequence=name, model=method, track=0,
                       source_frame=0, source_xy=[12.,12.], lag=lag, frame=lag,
                       candidate_xy=[12.,12.], status="renderable")
                  for method in ("dage", "da3") for lag in (15,30,60)]
        records = [dict(row, scoring_status="labeled", label="far" if row["lag"] == 15 else "close",
                        image=image.name, image_sha256=digest(image)) for row in frozen]
        geometry, export = directory/"geometry.json", directory/"evaluation.jsonl"
        geometry.write_text(json.dumps(dict(records=frozen)))
        export.write_text("")
        manifest = dict(sequence=name, prompt=PROMPT, source_manifest_sha256=digest(staging),
                        source_file_sha256=entries[name]["file_sha256"], geometry_sha256=digest(geometry),
                        export_sha256=digest(export), records=records)
        (directory/"manifest.json").write_text(json.dumps(manifest))
    plan = tmp_path/"protocol.json"
    plan.write_text(json.dumps(dict(sequences=names, methods=["dage","da3"])))
    output = tmp_path/"combined"
    result = combine(source, corpora, plan, output)
    assert result["planned_presentations"] == 120
    assert result["counts"] == {"far":40, "close":80}
    # Identical source/candidate patches yield the same baseline for both labels.
    assert all(r["patch_baseline_prediction"] == "close" for r in result["records"])
    dev = dict(partitions={f"dev{i}":"train" for i in range(10)}, records=[],
               source_manifest_sha256=digest(staging), prompt=PROMPT)
    _, rows, allowed = select_evaluation(output, dict(training_ids=[]), result, "test", dev)
    assert len(rows) == len(allowed) == 120
    for row in rows:
        assert digest(output/row["image"]) == allowed[row["id"]]["image_sha256"]


def test_reference_scoring_cannot_change_frozen_candidate():
    row = dict(id="one", sequence="sample", model="dage", track=0, source_frame=0,
               source_xy=[1,2], lag=15, frame=15, candidate_xy=[3,4], status="renderable")
    scored = dict(row, label="far")
    validate_frozen_proposals([scored], [row])
    altered = copy.deepcopy(scored)
    altered["candidate_xy"] = [4,5]
    with pytest.raises(ValueError, match="geometry"):
        validate_frozen_proposals([altered], [row])
