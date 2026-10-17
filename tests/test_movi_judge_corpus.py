import json

import numpy as np
import pytest

from paper.build_movi_judge_corpus import baseline
from paper.evaluate_tapvid_lora import digest, select_camera_evaluation


def test_baseline_abstains_without_texture_or_tracks():
    gray = np.zeros((256, 256), np.uint8)
    result = baseline([12, 12], dict(xy=[12, 12], forward_backward_pixels=0), gray, gray, [12, 12])
    assert result['rgb_baseline_answer'] == 'uncertain'
    result = baseline([12, 12], dict(xy=None, forward_backward_pixels=None), gray, gray, [12, 12])
    assert result['rgb_baseline_answer'] == 'uncertain'


def test_complete_movi_roster_and_target_exclusion(tmp_path):
    image = tmp_path/'panel.png'
    image.write_bytes(b'hash fixture')
    records = [dict(id=f'{n}:{m}:{s}:{f}', sequence=n, method=m, track=s, frame=f,
                    source_frame=0, source_xy=[12+32*(s%8), 12+32*(s//8)], status='missing_prefix_observations')
               for n in ('3835', '5397', '4385', '4970') for m in ('dage', 'da3')
               for s in range(64) for f in (12, 18, 23)]
    records[0].update(status='renderable', image=image.name, image_sha256=digest(image))
    export = tmp_path/'inference.jsonl'
    row = dict(id=records[0]['id'], image=image.name, prompt='frozen')
    export.write_text(json.dumps(row)+'\n')
    manifest = dict(application='movi_marker_development', records=records,
                    prompt='frozen', planned_presentations=1536, export_sha256=digest(export))
    train, dev = dict(training_ids=[]), dict(prompt='frozen')
    _, rows, allowed = select_camera_evaluation(tmp_path, train, manifest, dev)
    assert len(rows) == len(allowed) == 1
    row['target'] = 'close'
    export.write_text(json.dumps(row)+'\n')
    manifest['export_sha256'] = digest(export)
    with pytest.raises(ValueError, match='target'):
        select_camera_evaluation(tmp_path, train, manifest, dev)
    records.pop()
    with pytest.raises(ValueError, match='roster'):
        select_camera_evaluation(tmp_path, train, manifest, dev)
