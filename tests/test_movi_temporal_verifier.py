import json
from pathlib import Path

from paper.analyze_movi_temporal_verifier import run
from paper.decode_movi_geometry_pilot import digest


def test_temporal_verifier_matched_diagnostic_replay():
    paper=Path(__file__).resolve().parents[1]/'paper'
    result_root=paper/'results/movi-temporal-verifier-v1'
    path=result_root/'manifest.json'
    saved=json.loads(path.read_text())
    assert saved['script_sha256']==digest(paper/'probe_movi_temporal_verifier.py')
    result=run(paper/'results/movi-camera-pilot-v1',path)
    assert result==json.loads((result_root/'diagnostic.json').read_text())
    assert result['results']['all']['diagnostic_labeled']==389
    assert result['results']['all']['correct']==366
    assert result['results']['all']['answered']==372
    assert result['results']['rgb_abstained']['correct']==121
    assert result['results']['rgb_abstained']['diagnostic_labeled']==142
