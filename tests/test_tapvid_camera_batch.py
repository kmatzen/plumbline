import json
from types import SimpleNamespace

import pytest

from paper.run_tapvid_camera_batch import run


def test_all_attempts_retained_after_failed_process(tmp_path, monkeypatch):
    names = sorted(f"video{i}" for i in range(20))
    (tmp_path/"manifest.json").write_text(json.dumps(dict(sequences={n:dict(split="held_out") for n in names})))
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps(dict(sequences=names, methods=["dage", "da3"])))
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr("paper.run_tapvid_camera_batch.subprocess.run", execute)
    output = tmp_path/"batch"
    run(tmp_path, plan, tmp_path/"dage", tmp_path/"da3", output)
    result = json.loads((output/"batch.json").read_text())
    assert len(calls) == len(result["attempts"]) == 40
    assert result["attempts"][0]["status"] == "process_failed"
    assert result["status"] == "finished_all_attempts"
    with pytest.raises(ValueError, match="preserve"):
        run(tmp_path, plan, tmp_path, tmp_path, output)
