import json

import pytest

from paper import run_movi_camera_batch as batch


def test_fixed_eight_attempts_continue_after_failure(tmp_path,monkeypatch):
    root=tmp_path/'input'; root.mkdir()
    (root/'manifest.json').write_text('{}')
    plan=tmp_path/'protocol.json'
    plan.write_text(json.dumps(dict(sequences=['3835','5397','4385','4970'],methods=['dage','da3'],
                                     input_manifest_sha256=batch.digest(root/'manifest.json'))))
    calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        return 1 if len(calls)==1 else 0
    monkeypatch.setattr(batch.subprocess,'call',execute)
    output=tmp_path/'runs'
    result=batch.run(root,plan,tmp_path/'dage',tmp_path/'da3',output)
    assert len(calls)==len(result['attempts'])==8
    assert result['attempts'][0]['status']=='process_failed'
    assert all(r['status']=='process_complete' for r in result['attempts'][1:])
    assert result['status']=='finished_all_attempts'
    with pytest.raises(FileExistsError):
        batch.run(root,plan,tmp_path/'dage',tmp_path/'da3',output)
