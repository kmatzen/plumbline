import hashlib
import json

import numpy as np
import pytest

from paper.run_movi_cameras import digest, load_rgb, run


def fixture(tmp_path,extra=False):
    video=np.zeros((24,256,256,3),dtype=np.uint8)
    payload=dict(video=video)
    if extra:
        payload['reference_camera']=np.eye(4)
    np.savez_compressed(tmp_path/'rgb.npz',**payload)
    scenes=[dict(name=n,rgb_file='rgb.npz',rgb_sha256=digest(tmp_path/'rgb.npz'),
                 decoded_rgb_sha256=hashlib.sha256(video.tobytes()).hexdigest(),reference_file='DO_NOT_READ.npz')
            for n in ('3835','5397','4385','4970')]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='complete',expected_scenes=4,scenes=scenes)))
    return video


def test_reads_rgb_only_and_keeps_all_frames(tmp_path,monkeypatch):
    expected=fixture(tmp_path)
    native=np.load; calls=[]
    def track(path,**kwargs):
        calls.append(path.name)
        return native(path,**kwargs)
    monkeypatch.setattr(np,'load',track)
    actual,provenance=load_rgb(tmp_path,'3835')
    np.testing.assert_array_equal(actual,expected)
    assert calls==['rgb.npz'] and len(provenance['frame_sha256'])==24


def test_reference_bearing_archive_rejected(tmp_path):
    fixture(tmp_path,extra=True)
    with pytest.raises(ValueError,match='RGB-only'):
        load_rgb(tmp_path,'3835')


def test_changed_rgb_and_existing_attempt_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path/'rgb.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        load_rgb(tmp_path,'3835')
    with pytest.raises(FileExistsError):
        run(tmp_path,'3835','dage',tmp_path,tmp_path)
