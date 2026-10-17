import hashlib
import json

import numpy as np
import pytest

from paper.run_tapvid_cameras import digest, load_video, run


def test_only_rgb_loaded_and_no_calibration_sequences(tmp_path):
    video = np.zeros((3, 8, 12, 3), np.uint8)
    # Object annotations would fail if read with allow_pickle=False.
    path = tmp_path/"sample.npz"
    np.savez(path, video=video, points=np.array([object()], dtype=object), occluded=np.array([object()], dtype=object))
    entry = dict(file=path.name, split="held_out", file_sha256=digest(path), video_shape=list(video.shape),
                 decoded_array_sha256=dict(video=hashlib.sha256(video.tobytes()).hexdigest()))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(sequences=dict(sample=entry))))
    loaded, provenance = load_video(tmp_path, "sample")
    np.testing.assert_array_equal(loaded, video)
    assert len(provenance["frame_sha256"]) == 3
    entry["split"] = "development"
    manifest.write_text(json.dumps(dict(sequences=dict(sample=entry))))
    with pytest.raises(ValueError, match="held-out"):
        load_video(tmp_path, "sample")


def test_existing_attempt_preserved_without_loading_model(tmp_path):
    with pytest.raises(ValueError, match="preserve"):
        run(tmp_path, "sample", "dage", tmp_path, tmp_path)
