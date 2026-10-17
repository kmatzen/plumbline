import json

import numpy as np
import pytest
from PIL import Image

from paper.run_static_dage import digest, load_rgb


def test_rgb_loading_ignores_validation_metadata_and_checks_source(tmp_path):
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(path)
    frames = [{"frame": i, "rgb_index": 3 * i, "path": path.name, "timestamp": i,
               "file_sha256": digest(path), "reference": {"not_an_inference_input": True},
               "depth": "not_read"} for i in range(90)]
    manifest = tmp_path / "staging-manifest.json"
    manifest.write_text(json.dumps({"frames": frames}))
    images, recorded, source_hash = load_rgb(tmp_path)
    assert images.shape == (90, 480, 640, 3)
    assert source_hash == digest(manifest)
    assert all("reference" not in r and "depth" not in r for r in recorded)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_rgb(tmp_path)
