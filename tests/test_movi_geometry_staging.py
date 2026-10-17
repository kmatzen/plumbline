import base64
import hashlib

import pytest

from paper.stage_movi_geometry_pilot import FILES, stage, validate_download


def test_publisher_size_and_checksum_both_required(tmp_path):
    path = tmp_path/"payload"
    path.write_bytes(b"fixture")
    metadata = dict(size="7", md5Hash=base64.b64encode(hashlib.md5(b"fixture").digest()).decode())
    assert validate_download(path,metadata) == hashlib.sha256(b"fixture").hexdigest()
    with pytest.raises(ValueError):
        validate_download(path,dict(metadata,size="8"))
    with pytest.raises(ValueError):
        validate_download(path,dict(metadata,md5Hash="wrong"))


def test_preserves_prior_output_and_limits_to_validation(tmp_path):
    with pytest.raises(FileExistsError):
        stage(tmp_path)
    shards = [n for n in FILES if "tfrecord" in n]
    assert shards == ["movi_e-validation.tfrecord-00000-of-00064"]


def test_invalid_shard_rejected_before_output_creation(tmp_path):
    for index in (-1, 64, 'test'):
        with pytest.raises(ValueError, match='validation shard'):
            stage(tmp_path/'new', index)
        assert not (tmp_path/'new').exists()
