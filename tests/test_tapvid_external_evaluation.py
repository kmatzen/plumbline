import copy

import pytest

from paper.evaluate_tapvid_lora import validate_test_partition


def manifests():
    test = dict(partitions={f"test{i}":"test" for i in range(20)}, prompt="same",
                source_manifest_sha256="same", records=[dict(id="t",partition="test",sequence="test0")])
    development = dict(partitions={f"dev{i}":"train" for i in range(10)}, prompt="same",
                       source_manifest_sha256="same", records=[dict(id="d")])
    return test,development


def test_requires_disjoint_videos_and_ids_without_reading_calibration():
    test,dev = manifests()
    validate_test_partition(test,dev)
    with pytest.raises(ValueError, match="original development"):
        validate_test_partition(test,None)
    modified = copy.deepcopy(test)
    modified["partitions"]["dev0"] = modified["partitions"].pop("test0")
    with pytest.raises(ValueError, match="disjoint"):
        validate_test_partition(modified,dev)
    modified = copy.deepcopy(test)
    modified["records"][0]["id"] = "d"
    with pytest.raises(ValueError, match="overlap"):
        validate_test_partition(modified,dev)


@pytest.mark.parametrize("field",["prompt","source_manifest_sha256"])
def test_changed_protocol_or_source_rejected(field):
    test,dev = manifests()
    test[field] = "changed"
    with pytest.raises(ValueError, match="differs"):
        validate_test_partition(test,dev)


def test_record_outside_partition_rejected():
    test,dev = manifests()
    test["records"][0]["sequence"] = "dev0"
    with pytest.raises(ValueError, match="outside"):
        validate_test_partition(test,dev)
