import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def test_archived_counterfactual_pairing_and_lineage():
    root = Path(__file__).resolve().parents[1]/'paper'
    saved = json.loads((root/'results/correspondence-image-dependence-v1/manifest.json').read_text())
    assert saved['status'] == 'complete'
    assert saved['script_sha256'] == digest(root/'audit_correspondence_image_dependence.py')
    assert saved['training_manifest_sha256'] == digest(root/'results/tapvid-lora-twenty-pass-v1/training.json')
    assert saved['corpus_manifest_sha256'] == digest(root/'results/tapvid-judge-corpus-v1/manifest.json')
    rows = saved['records']
    assert len(rows) == 6
    for first, second in zip(rows[:3], rows[3:], strict=True):
        assert first['arm'] == 'base' and second['arm'] == 'adapted'
        assert first['id'] == second['id']
        assert first['image_tokens'] == second['image_tokens'] > 0
        for condition in ('original', 'zero_pixels'):
            assert first['conditions'][condition]['pixel_sha256'] == second['conditions'][condition]['pixel_sha256']
        assert first['conditions']['original']['pixel_sha256'] != first['conditions']['zero_pixels']['pixel_sha256']
        assert 0 < first['first_token_total_variation'] <= 1
        assert 0 < second['first_token_total_variation'] <= 1
