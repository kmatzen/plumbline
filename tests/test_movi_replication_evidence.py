import pytest

from paper.run_movi_replication_evidence import queries


def test_replication_grid_does_not_depend_on_camera_results():
    geometry = dict(tracks=[dict(track=i, query_xy=[float(x),float(y)])
        for i,(x,y) in enumerate((x,y) for y in range(12,256,32) for x in range(12,256,32))])
    before = queries(geometry)
    geometry['records'] = [{'status':'failed', 'candidate_xy':None}]
    assert (queries(geometry) == before).all()
    assert before.shape == (64,3)
    geometry['tracks'][0]['query_xy'][0] += 1
    with pytest.raises(ValueError, match='grid changed'):
        queries(geometry)
