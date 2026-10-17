import numpy as np

from paper.probe_movi_crossfit_support import crossfit


def test_heldout_observations_do_not_change_their_fitted_matrix():
    rng=np.random.default_rng(0)
    first=rng.uniform(10,240,(64,2));second=first.copy()
    second[:,0]+=rng.uniform(1,20,64)
    visible=np.ones(64,bool)
    before=crossfit(first,second,visible)
    heldout=before['fits'][0]['heldout_slots']
    changed=second.copy();changed[heldout]+=rng.uniform(-30,30,(len(heldout),2))
    after=crossfit(first,changed,visible)
    assert before['fits'][0]['fit']['fundamental_matrix']==after['fits'][0]['fit']['fundamental_matrix']
    for fit in before['fits']:
        assert not set(fit['training_slots'])&set(fit['heldout_slots'])
