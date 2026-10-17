import numpy as np
import pytest

from plumbline.paired_location_bounds import squared_difference_bounds


def test_shared_location_extrema_are_attained():
    rng = np.random.default_rng(0)
    a,b,q = rng.normal(size=(3,20,2))
    r = rng.uniform(0,3,20)
    result = squared_difference_bounds(a,b,q,r)
    direction = (a-b)/np.linalg.norm(a-b,axis=1,keepdims=True)
    values = []
    for target in (q+r[:,None]*direction,q-r[:,None]*direction):
        values.append(np.mean(np.sum((a-target)**2-(b-target)**2,axis=1)))
    np.testing.assert_allclose(values,result['mean_squared_error_difference_bounds'])
    translated = squared_difference_bounds(a+50,b+50,q+50,r)
    np.testing.assert_allclose(translated['mean_squared_error_difference_bounds'],values)


def test_identical_predictions_cancel_shared_error():
    result = squared_difference_bounds([[1,2]],[[1,2]],[[90,80]],1e6)
    assert result['ordering'] == 'equal'
    assert result['mean_squared_error_difference_bounds'] == [0,0]


def test_sensitivity_widens_and_does_not_certify_missing_data():
    assert squared_difference_bounds([[0,0]],[[2,0]],[[0,0]],0)['ordering'] == 'left'
    assert squared_difference_bounds([[0,0]],[[2,0]],[[0,0]],1)['ordering'] == 'unresolved'
    for radius in (-1,float('inf')):
        with pytest.raises(ValueError):
            squared_difference_bounds([[0,0]],[[2,0]],[[0,0]],radius)
    with pytest.raises(ValueError):
        squared_difference_bounds([],[],[],1)
    with pytest.raises(ValueError):
        squared_difference_bounds([[0,0]],[[2,0]],[[None,None]],1)
