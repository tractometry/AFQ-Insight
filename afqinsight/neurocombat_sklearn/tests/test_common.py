import pytest
from neurocombat_sklearn import CombatModel
from sklearn.utils.estimator_checks import check_estimator


@pytest.mark.parametrize("Estimator", [CombatModel])
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
