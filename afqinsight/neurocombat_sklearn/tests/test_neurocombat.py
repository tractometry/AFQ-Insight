import pytest
from sklearn.utils.estimator_checks import check_estimator, check_transformer_general

from afqinsight.neurocombat_sklearn import CombatModel


@pytest.mark.parametrize("Transformer", [CombatModel()])
def test_all_transformers(Transformer):
    return check_transformer_general("CombatModel", Transformer)


@pytest.mark.parametrize("Estimator", [CombatModel()])
def test_all_estimators(Estimator):
    return check_estimator("CombatModel", Estimator)
