from afqinsight import AFQDataset
from afqinsight.parametric import NodeWiseRegression, node_wise_regression


def test_node_wise_regression():
    data = AFQDataset.from_study("sarica")
    tracts = ["Left Corticospinal", "Left SLF"]
    for tract in tracts:
        for lme in [True, False]:
            tract_dict = node_wise_regression(  # noqa F841
                data, tract, "fa ~ C(group) + age", lme=lme
            )


def test_NodeWiseRegression():
    data = AFQDataset.from_study("sarica")
    tracts = ["Left Corticospinal", "Left SLF"]
    for lme in [True, False]:
        model = NodeWiseRegression("fa ~ C(group) + age", lme=lme)
        model.fit(data, tracts)
