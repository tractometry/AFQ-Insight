import numpy as np

from afqinsight import AFQDataset
from afqinsight.parametric import NodeWiseRegression, node_wise_regression


def test_node_wise_regression():
    # Store results
    group_dict = {}
    group_age_dict = {}
    age_dict = {}  # noqa F841

    data = AFQDataset.from_study("sarica")
    tracts = ["Right Corticospinal", "Right SLF"]
    for tract in tracts:
        for lme in [True, False]:
            # Run different versions of this: with age, without age, only with
            # age:

            group_dict[tract] = node_wise_regression(
                data, tract, "fa ~ C(group)", lme=lme, group="group"
            )
            group_age_dict[tract] = node_wise_regression(
                data, tract, "fa ~ C(group) + age", lme=lme, group="group"
            )
            # age_dict[tract] = node_wise_regression(
            #     data, tract, "fa ~ age", lme=lme
            # )

        assert group_dict[tract]["pvals"].shape == (100,)
        assert group_age_dict[tract]["pvals"].shape == (100,)
        # assert age_dict[tract]["pval"].shape == (100,)

    assert np.any(group_dict["Right Corticospinal"]["pvals_corrected"] < 0.05)
    assert np.all(group_dict["Right SLF"]["pvals_corrected"] > 0.05)
    assert np.any(group_age_dict["Right Corticospinal"]["pvals_corrected"] < 0.05)
    assert np.all(group_age_dict["Right SLF"]["pvals_corrected"] > 0.05)


def test_NodeWiseRegression():
    data = AFQDataset.from_study("sarica")
    tracts = ["Left Corticospinal", "Left SLF"]
    for lme in [True, False]:
        model = NodeWiseRegression("fa ~ C(group) + age", lme=lme)
        model.fit(data, tracts, group="group")
