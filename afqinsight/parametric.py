"""Perform linear modeling at each node along the tract."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests


def node_wise_regression(
    afq_dataset,
    tract,
    formula,
    group="group",
    lme=False,
    rand_eff="subjectID",
    impute="median",
):
    """Model group differences using node-wise regression along the length of the tract.

       Returns a list of beta-weights, confidence intervals, p-values, and
       rejection criteria based on multiple-comparison correction.

    Parameters
    ----------
    afq_dataset: AFQDataset
        Loaded AFQDataset object
    tract: str
        String specifying the tract to model

    formula: str
        An R-style formula <https://www.statsmodels.org/dev/example_formulas.html>
        specifying the regression model to fit at each node. This can take the form
        of either a linear mixed-effects model or OLS regression

    lme: Bool, default=False
        Boolean specifying whether to fit a linear mixed-effects model

    rand_eff: str, default='subjectID'
        String specifying the random effect grouping structure for linear
        mixed-effects models. If using anything other than the default value,
        this column must be present in the 'target_cols' of the AFQDataset object

    impute: str or None, default='median'
        String specifying the imputation strategy to use for missing data.


    Returns
    -------
    tract_dict: dict
        A dictionary with the following key-value pairs:

        {'tract': tract,
         'reference_coefs': coefs_default,
         'group_coefs': coefs_treat,
         'reference_CI': cis_default,
         'group_CI': cis_treat,
         'pvals': pvals,
         'reject_idx': reject_idx,
         'model_fits': fits}

        tract: str
            The tract described by this dictionary

        reference_coefs: list of floats
            A list of beta-weights representing the average diffusion metric
            for the reference group on a diffusion metric at a given location
            along the tract

        group_coefs: list of floats
            A list of beta-weights representing the average group effect metric
            for the treatment group on a diffusion metric at a given location
            along the tract

        reference_CI: np.array of np.array
            A numpy array containing a series of numpy arrays indicating the
            95% confidence interval around the estimated beta-weight of the
            reference category at a given location along the tract

        group_CI: np.array of np.array
            A numpy array containing a series of numpy arrays indicating the
            95% confidence interval around the estimated beta-weight of the
            treatment effect at a given location along the tract

        pvals: list of floats
            A list of p-values testing whether or not the beta-weight of the
            group effect is different from 0

        reject_idx: list of Booleans
            A list of node indices where the null hypothesis is rejected after
            multiple-comparison corrections

        model_fits: list of statsmodels objects
            A list of the statsmodels object fit along the length of the nodes

    """
    if impute is not None:
        X = SimpleImputer(strategy=impute).fit_transform(afq_dataset.X)
    afq_dataset.target_cols[0] = group
    metric = formula.split("~")[0].strip()

    tract_data = (
        pd.DataFrame(columns=afq_dataset.feature_names, data=X)
        .filter(like=tract)
        .filter(like=metric)
    )

    pvals = np.zeros(tract_data.shape[-1])
    coefs_default = np.zeros(tract_data.shape[-1])
    coefs_treat = np.zeros(tract_data.shape[-1])
    cis_default = np.zeros((tract_data.shape[-1], 2))
    cis_treat = np.zeros((tract_data.shape[-1], 2))
    fits = {}

    # Loop through each node and fit model
    for ii, column in enumerate(tract_data.columns):
        # fit linear mixed-effects model
        if lme:
            this = pd.DataFrame(afq_dataset.y, columns=afq_dataset.target_cols)
            this[metric] = tract_data[column]

            # if no random effect specified, use subjectID as random effect
            if rand_eff == "subjectID":
                this["subjectID"] = afq_dataset.subjects

            model = smf.mixedlm(formula, this, groups=rand_eff)
            fit = model.fit()
            fits[column] = fit

        # fit OLS model
        else:
            _, _, _ = column
            this = pd.DataFrame(afq_dataset.y, columns=afq_dataset.target_cols)
            this[metric] = tract_data[column]

            model = OLS.from_formula(formula, this)
            fit = model.fit()
            fits[column] = fit

        # pull out coefficients, CIs, and p-values from our model
        coefs_default[ii] = fit.params.filter(regex="Intercept", axis=0).iloc[0]
        coefs_treat[ii] = fit.params.filter(regex=group, axis=0).iloc[0]

        cis_default[ii] = (
            fit.conf_int(alpha=0.05).filter(regex="Intercept", axis=0).values
        )
        cis_treat[ii] = fit.conf_int(alpha=0.05).filter(regex=group, axis=0).values
        pvals[ii] = fit.pvalues.filter(regex=group, axis=0).iloc[0]

    # Correct p-values for multiple comparisons
    reject, pval_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    reject_idx = np.where(reject)

    tract_dict = {
        "tract": tract,
        "reference_coefs": coefs_default,
        "group_coefs": coefs_treat,
        "reference_CI": cis_default,
        "group_CI": cis_treat,
        "pvals": pvals,
        "pvals_corrected": pval_corrected,
        "reject_idx": reject_idx,
        "model_fits": fits,
    }

    return tract_dict


class RegressionResults(object):
    def __init__(self, kwargs):
        self.tract = kwargs.get("tract", None)
        self.reference_coefs = kwargs.get("reference_coefs", None)
        self.group_coefs = kwargs.get("group_coefs", None)
        self.reference_ci = kwargs.get("reference_ci", None)
        self.group_ci = kwargs.get("group_ci", None)
        self.pvals = kwargs.get("pvals", None)
        self.pvals_corrected = kwargs.get("pvals_corrected", None)
        self.reject_idx = kwargs.get("reject_idx", None)
        self.model_fits = kwargs.get("model_fits", None)


class NodeWiseRegression(object):
    def __init__(self, formula, lme=False):
        self.formula = formula
        self.lme = lme

    def fit(self, dataset, tracts, group="group", rand_eff="subjectID"):
        self.result_ = {}
        for tract in tracts:
            self.result_[tract] = node_wise_regression(
                dataset,
                tract,
                self.formula,
                lme=self.lme,
                group=group,
                rand_eff=rand_eff,
            )
        self.is_fitted = True
        return self

    def predict(self, dataset, tract, metric, group="group", rand_eff="subjectID"):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Please call fit() method first.")
        result = self.result_.get(tract, None)
        if result is None:
            raise ValueError(f"Tract {tract} not found in the fitted model.")
