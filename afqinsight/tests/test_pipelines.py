import groupyr as gpr
import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from afqinsight import make_afq_classifier_pipeline, make_afq_regressor_pipeline
from afqinsight._serial_bagging import SerialBaggingClassifier, SerialBaggingRegressor
from afqinsight.pipeline import (
    IdentityTransformer,
    compare_brain_covariates_cv,
    make_base_afq_pipeline,
)

scaler_args = [
    ("standard", StandardScaler),
    ("maxabs", MaxAbsScaler),
    ("minmax", MinMaxScaler),
    ("robust", RobustScaler),
    (Normalizer, Normalizer),
    (None, None),
]
imputer_args = [
    ("simple", SimpleImputer),
    ("knn", KNNImputer),
    (IterativeImputer, IterativeImputer),
]
power_args = [
    (True, PowerTransformer),
    (False, None),
    (QuantileTransformer, QuantileTransformer),
]
type_args = [
    (make_afq_classifier_pipeline, True, gpr.LogisticSGLCV),
    (make_afq_classifier_pipeline, False, gpr.LogisticSGL),
    (make_afq_regressor_pipeline, True, gpr.SGLCV),
    (make_afq_regressor_pipeline, False, gpr.SGL),
]
ensembler_args = [
    ("bagging", {True: BaggingClassifier, False: BaggingRegressor}),
    ("adaboost", {True: AdaBoostClassifier, False: AdaBoostRegressor}),
    ("serial-bagging", {True: SerialBaggingClassifier, False: SerialBaggingRegressor}),
    (AdaBoostClassifier, {True: AdaBoostClassifier, False: AdaBoostClassifier}),
    (None, None),
]


@pytest.mark.parametrize("scaler, ScalerStep", scaler_args)
@pytest.mark.parametrize("imputer, ImputerStep", imputer_args)
@pytest.mark.parametrize("feature_transformer, PowerStep", power_args)
@pytest.mark.parametrize("make_pipe, use_cv, EstimatorStep", type_args)
@pytest.mark.parametrize("target_transformer", [None, PowerTransformer])
@pytest.mark.parametrize("ensembler, EnsembleEstimators", ensembler_args)
def test_classifier_pipeline_steps(
    make_pipe,
    use_cv,
    EstimatorStep,
    scaler,
    ScalerStep,
    imputer,
    ImputerStep,
    feature_transformer,
    PowerStep,
    target_transformer,
    ensembler,
    EnsembleEstimators,
):
    pipeline = make_pipe(
        imputer=imputer,
        scaler=scaler,
        use_cv_estimator=use_cv,
        feature_transformer=feature_transformer,
        target_transformer=target_transformer,
        ensemble_meta_estimator=ensembler,
    )

    if scaler is not None:
        assert isinstance(pipeline.named_steps["scale"], ScalerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["scale"].get_params() == ScalerStep().get_params()
        )
    else:
        assert pipeline.named_steps["scale"] is None  # nosec

    if imputer is not None:
        assert isinstance(pipeline.named_steps["impute"], ImputerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["impute"].get_params() == ImputerStep().get_params()
        )
    else:
        assert pipeline.named_steps["impute"] is None  # nosec

    if feature_transformer:
        assert isinstance(pipeline.named_steps["feature_transform"], PowerStep)  # nosec
        assert (  # nosec
            pipeline.named_steps["feature_transform"].get_params()
            == PowerStep().get_params()
        )
    else:
        assert pipeline.named_steps["feature_transform"] is None  # nosec

    if ensembler is not None:
        EnsembleStep = EnsembleEstimators[is_classifier(EstimatorStep())]

    if target_transformer is None:
        if ensembler is None:
            assert isinstance(pipeline.named_steps["estimate"], EstimatorStep)  # nosec
            assert (  # nosec
                pipeline.named_steps["estimate"].get_params()
                == EstimatorStep().get_params()
            )
        else:
            assert isinstance(pipeline.named_steps["estimate"], EnsembleStep)  # nosec
            ensemble_params = pipeline.named_steps["estimate"].get_params()
            correct_params = EnsembleStep(estimator=EstimatorStep()).get_params()
            ensemble_base_est = ensemble_params.pop("estimator")
            correct_params.pop("estimator")
            assert ensemble_params == correct_params  # nosec
            assert isinstance(ensemble_base_est, EstimatorStep)  # nosec
    else:
        if ensembler is None:
            assert isinstance(  # nosec
                pipeline.named_steps["estimate"].regressor, EstimatorStep
            )
            assert (  # nosec
                pipeline.named_steps["estimate"].regressor.get_params()
                == EstimatorStep().get_params()
            )
        else:
            assert isinstance(  # nosec
                pipeline.named_steps["estimate"].regressor, EnsembleStep
            )
            ensemble_params = pipeline.named_steps["estimate"].regressor.get_params()
            correct_params = EnsembleStep(estimator=EstimatorStep()).get_params()
            ensemble_base_est = ensemble_params.pop("estimator")
            correct_params.pop("estimator")
            assert ensemble_params == correct_params  # nosec
            assert isinstance(ensemble_base_est, EstimatorStep)  # nosec


def test_pipeline_value_errors():
    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler="error")

    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(scaler=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer="error")

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(imputer=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(feature_transformer=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(feature_transformer=1729)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=object)

    with pytest.raises(ValueError):
        make_base_afq_pipeline(estimator=1729)

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator="error")

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator=object)

    with pytest.raises(ValueError):
        make_afq_regressor_pipeline(ensemble_meta_estimator=1729)


def test_base_pipeline_with_none_estimator():
    pipeline = make_base_afq_pipeline()
    assert pipeline.named_steps["estimate"] is None  # nosec


def test_base_pipeline_pass_kwargs():
    pipeline = make_base_afq_pipeline(scaler_kwargs={"with_mean": False})
    assert (  # nosec
        pipeline.named_steps["scale"].get_params()
        == StandardScaler(with_mean=False).get_params()
    )


def test_base_pipeline_pass_ensemble_kwargs():
    pipeline = make_afq_classifier_pipeline(
        ensemble_meta_estimator="bagging",
        ensemble_meta_estimator_kwargs={"n_estimators": 100},
    )
    ensemble_params = pipeline.named_steps["estimate"].get_params()
    assert ensemble_params["n_estimators"] == 100  # nosec


class TestIdentityTransformer:
    """Test the IdentityTransformer class."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        transformer = IdentityTransformer()
        X = np.array([[1, 2], [3, 4]])
        result = transformer.fit(X)
        assert result is transformer

    def test_fit_with_y_returns_self(self):
        """Test that fit with y returns self."""
        transformer = IdentityTransformer()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 0])
        result = transformer.fit(X, y)
        assert result is transformer

    def test_transform_returns_input_unchanged(self):
        """Test that transform returns input data unchanged."""
        transformer = IdentityTransformer()
        X = np.array([[1, 2], [3, 4]])
        result = transformer.transform(X)
        np.testing.assert_array_equal(result, X)
        assert result is X

    def test_fit_transform_works(self):
        """Test that fit_transform works correctly."""
        transformer = IdentityTransformer()
        X = np.array([[1, 2], [3, 4]])
        result = transformer.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_works_with_different_array_types(self):
        """Test that it works with different array types."""
        transformer = IdentityTransformer()

        # List
        X_list = [[1, 2], [3, 4]]
        result = transformer.fit_transform(X_list)
        assert result == X_list

        # 1D array
        X_1d = np.array([1, 2, 3])
        result = transformer.fit_transform(X_1d)
        np.testing.assert_array_equal(result, X_1d)


class TestCompareBrainCovariatesCV:
    """Test the compare_brain_covariates_cv function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples, n_brain_features, n_covariates = 50, 20, 3
        X_brain = np.random.randn(n_samples, n_brain_features)
        X_covariates = np.random.randn(n_samples, n_covariates)
        y = np.random.randn(n_samples)
        return X_brain, X_covariates, y

    def test_basic_functionality(self, sample_data):
        """Test basic functionality with default parameters."""
        X_brain, X_covariates, y = sample_data

        results = compare_brain_covariates_cv(
            X_brain, X_covariates, y, n_repeats=1, n_splits=3, verbose=False
        )

        # Should have 3 CV folds (1 repeat × 3 splits)
        assert len(results) == 3

        # Check that each fold has expected keys
        expected_keys = {
            "y_true",
            "y_pred_brain",
            "y_pred_covariates",
            "brain_test_mae",
            "brain_test_r2",
            "covariates_test_mae",
            "covariates_test_r2",
            "brain_train_mae",
            "brain_train_r2",
            "covariates_train_mae",
            "covariates_train_r2",
            "covariates_coefs",
            "covariates_alpha",
            "train_idx",
            "test_idx",
        }

        for _, fold_results in results.items():
            assert set(fold_results.keys()) == expected_keys
            assert isinstance(fold_results["y_true"], np.ndarray)
            assert isinstance(fold_results["brain_test_r2"], (int, float))

    def test_regress_covariates_false(self, sample_data):
        """Test with regress_covariates=False."""
        X_brain, X_covariates, y = sample_data

        results = compare_brain_covariates_cv(
            X_brain,
            X_covariates,
            y,
            regress_covariates=False,
            n_repeats=1,
            n_splits=3,
            verbose=False,
        )

        assert len(results) == 3
        # Should still have all the same output structure
        for fold_results in results.values():
            assert "brain_test_r2" in fold_results
            assert "covariates_test_r2" in fold_results

    def test_shuffle_option(self, sample_data):
        """Test shuffle=True option."""
        X_brain, X_covariates, y = sample_data

        results_no_shuffle = compare_brain_covariates_cv(
            X_brain,
            X_covariates,
            y,
            shuffle=False,
            n_repeats=1,
            n_splits=3,
            verbose=False,
            random_state=42,
        )

        results_shuffle = compare_brain_covariates_cv(
            X_brain,
            X_covariates,
            y,
            shuffle=True,
            n_repeats=1,
            n_splits=3,
            verbose=False,
            random_state=42,
        )

        # Results should be different when shuffling
        r2_no_shuffle = [r["brain_test_r2"] for r in results_no_shuffle.values()]
        r2_shuffle = [r["brain_test_r2"] for r in results_shuffle.values()]

        # At least one should be different (very high probability with random data)
        assert not np.allclose(r2_no_shuffle, r2_shuffle, rtol=1e-10)

    def test_different_cv_splits(self, sample_data):
        """Test different CV split configurations."""
        X_brain, X_covariates, y = sample_data

        # Test 2 repeats, 2 splits = 4 total folds
        results = compare_brain_covariates_cv(
            X_brain, X_covariates, y, n_repeats=2, n_splits=2, verbose=False
        )

        assert len(results) == 4

    def test_random_state_reproducibility(self, sample_data):
        """Test that random_state makes results reproducible."""
        X_brain, X_covariates, y = sample_data

        results1 = compare_brain_covariates_cv(
            X_brain,
            X_covariates,
            y,
            n_repeats=1,
            n_splits=3,
            random_state=123,
            verbose=False,
        )

        results2 = compare_brain_covariates_cv(
            X_brain,
            X_covariates,
            y,
            n_repeats=1,
            n_splits=3,
            random_state=123,
            verbose=False,
        )

        # Results should be identical
        for fold_idx in results1.keys():
            np.testing.assert_array_almost_equal(
                results1[fold_idx]["y_pred_brain"], results2[fold_idx]["y_pred_brain"]
            )
