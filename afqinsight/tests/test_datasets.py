import os.path as op
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso

import afqinsight as afqi
from afqinsight.datasets import (
    AFQDataset,
    bundles2channels,
    download_sarica,
    download_weston_havens,
    load_afq_data,
    standardize_subject_id,
)

data_path = op.join(afqi.__path__[0], "data")
test_data_path = op.join(data_path, "test_data")


def test_bundles2channels():
    X0 = np.random.rand(50, 4000)
    X1 = bundles2channels(X0, n_nodes=100, n_channels=40, channels_last=True)
    assert X1.shape == (50, 100, 40)
    assert np.allclose(X0[:, :100], X1[:, :, 0])

    X1 = bundles2channels(X0, n_nodes=100, n_channels=40, channels_last=False)
    assert X1.shape == (50, 40, 100)
    assert np.allclose(X0[:, :100], X1[:, 0, :])

    with pytest.raises(ValueError):
        bundles2channels(X0, n_nodes=1000, n_channels=7)


def test_standardize_subject_id():
    assert standardize_subject_id("sub-01") == "sub-01"
    assert standardize_subject_id("01") == "sub-01"


def test_afqdataset_label_encode():
    sub_dicts = [
        {"subject_id": "1", "age": 0, "site": "A"},
        {"subject_id": "2", "age": 1, "site": "B"},
        {"subject_id": "3", "age": 2},
    ]
    node_dicts = [
        {"subjectID": "sub-1", "tractID": "A", "nodeID": 0, "fa": 0.1},
        {"subjectID": "sub-1", "tractID": "A", "nodeID": 1, "fa": 0.2},
        {"subjectID": "sub-1", "tractID": "B", "nodeID": 0, "fa": 0.3},
        {"subjectID": "sub-1", "tractID": "B", "nodeID": 1, "fa": 0.3},
        {"subjectID": "sub-2", "tractID": "A", "nodeID": 0, "fa": 0.4},
        {"subjectID": "sub-2", "tractID": "A", "nodeID": 1, "fa": 0.5},
        {"subjectID": "sub-2", "tractID": "B", "nodeID": 0, "fa": 0.6},
        {"subjectID": "sub-2", "tractID": "B", "nodeID": 1, "fa": 0.6},
        {"subjectID": "3", "tractID": "A", "nodeID": 0, "fa": 0.7},
        {"subjectID": "3", "tractID": "A", "nodeID": 1, "fa": 0.8},
        {"subjectID": "3", "tractID": "B", "nodeID": 0, "fa": 0.9},
        {"subjectID": "3", "tractID": "B", "nodeID": 1, "fa": 0.9},
    ]
    subs = pd.DataFrame(sub_dicts)
    nodes = pd.DataFrame(node_dicts)

    with tempfile.TemporaryDirectory() as temp_dir:
        subs.to_csv(op.join(temp_dir, "subjects.csv"), index=False)
        nodes.to_csv(op.join(temp_dir, "nodes.csv"), index=False)

        tmp_dataset = afqi.AFQDataset.from_files(
            fn_nodes=op.join(temp_dir, "nodes.csv"),
            fn_subjects=op.join(temp_dir, "subjects.csv"),
            target_cols=["site"],
            dwi_metrics=["fa"],
            index_col="subject_id",
            label_encode_cols=["site"],
        )

        assert tmp_dataset.y.shape == (3,)
        tmp_dataset.drop_target_na()
        assert tmp_dataset.y.shape == (2,)

        tmp_dataset = afqi.AFQDataset.from_files(
            fn_nodes=op.join(temp_dir, "nodes.csv"),
            fn_subjects=op.join(temp_dir, "subjects.csv"),
            target_cols=["age", "site"],
            dwi_metrics=["fa"],
            index_col="subject_id",
            label_encode_cols=["site"],
        )

        assert tmp_dataset.y.shape == (3, 2)
        tmp_dataset.drop_target_na()
        assert tmp_dataset.y.shape == (2, 2)


def test_afqdataset_sub_prefix():
    sub_dicts = [
        {"subject_id": "1", "age": 0},
        {"subject_id": "2", "age": 1},
        {"subject_id": "3", "age": 2},
    ]
    node_dicts = [
        {"subjectID": "sub-1", "tractID": "A", "nodeID": 0, "fa": 0.1},
        {"subjectID": "sub-1", "tractID": "A", "nodeID": 1, "fa": 0.2},
        {"subjectID": "sub-1", "tractID": "B", "nodeID": 0, "fa": 0.3},
        {"subjectID": "sub-1", "tractID": "B", "nodeID": 1, "fa": 0.3},
        {"subjectID": "sub-2", "tractID": "A", "nodeID": 0, "fa": 0.4},
        {"subjectID": "sub-2", "tractID": "A", "nodeID": 1, "fa": 0.5},
        {"subjectID": "sub-2", "tractID": "B", "nodeID": 0, "fa": 0.6},
        {"subjectID": "sub-2", "tractID": "B", "nodeID": 1, "fa": 0.6},
        {"subjectID": "3", "tractID": "A", "nodeID": 0, "fa": 0.7},
        {"subjectID": "3", "tractID": "A", "nodeID": 1, "fa": 0.8},
        {"subjectID": "3", "tractID": "B", "nodeID": 0, "fa": 0.9},
        {"subjectID": "3", "tractID": "B", "nodeID": 1, "fa": 0.9},
    ]
    subs = pd.DataFrame(sub_dicts)
    nodes = pd.DataFrame(node_dicts)

    with tempfile.TemporaryDirectory() as temp_dir:
        subs.to_csv(op.join(temp_dir, "subjects.csv"), index=False)
        nodes.to_csv(op.join(temp_dir, "nodes.csv"), index=False)

        tmp_dataset = afqi.AFQDataset.from_files(
            fn_nodes=op.join(temp_dir, "nodes.csv"),
            fn_subjects=op.join(temp_dir, "subjects.csv"),
            target_cols=["age"],
            dwi_metrics=["fa"],
            index_col="subject_id",
        )

    assert set(tmp_dataset.subjects) == set([f"sub-{i}" for i in range(1, 4)])  # noqa C416
    assert tmp_dataset.X.shape == (3, 4)
    assert tmp_dataset.y.shape == (3,)
    assert np.isnan(tmp_dataset.y).sum() == 0


def test_AFQDataset_shape_len_index():
    dataset = AFQDataset(
        X=np.random.rand(10, 4), y=np.random.rand(10), target_cols=["class"]
    )
    assert len(dataset) == 10
    assert dataset.shape == ((10, 4), (10,))
    assert len(dataset[:2]) == 2
    assert isinstance(dataset[:2], AFQDataset)
    assert (
        repr(dataset)
        == "AFQDataset(n_samples=10, n_features=4, n_targets=1, targets=['class'])"
    )

    dataset = AFQDataset(X=np.random.rand(10, 4), y=np.random.rand(10))
    assert len(dataset) == 10
    assert dataset.shape == ((10, 4), (10,))
    assert len(dataset[:2]) == 2
    assert isinstance(dataset[:2], AFQDataset)
    assert repr(dataset) == "AFQDataset(n_samples=10, n_features=4, n_targets=1)"

    dataset = AFQDataset(X=np.random.rand(10, 4))
    assert len(dataset) == 10
    assert dataset.shape == (10, 4)
    assert len(dataset[:2]) == 2
    assert isinstance(dataset[:2], AFQDataset)
    assert repr(dataset) == "AFQDataset(n_samples=10, n_features=4)"


def test_AFQDataset_fit_transform():
    sarica_dir = download_sarica()
    dataset = AFQDataset.from_files(
        fn_nodes=op.join(sarica_dir, "nodes.csv"),
        fn_subjects=op.join(sarica_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        target_cols=["class"],
        label_encode_cols=["class"],
    )

    # Test that model_fit fits the imputer
    imputer = dataset.model_fit(SimpleImputer())
    assert np.allclose(imputer.statistics_, np.nanmean(dataset.X, axis=0))

    # Test that model_transform imputes the data
    dataset_imputed = dataset.model_transform(imputer)
    assert np.allclose(dataset_imputed.X, imputer.transform(dataset.X))

    # Test that fit_transform does the same as fit and then transform
    dataset_transformed = dataset.model_fit_transform(SimpleImputer())
    assert np.allclose(dataset_transformed.X, dataset_imputed.X)


def test_AFQDataset_copy():
    wh_dir = download_weston_havens()
    dataset_1 = AFQDataset.from_files(
        fn_nodes=op.join(wh_dir, "nodes.csv"),
        fn_subjects=op.join(wh_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        target_cols=["Age"],
    )
    dataset_2 = dataset_1.copy()

    # Test that it copied
    assert np.allclose(dataset_1.X, dataset_2.X, equal_nan=True)
    assert dataset_1.groups == dataset_2.groups
    assert dataset_1.group_names == dataset_2.group_names
    assert dataset_1.subjects == dataset_2.subjects

    # Test that it's a deep copy
    dataset_1.X = np.zeros_like(dataset_2.X)
    dataset_1.y = np.zeros_like(dataset_2.y)
    assert not np.allclose(dataset_2.X, dataset_1.X, equal_nan=True)
    assert not np.allclose(dataset_1.y, dataset_2.y, equal_nan=True)


def test_AFQDataset_predict_score():
    wh_dir = download_weston_havens()
    dataset = AFQDataset.from_files(
        fn_nodes=op.join(wh_dir, "nodes.csv"),
        fn_subjects=op.join(wh_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        target_cols=["Age"],
    )
    dataset = dataset.model_fit_transform(SimpleImputer(strategy="median"))
    estimator = dataset.model_fit(Lasso())
    y_pred = dataset.model_predict(estimator)
    assert np.allclose(estimator.predict(dataset.X), y_pred)
    assert np.allclose(
        estimator.score(dataset.X, dataset.y), dataset.model_score(estimator)
    )


def test_drop_target_na():
    dataset = AFQDataset(X=np.random.rand(10, 4), y=np.random.rand(10))
    dataset.y[:5] = np.nan
    dataset.drop_target_na()
    assert len(dataset) == 5

    dataset = AFQDataset(
        X=np.random.rand(10, 4),
        y=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]),
        target_cols=["class"],
        classes={"class": np.array(["A", "B", "NaN", "C"], dtype=object)},
    )
    dataset.drop_target_na()
    assert len(dataset) == 7


@pytest.mark.parametrize("target_cols", [["class"], ["age", "class"]])
def test_AFQDataset(target_cols):
    sarica_dir = download_sarica()
    afq_data = AFQDataset.from_files(
        fn_nodes=op.join(sarica_dir, "nodes.csv"),
        fn_subjects=op.join(sarica_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        target_cols=target_cols,
        label_encode_cols=["class"],
    )

    y_shape = (48, 2) if len(target_cols) == 2 else (48,)

    assert afq_data.X.shape == (48, 4000)
    assert afq_data.y.shape == y_shape
    assert len(afq_data.groups) == 40
    assert len(afq_data.feature_names) == 4000
    assert len(afq_data.group_names) == 40
    assert len(afq_data.subjects) == 48
    assert afq_data.bundle_means().shape == (48, 40)

    # Test pytorch dataset method

    pt_dataset = afq_data.as_torch_dataset()
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 40, 100)
    assert pt_dataset.y.shape == y_shape
    assert np.allclose(pt_dataset[0][0][0], afq_data.X[0, :100], equal_nan=True)

    pt_dataset = afq_data.as_torch_dataset(channels_last=True)
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 100, 40)
    assert pt_dataset.y.shape == y_shape
    assert np.allclose(pt_dataset[0][0][:, 0], afq_data.X[0, :100], equal_nan=True)

    pt_dataset = afq_data.as_torch_dataset(bundles_as_channels=False)
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 4000)
    assert pt_dataset.y.shape == y_shape
    assert np.allclose(pt_dataset[0][0], afq_data.X[0], equal_nan=True)

    # Test tensorflow dataset method

    tf_dataset = list(afq_data.as_tensorflow_dataset().as_numpy_iterator())
    assert len(tf_dataset) == 48
    assert np.allclose(tf_dataset[0][0][:, 0], afq_data.X[0, :100], equal_nan=True)

    tf_dataset = list(
        afq_data.as_tensorflow_dataset(channels_last=False).as_numpy_iterator()
    )
    assert len(tf_dataset) == 48
    assert np.allclose(tf_dataset[0][0][0], afq_data.X[0, :100], equal_nan=True)

    tf_dataset = list(
        afq_data.as_tensorflow_dataset(bundles_as_channels=False).as_numpy_iterator()
    )
    assert len(tf_dataset) == 48
    assert np.allclose(tf_dataset[0][0], afq_data.X[0], equal_nan=True)

    # Test the drop_target_na method
    afq_data.y = afq_data.y.astype(float)
    if len(target_cols) == 2:
        afq_data.y[0, 0] = np.nan
        y_shape = (47, 2)
    else:
        afq_data.y[0] = np.nan
        y_shape = (47,)

    afq_data.drop_target_na()
    assert afq_data.X.shape == (47, 4000)
    assert afq_data.y.shape == y_shape
    assert len(afq_data.subjects) == 47

    # Do it all again for an unsupervised dataset

    afq_data = AFQDataset.from_files(
        fn_nodes=op.join(sarica_dir, "nodes.csv"),
        fn_subjects=op.join(sarica_dir, "subjects.csv"),
        dwi_metrics=["md", "fa"],
        unsupervised=True,
    )

    assert afq_data.X.shape == (48, 4000)
    assert afq_data.y is None
    assert len(afq_data.groups) == 40
    assert len(afq_data.feature_names) == 4000
    assert len(afq_data.group_names) == 40
    assert len(afq_data.subjects) == 48

    pt_dataset = afq_data.as_torch_dataset()
    assert len(pt_dataset) == 48
    assert pt_dataset.X.shape == (48, 40, 100)
    assert torch.all(torch.eq(pt_dataset.y, torch.tensor([])))
    assert np.allclose(pt_dataset[0][0], afq_data.X[0, :100], equal_nan=True)

    tf_dataset = list(afq_data.as_tensorflow_dataset().as_numpy_iterator())
    assert len(tf_dataset) == 48
    assert np.allclose(tf_dataset[0][:, 0], afq_data.X[0, :100], equal_nan=True)

    # Test the drop_target_na method does nothing in the unsupervised case
    afq_data.drop_target_na()
    assert afq_data.X.shape == (48, 4000)
    assert afq_data.y is None
    assert len(afq_data.subjects) == 48


@pytest.mark.parametrize("study", ["sarica", "weston-havens", "hbn"])
def test_from_study(study):
    dataset = AFQDataset.from_study(study=study)

    shapes = {
        "sarica": {
            "n_subjects": 48,
            "n_features": 4000,
            "n_groups": 40,
            "target_cols": ["class"],
        },
        "weston-havens": {
            "n_subjects": 77,
            "n_features": 4000,
            "n_groups": 40,
            "target_cols": ["Age"],
        },
        "hbn": {
            "n_subjects": 1878,
            "n_features": 4800,
            "n_groups": 48,
            "target_cols": ["age", "sex", "scan_site_id"],
        },
    }

    n_subjects = shapes[study]["n_subjects"]
    n_features = shapes[study]["n_features"]
    target_cols = shapes[study]["target_cols"]
    n_targets = len(target_cols)
    n_groups = shapes[study]["n_groups"]
    X_shape = (n_subjects, n_features)
    y_shape = (n_subjects, n_targets) if n_targets > 1 else (n_subjects,)

    assert dataset.shape == (X_shape, y_shape)
    assert dataset.target_cols == target_cols
    assert len(dataset.groups) == n_groups
    assert len(dataset.group_names) == n_groups


@pytest.mark.parametrize("dwi_metrics", [["md", "fa"], None])
@pytest.mark.parametrize("enforce_sub_prefix", [True, False])
def test_fetch(dwi_metrics, enforce_sub_prefix):
    sarica_dir = download_sarica()

    with pytest.raises(ValueError):
        load_afq_data(
            fn_nodes=op.join(sarica_dir, "nodes.csv"),
            fn_subjects=op.join(sarica_dir, "subjects.csv"),
            dwi_metrics=dwi_metrics,
            target_cols=["class"],
            label_encode_cols=["class"],
            concat_subject_session=True,
        )

    X, y, groups, feature_names, group_names, subjects, _, _ = load_afq_data(
        fn_nodes=op.join(sarica_dir, "nodes.csv"),
        fn_subjects=op.join(sarica_dir, "subjects.csv"),
        dwi_metrics=dwi_metrics,
        target_cols=["class"],
        label_encode_cols=["class"],
        enforce_sub_prefix=enforce_sub_prefix,
    )

    n_features = 16000 if dwi_metrics is None else 4000
    n_groups = 160 if dwi_metrics is None else 40

    assert X.shape == (48, n_features)
    assert y.shape == (48,)
    assert len(groups) == n_groups
    assert len(feature_names) == n_features
    assert len(group_names) == n_groups
    assert len(subjects) == 48
    assert op.isfile(op.join(afqi.datasets._DATA_DIR, "sarica", "nodes.csv"))
    assert op.isfile(op.join(afqi.datasets._DATA_DIR, "sarica", "subjects.csv"))

    wh_dir = download_weston_havens()
    X, y, groups, feature_names, group_names, subjects, _, _ = load_afq_data(
        fn_nodes=op.join(wh_dir, "nodes.csv"),
        fn_subjects=op.join(wh_dir, "subjects.csv"),
        dwi_metrics=dwi_metrics,
        target_cols=["Age"],
    )

    n_features = 10000 if dwi_metrics is None else 4000
    n_groups = 100 if dwi_metrics is None else 40

    assert X.shape == (77, n_features)
    assert y.shape == (77,)
    assert len(groups) == n_groups
    assert len(feature_names) == n_features
    assert len(group_names) == n_groups
    assert len(subjects) == 77
    assert op.isfile(op.join(afqi.datasets._DATA_DIR, "weston_havens", "nodes.csv"))
    assert op.isfile(op.join(afqi.datasets._DATA_DIR, "weston_havens", "subjects.csv"))

    with tempfile.TemporaryDirectory() as td:
        _ = download_sarica(data_home=td)
        _ = download_weston_havens(data_home=td)
        assert op.isfile(op.join(td, "sarica", "nodes.csv"))
        assert op.isfile(op.join(td, "sarica", "subjects.csv"))
        assert op.isfile(op.join(td, "weston_havens", "nodes.csv"))
        assert op.isfile(op.join(td, "weston_havens", "subjects.csv"))


def test_load_afq_data_smoke():
    output = load_afq_data(
        fn_nodes=op.join(test_data_path, "nodes.csv"),
        fn_subjects=op.join(test_data_path, "subjects.csv"),
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
    )
    assert len(output) == 8

    output = load_afq_data(
        fn_nodes=op.join(test_data_path, "nodes.csv"),
        fn_subjects=op.join(test_data_path, "subjects.csv"),
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 8
    assert output.y is None
    assert output.classes is None

    output = load_afq_data(
        fn_nodes=op.join(test_data_path, "nodes.csv"),
        fn_subjects=op.join(test_data_path, "subjects.csv"),
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        unsupervised=True,
    )
    assert len(output) == 8
    assert output.y is None
    assert output.classes is None


@pytest.mark.parametrize("dwi_metrics", [["volume", "md"], None])
def test_load_afq_data(dwi_metrics):
    (X, y, groups, feature_names, group_names, subjects, _, classes) = load_afq_data(
        fn_nodes=op.join(test_data_path, "nodes.csv"),
        fn_subjects=op.join(test_data_path, "subjects.csv"),
        dwi_metrics=dwi_metrics,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_bundle_means=False,
        enforce_sub_prefix=False,
    )

    nodes = pd.read_csv(op.join(test_data_path, "nodes.csv"))
    X_ref = np.load(op.join(test_data_path, "test_transform_x.npy"))
    y_ref = np.load(op.join(test_data_path, "test_data_y.npy"))
    groups_ref = np.load(op.join(test_data_path, "test_transform_groups.npy"))
    cols_ref = [
        tuple(item)
        for item in np.load(op.join(test_data_path, "test_transform_cols.npy"))
    ]

    assert np.allclose(X, X_ref, equal_nan=True)
    assert np.array_equal(y, y_ref)
    assert np.allclose(groups, groups_ref)
    assert feature_names == cols_ref
    assert group_names == [tup[0:2] for tup in cols_ref if tup[2] == 0]
    assert set(subjects) == set(nodes.subjectID.unique())
    assert all(classes["test_class"] == np.array(["c0", "c1"]))

    (X, y, groups, feature_names, group_names, subjects, _, classes) = load_afq_data(
        fn_nodes=op.join(test_data_path, "nodes.csv"),
        fn_subjects=op.join(test_data_path, "subjects.csv"),
        dwi_metrics=dwi_metrics,
        target_cols=["test_class"],
        label_encode_cols=["test_class"],
        return_bundle_means=True,
        enforce_sub_prefix=False,
    )

    means_ref = (
        nodes.groupby(["subjectID", "tractID", "sessionID"])
        .agg("mean")
        .drop("nodeID", axis="columns")
        .unstack("tractID")
    )
    assert np.allclose(X, means_ref.to_numpy(), equal_nan=True)
    assert group_names == means_ref.columns.to_list()
    assert feature_names == means_ref.columns.to_list()
    assert set(subjects) == set(nodes.subjectID.unique())

    with pytest.raises(ValueError):
        load_afq_data(
            fn_nodes=op.join(test_data_path, "nodes.csv"),
            fn_subjects=op.join(test_data_path, "subjects.csv"),
            target_cols=["test_class"],
            label_encode_cols=["test_class", "error"],
        )
    with pytest.raises(ValueError) as ee:
        load_afq_data(
            fn_nodes=op.join(test_data_path, "nodes.csv"),
            fn_subjects=op.join(test_data_path, "subjects.csv"),
        )

    assert "please set `unsupervised=True`" in str(ee.value)
