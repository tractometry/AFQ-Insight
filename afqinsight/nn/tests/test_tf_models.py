import tempfile

import numpy as np
import pytest
import tensorflow as tf

from afqinsight import AFQDataset
from afqinsight.nn.tf_models import (
    blstm1,
    blstm2,
    cnn_lenet,
    cnn_resnet,
    cnn_vgg,
    lstm1,
    lstm1v0,
    lstm2,
    lstm_fcn,
    mlp4,
)
from afqinsight.nn.utils import prep_tensorflow_data


@pytest.fixture
def dataset():
    """Fixture to load the AFQ dataset."""
    return AFQDataset.from_study("hbn")


@pytest.fixture
def data_loaders(dataset):
    """Fixture to prepare TensorFlow datasets."""
    train_dataset, X_test, X_train, y_test, val_dataset = prep_tensorflow_data(dataset)
    return train_dataset, X_test, X_train, y_test, val_dataset


def run_tensorflow_model(model, data_loaders, n_epochs=100):
    """General test function for training TensorFlow models."""
    train_dataset, X_test, X_train, y_test, val_dataset = data_loaders
    lr = 0.0001

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            "mean_squared_error",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            "mean_absolute_error",
        ],
    )

    ckpt_filepath = tempfile.NamedTemporaryFile(suffix=".weights.h5").name
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.001, mode="min", patience=100
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, verbose=1
    )

    callbacks = [early_stopping, ckpt, reduce_lr]

    history = model.fit(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=2,
    )
    print(history.history)

    results = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32))

    print(
        f"Test Results - Loss: {results[0]}, RMSE: {results[1]}, MAE: {results[2]},"
        f"MSE: {results[3]}"
    )


@pytest.mark.parametrize(
    "model_fn",
    [
        mlp4,
        cnn_lenet,
        cnn_vgg,
        lstm1v0,
        lstm1,
        lstm2,
        blstm1,
        blstm2,
        lstm_fcn,
        cnn_resnet,
    ],
)
def test_tensorflow_models(model_fn, data_loaders):
    """Test multiple TensorFlow models."""
    train_dataset, X_test, X_train, y_test, val_dataset = data_loaders
    input_shape = X_train.shape[1:]  # Dynamically infer the input shape
    tf_model = model_fn(
        input_shape=input_shape, n_classes=1, verbose=True, output_activation="linear"
    )
    run_tensorflow_model(
        tf_model,
        data_loaders,
        n_epochs=1,  # Reduced epochs for testing
    )
