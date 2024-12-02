import tempfile

import numpy as np
import tensorflow as tf

from afqinsight import AFQDataset
from afqinsight.nn.performance_utils import (
    prep_tensorflow_data,
)
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

dataset = AFQDataset.from_study("hbn")

epochs = 1
train_dataset, X_test, X_train, y_test, val_dataset = prep_tensorflow_data(dataset)


# print(X_train.shape[1:])
def test_tensorflow_model(
    model, train_dataset, val_dataset, X_test, y_test, n_epochs=100
):
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


def test_mlp4():
    tf_model = mlp4(
        input_shape=X_train.shape[1:],
        n_classes=1,
        verbose=True,
        output_activation="linear",
    )
    print(X_train.shape[1:])
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_cnn_lenet():
    tf_model = cnn_lenet(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_cnn_vgg():
    tf_model = cnn_vgg(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_lstm1v0():
    tf_model = lstm1v0(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_lstm1():
    tf_model = lstm1(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_lstm2():
    tf_model = lstm2(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_blstm1():
    tf_model = blstm1(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_blstm2():
    tf_model = blstm2(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_lstm_fcn():
    tf_model = lstm_fcn(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


def test_cnn_resnet():
    tf_model = cnn_resnet(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=epochs
    )


test_mlp4()  # done
test_cnn_lenet()  # done
test_cnn_vgg()  # done
test_lstm1v0()  # no
test_lstm1()  # no
test_lstm2()  # no
test_blstm1()  # no
test_blstm2()  #
test_lstm_fcn()  # done
test_cnn_resnet()  # done
