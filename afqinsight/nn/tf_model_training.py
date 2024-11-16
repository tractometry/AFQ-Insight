import tempfile

import numpy as np
import tensorflow as tf
from neurocombat_sklearn import CombatModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from afqinsight import AFQDataset
from afqinsight.nn.tf_models import mlp4

dataset = AFQDataset.from_study("hbn")
print(dataset)
dataset.drop_target_na()
# tensorflow_dataset = dataset.as_tensorflow_dataset(
#     bundles_as_channels=True, channels_last=True
# )
# print(tensorflow_dataset)


# converts the input array to a tensor, fit for training
def array_to_tensor(input_array):
    return np.transpose(
        np.array([input_array[:, i * 100 : (i + 1) * 100] for i in range(48)]),
        (1, 2, 0),
    )


# fills in missing values with the median of the column
def prep_data(input_array, site):
    return array_to_tensor(
        CombatModel().fit_transform(
            SimpleImputer(strategy="median").fit_transform(input_array), site
        )
    )


X = dataset.X
y = dataset.y[:, 0]
site = dataset.y[:, 2, None]
groups = dataset.groups
feature_names = dataset.feature_names
group_names = dataset.group_names
subjects = dataset.subjects

X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
    X, y, site, test_size=0.2
)

X_train, X_val, y_train, y_val, site_train, site_val = train_test_split(
    X_train, y_train, site_train, test_size=0.2
)

X_train = prep_data(X_train, site_train)
X_test = prep_data(X_test, site_test)
X_val = prep_data(X_val, site_val)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train.astype(np.float32), y_train.astype(np.float32))
)
print(train_dataset)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val.astype(np.float32), y_val.astype(np.float32))
)

for i in range(10):
    print(f"Y_TRAIN {i}: {y_train[i]}")
model = mlp4(
    input_shape=X_train.shape[1:], n_classes=1, verbose=True, output_activation="linear"
)
# model = cnn_lenet(input_shape=(100, 48),
#           n_classes=1, verbose=True, output_activation='linear')
# model = cnn_vgg(input_shape=(100, 48, 1),
#           n_classes=1, verbose=True, output_activation='linear')

lr = 0.0001
batch_size = 32
n_epochs = 100

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
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

history = model.fit(
    train_dataset,
    epochs=n_epochs,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=2,
)

results = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32))
print(
    f"Test Results - Loss: {results[0]}, RMSE: {
    results[1]}, MAE: {results[2]}, MSE: {results[3]}"
)
