import tempfile

import numpy as np
import tensorflow as tf
import torch
from neurocombat_sklearn import CombatModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from afqinsight import AFQDataset


def prep_tensorflow_data(dataset):
    dataset.drop_target_na()

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

    batch_size = 32

    print("test is running")
    X = dataset.X
    y = dataset.y[:, 0]
    site = dataset.y[:, 2, None]
    # groups = dataset.groups
    # feature_names = dataset.feature_names
    # group_names = dataset.group_names
    # subjects = dataset.subjects

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
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, X_test, X_train, y_test, val_dataset


def test_tensorflow_model(
    model, train_dataset, val_dataset, X_test, y_test, n_epochs=100
):
    lr = 0.0001

    print("XXXXXXXX")
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

    assert history.history[
        "loss"
    ], "Training history is empty. Model might not have run."
    assert not np.isnan(history.history["loss"][-1]), "Final training loss is NaN."
    assert not np.isinf(history.history["loss"][-1]), "Final training loss is Inf."

    results = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32))

    assert not np.isnan(results[0]), "Test loss is NaN."
    assert not np.isinf(results[0]), "Test loss is Inf."

    print(
        f"Test Results - Loss: {results[0]}, RMSE: {results[1]}, MAE: {results[2]},"
        f"MSE: {results[3]}"
    )


def get_pytorch_dataset():
    dataset = AFQDataset.from_study("hbn")
    dataset.drop_target_na()
    # imputer = dataset.model_fit(SimpleImputer(strategy="median"))
    # dataset = dataset.model_transform(imputer)
    torch_dataset = dataset.as_torch_dataset(
        bundles_as_channels=True, channels_last=False
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        torch_dataset,
        [
            int(0.8 * len(torch_dataset)),
            len(torch_dataset) - int(0.8 * len(torch_dataset)),
        ],
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(0.8 * len(train_dataset)),
            len(train_dataset) - int(0.8 * len(train_dataset)),
        ],
    )

    return train_dataset, val_dataset, test_dataset


def prep_pytorch_data():
    dataset = AFQDataset.from_study("hbn")
    dataset.drop_target_na()
    torch_dataset = dataset.as_torch_dataset(
        bundles_as_channels=True, channels_last=False
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        torch_dataset,
        [
            int(0.8 * len(torch_dataset)),
            len(torch_dataset) - int(0.8 * len(torch_dataset)),
        ],
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(0.8 * len(train_dataset)),
            len(train_dataset) - int(0.8 * len(train_dataset)),
        ],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    return torch_dataset, train_loader, test_loader, val_loader


def clean_tensor(tensor):
    tensor_mean = tensor[~torch.isnan(tensor)].mean()
    return torch.nan_to_num(tensor, nan=tensor_mean)


def test_pytorch_model(
    model,
    device,
    torch_dataset,
    train_loader,
    val_loader,
    test_loader,
    n_epochs=100,
    permute=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        num_samples = 0
        for input_batch, gt_batch in train_loader:
            input_batch = input_batch.to(device).float()
            gt_batch = gt_batch.to(device).float()
            input_batch = clean_tensor(input_batch)
            gt_batch = clean_tensor(gt_batch)

            # Assertions to validate input batches
            assert not torch.isnan(input_batch).any(), "NaN detected in input batch."
            assert not torch.isinf(input_batch).any(), "Inf detected in input batch."
            assert not torch.isnan(gt_batch).any(), "NaN detected in target batch."
            assert not torch.isinf(gt_batch).any(), "Inf detected in target batch."

            if permute:
                input_batch = input_batch.permute(0, 2, 1)

            optimizer.zero_grad()
            output = model(input_batch).squeeze(-1)
            gt_batch = gt_batch.squeeze(-1)

            # Assertions to validate model outputs
            assert not torch.isnan(output).any(), "NaN detected in model output."
            assert not torch.isinf(output).any(), "Inf detected in model output."

            loss = criterion(output, gt_batch)

            # Assertions to validate loss
            assert not torch.isnan(loss), "NaN detected in loss."
            assert not torch.isinf(loss), "Inf detected in loss."

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_batch.size(0)
            num_samples += input_batch.size(0)

        if num_samples == 0:
            print(
                f"No samples processed in epoch {epoch}. Skipping"
                f"training loss computation."
            )
            train_loss = float("nan")
        else:
            train_loss /= num_samples

        model.eval()
        val_loss = 0
        num_samples = 0
        with torch.no_grad():
            for input_batch, gt_batch in val_loader:
                input_batch = input_batch.to(device).float()
                gt_batch = gt_batch.to(device).float()
                input_batch = clean_tensor(input_batch)
                gt_batch = clean_tensor(gt_batch)

                # Assertions to validate validation batches
                assert not torch.isnan(
                    input_batch
                ).any(), "NaN detected in validation input batch."
                assert not torch.isinf(
                    input_batch
                ).any(), "Inf detected in validation input batch."
                assert not torch.isnan(
                    gt_batch
                ).any(), "NaN detected in validation target batch."
                assert not torch.isinf(
                    gt_batch
                ).any(), "Inf detected in validation target batch."

                if permute:
                    input_batch = input_batch.permute(0, 2, 1)

                output = model(input_batch).squeeze(-1)

                # Assertions to validate model outputs
                assert not torch.isnan(
                    output
                ).any(), "NaN detected in validation model output."
                assert not torch.isinf(
                    output
                ).any(), "Inf detected in validation model output."

                loss = criterion(output, gt_batch)

                # Assertions to validate validation loss
                assert not torch.isnan(loss), "NaN detected in validation loss."
                assert not torch.isinf(loss), "Inf detected in validation loss."

                val_loss += loss.item() * input_batch.size(0)
                num_samples += input_batch.size(0)

        if num_samples == 0:
            print(
                f"No samples processed in validation epoch {epoch}."
                f"Skipping validation loss computation."
            )
            val_loss = float("nan")
        else:
            val_loss /= num_samples

        scheduler.step(val_loss)

        # Assertions for validation loss
        assert not np.isnan(val_loss), "Validation loss is NaN."
        assert not np.isinf(val_loss), "Validation loss is Inf."

        print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")

        # Early stopping if validation loss is NaN
        if np.isnan(val_loss):
            print("Validation loss is NaN. Stopping training.")
            break
