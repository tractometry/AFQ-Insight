import numpy as np
import tensorflow as tf
import torch
from neurocombat_sklearn import CombatModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


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


def prep_tensorflow_data(dataset):
    dataset.drop_target_na()
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


def prep_pytorch_data(dataset):
    dataset.drop_target_na()
    imputer = dataset.model_fit(SimpleImputer(strategy="median"))
    dataset = dataset.model_transform(imputer)
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
