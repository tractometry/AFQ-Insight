import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from afqinsight.neurocombat_sklearn import CombatModel


def extract_layer_info_pytorch(layer):
    info = {}
    if isinstance(layer, nn.Conv2d):
        info["type"] = "Conv2D"
        info["in_channels"] = layer.in_channels
        info["out_channels"] = layer.out_channels
        info["kernel_size"] = layer.kernel_size
        info["stride"] = layer.stride
        info["padding"] = layer.padding
        info["params"] = sum(p.numel() for p in layer.parameters())
    elif isinstance(layer, nn.Conv1d):
        info["type"] = "Conv1D"
        info["in_channels"] = layer.in_channels
        info["out_channels"] = layer.out_channels
        info["kernel_size"] = layer.kernel_size
        info["stride"] = layer.stride
        info["padding"] = layer.padding
        info["params"] = sum(p.numel() for p in layer.parameters())
    elif isinstance(layer, nn.Linear):
        info["type"] = "Dense"
        info["in_features"] = layer.in_features
        info["out_features"] = layer.out_features
        info["params"] = sum(p.numel() for p in layer.parameters())
    elif isinstance(layer, nn.LSTM):
        info["type"] = "LSTM"
        info["input_size"] = layer.input_size
        info["hidden_size"] = layer.hidden_size
        info["num_layers"] = layer.num_layers
        info["bidirectional"] = layer.bidirectional
        info["params"] = sum(p.numel() for p in layer.parameters())
    elif isinstance(layer, nn.MaxPool1d):
        info["type"] = "MaxPooling1D"
        info["kernel_size"] = layer.kernel_size
        info["stride"] = layer.stride
        info["padding"] = layer.padding
        info["params"] = 0

    return info


def extract_layer_info_tensorflow(layer):
    info = {}
    if isinstance(layer, layers.Conv2D):
        info["type"] = "Conv2D"
        kernel_shape = layer.kernel.shape
        info["in_channels"] = kernel_shape[2]
        info["out_channels"] = kernel_shape[3]
        info["kernel_size"] = layer.kernel_size
        info["stride"] = layer.strides
        info["padding"] = layer.padding
        info["params"] = layer.count_params()
    elif isinstance(layer, layers.Conv1D):
        info["type"] = "Conv1D"
        kernel_shape = layer.kernel.shape
        info["in_channels"] = kernel_shape[1]
        info["out_channels"] = kernel_shape[2]
        info["kernel_size"] = layer.kernel_size
        info["stride"] = layer.strides
        info["padding"] = layer.padding
        info["params"] = layer.count_params()
    elif isinstance(layer, layers.Dense):
        info["type"] = "Dense"
        kernel_shape = layer.kernel.shape
        info["in_features"] = kernel_shape[0]
        info["out_features"] = kernel_shape[1]
        info["params"] = layer.count_params()
    elif isinstance(layer, layers.LSTM):
        info["type"] = "LSTM"
        info["units"] = layer.units
        info["return_sequences"] = layer.return_sequences
        info["bidirectional"] = False

        if layer.weights:
            kernel_shape = layer.weights[0].shape
            input_size = kernel_shape[0]
            info["input_size"] = input_size
        else:
            info["input_size"] = None

        info["params"] = layer.count_params()
    elif isinstance(layer, layers.Bidirectional):
        info["type"] = "LSTM"
        forward_layer = layer.forward_layer
        backward_layer = layer.backward_layer
        info["units"] = forward_layer.units + backward_layer.units
        info["return_sequences"] = forward_layer.return_sequences
        info["bidirectional"] = True

        if forward_layer.weights:
            kernel_shape = forward_layer.weights[0].shape
            input_size = kernel_shape[0]
            info["input_size"] = input_size
        else:
            info["input_size"] = None

        info["params"] = layer.count_params()
    elif isinstance(layer, layers.MaxPooling1D):
        info["type"] = "MaxPooling1D"
        info["pool_size"] = layer.pool_size
        info["stride"] = layer.strides
        info["padding"] = layer.padding
        info["params"] = 0  # MaxPooling layers have no parameters
    return info


def compare_models(pytorch_model, tensorflow_model):
    pytorch_layers = [
        module
        for module in pytorch_model.modules()
        if type(module) in [nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM]
    ]
    tensorflow_layers = [
        layer
        for layer in tensorflow_model.layers
        if type(layer)
        in [
            layers.Conv1D,
            layers.Conv2D,
            layers.Dense,
            layers.LSTM,
            layers.Bidirectional,
        ]
    ]

    assert len(pytorch_layers) == len(
        tensorflow_layers
    ), "The models have a different number of layers."

    for pt_layer, tf_layer in zip(pytorch_layers, tensorflow_layers):
        pt_info = extract_layer_info_pytorch(pt_layer)
        tf_info = extract_layer_info_tensorflow(tf_layer)

        assert (
            pt_info["type"] == tf_info["type"]
        ), f"Layer types do not match: {pt_info['type']} vs {tf_info['type']}"

        if pt_info["type"] == "Conv2D":
            print("conv2d detected")
            pt_in_channels = pt_info["in_channels"]
            pt_out_channels = pt_info["out_channels"]
            pt_kernel_size = pt_info["kernel_size"]
            pt_stride = pt_info["stride"]
            tf_in_channels = tf_info["in_channels"]
            tf_out_channels = tf_info["out_channels"]
            tf_kernel_size = tf_info["kernel_size"]
            tf_stride = tf_info["stride"]
            assert (
                pt_info["in_channels"] == tf_info["in_channels"]
            ), f"Conv2D in_channels do not match: {pt_in_channels} vs {tf_in_channels}"
            assert (
                pt_info["out_channels"] == tf_info["out_channels"]
            ), f"Conv2D out_channels don't match:{pt_out_channels} vs {tf_out_channels}"
            assert (
                pt_info["kernel_size"] == tf_info["kernel_size"]
            ), f"Conv2D kernel_size do not match: {pt_kernel_size} vs {tf_kernel_size}"
            assert (
                pt_info["stride"] == tf_info["stride"]
            ), f"Conv2D stride do not match: {pt_stride} vs {tf_stride}"
            pt_type = pt_info["type"]
            pt_params = pt_info["params"]
            tf_params = tf_info["params"]
            assert (
                pt_info["params"] == tf_info["params"]
            ), f"# of parameters don't match in {pt_type}: {pt_params} vs {tf_params}."
        elif pt_info["type"] == "Conv1D":
            print("Conv1D detected")
            pt_in_channels = pt_info["in_channels"]
            pt_out_channels = pt_info["out_channels"]
            pt_kernel_size = pt_info["kernel_size"]
            pt_stride = pt_info["stride"]
            tf_in_channels = tf_info["in_channels"]
            tf_out_channels = tf_info["out_channels"]
            tf_kernel_size = tf_info["kernel_size"]
            tf_stride = tf_info["stride"]
            assert (
                pt_in_channels == tf_in_channels
            ), f"Conv1D in_channels do not match: {pt_in_channels} vs {tf_in_channels}"
            assert pt_out_channels == tf_out_channels, (
                f"Conv1D out_channels do not match"
                f": {pt_out_channels} vs {tf_out_channels}"
            )
            assert (
                pt_kernel_size == tf_kernel_size
            ), f"Conv1D kernel_size do not match: {pt_kernel_size} vs {tf_kernel_size}"
            assert (
                pt_stride == tf_stride
            ), f"Conv1D stride do not match: {pt_stride} vs {tf_stride}"
            pt_params = pt_info["params"]
            tf_params = tf_info["params"]
            assert (
                pt_params == tf_params
            ), f"# of parameters don't match in Conv1D: {pt_params} vs {tf_params}."
        elif pt_info["type"] == "Dense":
            print("dense detected")
            pt_in_features = pt_info["in_features"]
            tf_in_features = tf_info["in_features"]
            pt_out_features = pt_info["out_features"]
            tf_out_features = tf_info["out_features"]
            assert (
                pt_info["in_features"] == tf_info["in_features"]
            ), f"Dense in_features do not match: {pt_in_features} vs {tf_in_features}"
            assert (
                pt_info["out_features"] == tf_info["out_features"]
            ), f"Dense out_features don't match: {pt_out_features} vs {tf_out_features}"

            pt_type = pt_info["type"]
            pt_params = pt_info["params"]
            tf_params = tf_info["params"]
            assert (
                pt_info["params"] == tf_info["params"]
            ), f"# of parameters don't match in {pt_type}: {pt_params} vs {tf_params}."
        elif pt_info["type"] == "LSTM":
            print("lstm detected")
            pt_input_size = pt_info["input_size"]
            tf_input_size = tf_info.get("input_size")
            pt_hidden_size = pt_info["hidden_size"]
            tf_hidden_size = tf_info.get("units")
            pt_bidirectional = pt_info["bidirectional"]
            tf_bidirectional = tf_info["bidirectional"]

            assert (
                pt_input_size == tf_input_size
            ), f"LSTM input sizes do not match: {pt_input_size} vs {tf_input_size}"
            if pt_bidirectional and tf_bidirectional:
                assert pt_hidden_size * 2 == tf_hidden_size, (
                    f"LSTM hidden sizes do not match: "
                    f"{pt_hidden_size} vs {tf_hidden_size}"
                )
                assert pt_bidirectional == tf_bidirectional, (
                    f"Bidirectional settings do not match: "
                    f"{pt_bidirectional} vs {tf_bidirectional}"
                )

            num_directions = 2 if pt_bidirectional else 1
            bias_difference = num_directions * 4 * pt_hidden_size
            print(pt_info["params"])
            adjusted_pt_params = pt_info["params"] - bias_difference

            pt_params = adjusted_pt_params
            tf_params = tf_info["params"]
            assert pt_params == tf_params, (
                f"Adjusted number of parameters don't match in LSTM: "
                f"{pt_params} vs {tf_params}."
            )
        elif pt_info["type"] == "MaxPooling1D":
            print("MaxPooling1D detected")
            pt_kernel_size = pt_info["kernel_size"]
            pt_stride = pt_info["stride"]
            pt_padding = pt_info["padding"]
            tf_pool_size = tf_info["pool_size"]
            tf_stride = tf_info["stride"]
            tf_padding = tf_info["padding"]
            assert pt_kernel_size == tf_pool_size, (
                f"MaxPooling1D kernel_size/pool_size do not match:"
                f"{pt_kernel_size} vs {tf_pool_size}"
            )
            assert (
                pt_stride == tf_stride
            ), f"MaxPooling1D strides do not match: {pt_stride} vs {tf_stride}"
            assert (
                pt_padding == tf_padding
            ), f"MaxPooling1D padding do not match: {pt_padding} vs {tf_padding}"
        else:
            print(f"Unsupported layer type: {pt_info['type']}")
            continue

    print("All layers match between the PyTorch and TensorFlow models.")
    return True


def array_to_tensor(input_array, sequence_length, in_channels):
    """
    Converts the input array to a tensor, fit for training.

    Parameters
    ----------
    input_array : array
        The input array to be converted to a tensor.
    sequence_length : int
        The length of the sequence.
    in_channels : int
        The number of input channels.

    Returns
    -------
    The input array converted to a tensor.
    """
    return np.transpose(
        np.array(
            [
                input_array[:, i * in_channels : (i + 1) * in_channels]
                for i in range(sequence_length)
            ]
        ),
        (1, 2, 0),
    )


def prep_data(input_array, site, sequence_length, in_channels):
    """
    Fills in missing values with the median of the column.

    Parameters
    ----------
    input_array : array
        The input array.
    site : array
        The site array.
    sequence_length : int
        The length of the sequence.
    in_channels : int
        The number of input channels.

    Returns
    -------
    array: The input array with missing values
            filled in with the median of the column.
    """
    return array_to_tensor(
        CombatModel().fit_transform(
            SimpleImputer(strategy="median").fit_transform(input_array), site
        ),
        sequence_length,
        in_channels,
    )


def prep_tensorflow_data(dataset):
    """
    Prepares TensorFlow datasets, for training, testing, and validation.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to be prepared.

    Returns
    -------
    tuple :
        Training dataset,
        Test dataset,
        Training data,
        Test data,
        Validation dataset.

    """
    dataset.drop_target_na()
    batch_size = 32

    X = dataset.X
    y = dataset.y[:, 0]
    site = dataset.y[:, 2, None]

    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
        X, y, site, test_size=0.2
    )

    X_train, X_val, y_train, y_val, site_train, site_val = train_test_split(
        X_train, y_train, site_train, test_size=0.2
    )
    sequence_length = len(dataset.groups)  # 48
    in_channels = len(dataset.feature_names) // sequence_length  # 100

    X_train = prep_data(X_train, site_train, sequence_length, in_channels)
    X_test = prep_data(X_test, site_test, sequence_length, in_channels)
    X_val = prep_data(X_val, site_val, sequence_length, in_channels)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train.astype(np.float32), y_train.astype(np.float32))
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val.astype(np.float32), y_val.astype(np.float32))
    )
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, X_test, X_train, y_test, val_dataset


def prep_pytorch_data(dataset):
    """
    Prepares PyTorch datasets for training, testing, and validation.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to be prepared.

    Returns
    -------
    tuple:
        PyTorch dataset,
        Training data loader,
        Test data loader,
        Validation data loader.
    """
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
