import tensorflow as tf
import torch.nn as nn
from pt_models import cnn_lenet_pt, cnn_vgg_pt, lstm1_pt, lstm1v0_pt, mlp4_pt
from tensorflow.keras import layers
from tf_models import cnn_lenet, cnn_vgg, lstm1, lstm1v0, mlp4


# sample
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc = nn.Linear(in_features=16 * 32 * 32, out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc(x)
        return x


# sample
class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(units=10)

    def call(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


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
    elif isinstance(layer, nn.Linear):
        info["type"] = "Dense"
        info["in_features"] = layer.in_features
        info["out_features"] = layer.out_features
        info["params"] = sum(p.numel() for p in layer.parameters())
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
    elif isinstance(layer, layers.Dense):
        info["type"] = "Dense"
        kernel_shape = layer.kernel.shape
        info["in_features"] = kernel_shape[0]
        info["out_features"] = kernel_shape[1]
        info["params"] = layer.count_params()
    return info


def compare_models(pytorch_model, tensorflow_model):
    pytorch_layers = [
        module
        for module in pytorch_model.modules()
        if type(module) in [nn.Conv2d, nn.Linear]
    ]
    tensorflow_layers = [
        layer
        for layer in tensorflow_model.layers
        if type(layer) in [layers.Conv2D, layers.Dense]
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

        # Compare attributes based on layer type
        if pt_info["type"] == "Conv2D":
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
            # Padding might need special handling due to different conventions
        elif pt_info["type"] == "Dense":
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

        # Compare number of parameters
        pt_type = pt_info["type"]
        pt_params = pt_info["params"]
        tf_params = tf_info["params"]
        assert (
            pt_info["params"] == tf_info["params"]
        ), f"Number of parameters don't match in {pt_type}: {pt_params} vs {tf_params}."

    print("All layers match between the PyTorch and TensorFlow models.")
    return True


# sample test
# pytorch_model = PyTorchModel()
# tensorflow_model = TensorFlowModel()
# _ = tensorflow_model(tf.zeros([1, 32, 32, 3]))
# compare_models(pytorch_model, tensorflow_model)


# # test for mlp4
def test_mlp4():
    pytorch_mlp4 = mlp4_pt(input_shape=784, n_classes=10)
    tensorflow_mlp4 = mlp4(input_shape=(784,), n_classes=10, verbose=True)
    assert compare_models(pytorch_mlp4, tensorflow_mlp4)


# # test for cnn_lenet
def test_cnn_lenet():
    pytorch_cnn_lenet = cnn_lenet_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_cnn_lenet = cnn_lenet(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_cnn_lenet, tensorflow_cnn_lenet)


# # # test for cnn_vgg
def test_cnn_vgg():
    pytorch_cnn_vgg = cnn_vgg_pt(input_shape=(1, 784, 1), n_classes=10)
    tensorflow_cnn_vgg = cnn_vgg(input_shape=(1, 784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_cnn_vgg, tensorflow_cnn_vgg)


# # # test for LSTM1V0
def test_lstm1v0():
    pytorch_lstm1v0 = lstm1v0_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1v0 = lstm1v0(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1v0, tensorflow_lstm1v0)


# # test for LSTM1
def test_lstm1():
    pytorch_lstm1 = lstm1_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1 = lstm1(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1, tensorflow_lstm1)


test_mlp4()
test_cnn_lenet()
test_cnn_vgg()
test_lstm1v0()
test_lstm1()
