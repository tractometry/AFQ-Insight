from afqinsight.nn.pt_models import (
    blstm1_pt,
    cnn_lenet_pt,
    cnn_vgg_pt,
    lstm1_pt,
    lstm1v0_pt,
    lstm2_pt,
    mlp4_pt,
)
from afqinsight.nn.tf_models import (
    blstm1,
    cnn_lenet,
    cnn_vgg,
    lstm1,
    lstm1v0,
    lstm2,
    mlp4,
)
from afqinsight.nn.utils import compare_models


# test for mlp4
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


# test for LSTM1V0
def test_lstm1v0():
    pytorch_lstm1v0 = lstm1v0_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1v0 = lstm1v0(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1v0, tensorflow_lstm1v0)


# test for LSTM1
def test_lstm1():
    pytorch_lstm1 = lstm1_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1 = lstm1(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1, tensorflow_lstm1)


# test for LSTM2
def test_lstm2():
    pytorch_lstm2 = lstm2_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm2 = lstm2(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm2, tensorflow_lstm2)


def test_blstm1():
    pytorch_blstm1 = blstm1_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_blstm1 = blstm1(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_blstm1, tensorflow_blstm1)
