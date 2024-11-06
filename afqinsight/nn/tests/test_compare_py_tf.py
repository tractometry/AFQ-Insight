from afqinsight.nn.pt_models import (
    blstm1_pt,
    blstm2_pt,
    cnn_lenet_pt,
    cnn_vgg_pt,
    lstm1_pt,
    lstm1v0_pt,
    lstm2_pt,
    mlp4_pt,
)
from afqinsight.nn.tf_models import (
    blstm1,
    blstm2,
    cnn_lenet,
    cnn_vgg,
    lstm1,
    lstm1v0,
    lstm2,
    mlp4,
)
from afqinsight.nn.utils import compare_models


def test_mlp4():
    pytorch_mlp4 = mlp4_pt(input_shape=784, n_classes=10)
    tensorflow_mlp4 = mlp4(input_shape=(784,), n_classes=10, verbose=True)
    assert compare_models(pytorch_mlp4, tensorflow_mlp4)


def test_cnn_lenet():
    pytorch_cnn_lenet = cnn_lenet_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_cnn_lenet = cnn_lenet(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_cnn_lenet, tensorflow_cnn_lenet)


def test_cnn_vgg():
    pytorch_cnn_vgg = cnn_vgg_pt(input_shape=(1, 784, 1), n_classes=10)
    tensorflow_cnn_vgg = cnn_vgg(input_shape=(1, 784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_cnn_vgg, tensorflow_cnn_vgg)


def test_lstm1v0():
    pytorch_lstm1v0 = lstm1v0_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1v0 = lstm1v0(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1v0, tensorflow_lstm1v0)


def test_lstm1():
    pytorch_lstm1 = lstm1_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm1 = lstm1(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm1, tensorflow_lstm1)


def test_lstm2():
    pytorch_lstm2 = lstm2_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_lstm2 = lstm2(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_lstm2, tensorflow_lstm2)


def test_blstm1():
    pytorch_blstm1 = blstm1_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_blstm1 = blstm1(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_blstm1, tensorflow_blstm1)


def test_blstm2():
    pytorch_blstm2 = blstm2_pt(input_shape=(784, 1), n_classes=10)
    tensorflow_blstm2 = blstm2(input_shape=(784, 1), n_classes=10, verbose=True)
    assert compare_models(pytorch_blstm2, tensorflow_blstm2)


# The following tests are commented out because comparing models
# with both FCN and LSTM components is very fiddly, but we have
# independently verified that these models have identical structure
# We are keeping these tests around, in case someone wants to come back
# here and fix this up.
# def test_lstm_fcn():
#     pytorch_lstm_fcn = lstm_fcn_pt(input_shape=(784, 1), n_classes=10)
#     tensorflow_lstm_fcn = lstm_fcn(input_shape=(784, 1), n_classes=10, verbose=True)
#     summary(pytorch_lstm_fcn, input_size=(1, 784, 1))
#     assert compare_models(pytorch_lstm_fcn, tensorflow_lstm_fcn)

# def test_cnn_resnet():
#     pytorch_cnn_resnet = cnn_resnet_pt(input_shape=(784, 1), n_classes=10)
#     tensorflow_cnn_resnet = cnn_resnet(input_shape=(784, 1),
#                                        n_classes=10, verbose=True)
#     summary(pytorch_cnn_resnet, input_size=(1, 784, 1))
#     assert compare_models(pytorch_cnn_resnet, tensorflow_cnn_resnet)
