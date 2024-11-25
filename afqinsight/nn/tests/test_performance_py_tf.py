import math

import torch

from afqinsight.nn.performance_utils import (
    prep_pytorch_data,
    prep_tensorflow_data,
    test_pytorch_model,
    test_tensorflow_model,
)
from afqinsight.nn.pt_models import (
    blstm1_pt,
    blstm2_pt,
    cnn_lenet_pt,
    cnn_resnet_pt,
    cnn_vgg_pt,
    lstm1_pt,
    lstm1v0_pt,
    lstm2_pt,
    lstm_fcn_pt,
    mlp4_pt,
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

if torch.backends.mps.is_available():
    # device = torch.device("mps")
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, X_test, X_train, y_test, val_dataset = prep_tensorflow_data()
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data()


def test_mlp4():
    tf_model = mlp4(
        input_shape=X_train.shape[1:],
        n_classes=1,
        verbose=True,
        output_activation="linear",
    )
    input_shape = math.prod(torch_dataset[0][0].size())
    gt_shape = torch_dataset[0][1].size()[0]
    pt_model = mlp4_pt(input_shape, gt_shape).to(device)

    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
    )


def test_cnn_lenet():
    tf_model = cnn_lenet(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    # input_shape = math.prod(torch_dataset[0][0])
    # gt_shape = torch_dataset[0][1].size()[0]
    pt_model = cnn_lenet_pt((100, 48), 3).to(device)

    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
    )


def test_cnn_vgg():
    tf_model = cnn_vgg(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    # input_shape = math.prod(torch_dataset[0][0])
    # gt_shape = torch_dataset[0][1].size()[0]
    pt_model = cnn_vgg_pt((100, 48), 1).to(device)

    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
    )


def test_lstm1v0():
    tf_model = lstm1v0(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = lstm1v0_pt((100, 48), 3, output_activation=False).to(device)

    # for data, target in train_loader:
    #     print(f"Data shape: {data.shape}")
    #     data = data.permute(0, 2, 1)
    #     print(f"Data shape: {data.shape}")  # Verify the input shape here
    #     break
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_lstm1():
    tf_model = lstm1(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    print("hello")

    pt_model = lstm1_pt((100, 48), 3).to(device)
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_lstm2():
    tf_model = lstm2(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = lstm2_pt((100, 48), 3).to(device)
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_blstm1():
    tf_model = blstm1(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = blstm1_pt((100, 48), 3).to(device)
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_blstm2():
    tf_model = blstm2(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = blstm2_pt((100, 48), 3).to(device)
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_lstm_fcn():
    tf_model = lstm_fcn(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = lstm_fcn_pt((100, 48), 3).to(device)
    assert test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    assert test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


def test_cnn_resnet():
    tf_model = cnn_resnet(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    pt_model = cnn_resnet_pt((100, 48), 3).to(device)
    test_tensorflow_model(
        tf_model, train_dataset, val_dataset, X_test, y_test, n_epochs=20
    )
    test_pytorch_model(
        pt_model,
        device,
        torch_dataset,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=20,
        permute=True,
    )


test_mlp4()  # good
# test_cnn_lenet() #good
# test_cnn_vgg()
# test_lstm1v0() #good
# test_lstm1() #good
# test_lstm2() #good
# test_blstm1() #good
# test_blstm2() #good
# test_lstm_fcn() #good
# test_cnn_resnet() #bad
