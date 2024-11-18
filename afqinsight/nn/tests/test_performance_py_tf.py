import math

import torch

from afqinsight.nn.performance_utils import (
    prep_pytorch_data,
    prep_tensorflow_data,
    test_pytorch_model,
    test_tensorflow_model,
)
from afqinsight.nn.pt_models import (
    cnn_lenet_pt,
    cnn_vgg_pt,
    lstm1v0_pt,
    mlp4_pt,
)
from afqinsight.nn.tf_models import (
    cnn_lenet,
    cnn_vgg,
    lstm1v0,
    mlp4,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
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
    pt_model = cnn_lenet_pt((100, 48), gt_shape).to(device)

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


def test_cnn_vgg():
    tf_model = cnn_vgg(
        input_shape=(100, 48), n_classes=1, verbose=True, output_activation="linear"
    )
    # input_shape = math.prod(torch_dataset[0][0])
    # gt_shape = torch_dataset[0][1].size()[0]
    pt_model = cnn_vgg_pt((100, 48), gt_shape).to(device)

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
    pt_model = lstm1v0_pt((100, 48), 1).to(device)
    for data in train_loader:
        print(f"Data shape: {data.shape}")
        data = data.permute(0, 2, 1)
        print(f"Data shape: {data.shape}")  # Verify the input shape here
        break

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


# test_mlp4()
# test_cnn_lenet()
# test_cnn_vgg() FIX
test_lstm1v0()
