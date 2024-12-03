import numpy as np
import pytest
import torch

from afqinsight import AFQDataset
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
from afqinsight.nn.utils import prep_pytorch_data


@pytest.fixture
def device():
    """Fixture to set up the computing device."""
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dataset():
    """Fixture to load the AFQ dataset."""
    return AFQDataset.from_study("hbn")


@pytest.fixture
def data_loaders(dataset):
    """Fixture to prepare PyTorch datasets and data loaders."""
    torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset)
    return torch_dataset, train_loader, test_loader, val_loader


@pytest.fixture
def data_shapes(data_loaders):
    """Fixture to compute shapes for input and target tensors."""
    torch_dataset = data_loaders[0]
    gt_shape = torch_dataset[0][1].size()[0]
    sequence_length = torch_dataset[0][0].size()[0]  # 48
    in_channels = torch_dataset[0][0].size()[1]  # 100
    return gt_shape, sequence_length, in_channels


def run_pytorch_model(
    model,
    device,
    data_loaders,
    n_epochs=100,
    permute=False,
):
    """
    Smoke testing on PyTorch models to ensure it trains and tests correctly.

    Args:
        model : torch.nn.Module
            The PyTorch model to be trained and validated.
        device : torch.device
            The computing device to use for training.
        data_loaders : tuple
            Pytorch dataset
            Training data loader
            Testing data loader
            Validation data loader
        n_epochs : int
            Number of epochs to train the model.
            If no value provided, epochs is 100.
        permute : boolean
            Whether to permute the dimensions of the input batch for models
            that require input with a specific shape.

    Returns
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
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

            # Validate inputs
            assert not torch.isnan(input_batch).any()
            assert not torch.isinf(input_batch).any()
            assert not torch.isnan(gt_batch).any()
            assert not torch.isinf(gt_batch).any()

            if permute:
                input_batch = input_batch.permute(0, 2, 1)

            optimizer.zero_grad()
            output = model(input_batch)
            gt_batch = gt_batch.squeeze(-1)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

            loss = criterion(output, gt_batch)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_batch.size(0)
            num_samples += input_batch.size(0)

        train_loss /= max(num_samples, 1)

        model.eval()
        val_loss = 0
        num_samples = 0
        with torch.no_grad():
            for input_batch, gt_batch in val_loader:
                input_batch = input_batch.to(device).float()
                gt_batch = gt_batch.to(device).float()

                if permute:
                    input_batch = input_batch.permute(0, 2, 1)

                output = model(input_batch).squeeze(-1)

                loss = criterion(output, gt_batch)
                val_loss += loss.item() * input_batch.size(0)
                num_samples += input_batch.size(0)

        val_loss /= max(num_samples, 1)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")

        if np.isnan(val_loss):
            print("Validation loss is NaN. Stopping training.")
            break


@pytest.mark.parametrize(
    "model_fn, permute",
    [
        (mlp4_pt, False),
        (cnn_lenet_pt, False),
        (cnn_vgg_pt, False),
        (lstm1v0_pt, True),
        (lstm1_pt, True),
        (lstm2_pt, True),
        (blstm1_pt, True),
        (blstm2_pt, True),
        (lstm_fcn_pt, True),
        (cnn_resnet_pt, True),
    ],
)
def test_models(model_fn, permute, device, data_loaders, data_shapes):
    """
    Test multiple PyTorch models.
    """
    gt_shape, sequence_length, in_channels = data_shapes
    if model_fn in [mlp4_pt]:
        model = model_fn(in_channels * sequence_length, gt_shape).to(device)
    else:
        model = model_fn((in_channels, sequence_length), gt_shape).to(device)

    run_pytorch_model(
        model,
        device,
        data_loaders,
        n_epochs=1,  # Reduced epochs for testing
        permute=permute,
    )
