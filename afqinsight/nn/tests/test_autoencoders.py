import pytest
import torch

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import Autoencoder, VariationalAutoencoder
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


@pytest.mark.parametrize("latent_dims", [2, 10])
def test_autoencoder_forward(data_loaders, latent_dims, data_shapes):
    """
    Smoke test to check if the linear Autoencoder forward pass works
    without raising an exception and returns the expected shape.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    # Define input_shape = 48 * 100 = 4800
    model = Autoencoder(
        input_shape=sequence_length * in_channels, latent_dims=latent_dims
    )
    model.eval()  # We just do forward pass check, no training

    # Retrieve a single batch
    data_iter = iter(test_loader)
    x, _ = next(data_iter)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shapeß
    # The decoder expects to return shape: (batch_size, 48, 100)
    expected_shape = (x.size(0), sequence_length, in_channels)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, " f"but got {output.shape}."
    )


@pytest.mark.parametrize("latent_dims", [2, 10])
def test_variational_autoencoder_forward(data_loaders, latent_dims, data_shapes):
    """
    Smoke test to check if the linear VariationalAutoencoder forward pass
    works without throwing exceptions and returns the expected shape.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    model = VariationalAutoencoder(
        input_shape=sequence_length * in_channels, latent_dims=latent_dims
    )
    model.eval()

    data_iter = iter(test_loader)
    x, _ = next(data_iter)

    with torch.no_grad():
        output = model(x)

    # Check if shape matches (batch_size, 48, 100)
    expected_shape = (x.size(0), sequence_length, in_channels)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, " f"but got {output.shape}."
    )


def test_autoencoder_train_loop(data_loaders, data_shapes):
    """
    Simple smoke test for the training loop of the linear Autoencoder,
    checking for any exceptions.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    model = Autoencoder(input_shape=sequence_length * in_channels, latent_dims=10)
    model.train()

    # Fit the model on the random dataset for 1 epoch
    # This doesn't guarantee correctness, just that it runs
    model.fit(test_loader, epochs=1, lr=0.001)


def test_variational_autoencoder_train_loop(data_loaders, data_shapes):
    """
    Simple smoke test for the training loop of the linear VariationalAutoencoder,
    checking for any exceptions.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    model = VariationalAutoencoder(
        input_shape=sequence_length * in_channels, latent_dims=10
    )
    model.train()

    # Fit the model on the random dataset for 1 epoch
    model.fit(test_loader, epochs=1, lr=0.001)
