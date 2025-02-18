import numpy as np
import pytest
import torch

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import (
    Autoencoder,
    VAE_random_tracts,
    VariationalAutoencoder,
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
    """Fixture to prepare PyTorch datasets and data lsoaders."""
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


@pytest.mark.parametrize(
    "model_class", [Autoencoder, VariationalAutoencoder, VAE_random_tracts]
)
@pytest.mark.parametrize("latent_dims", [2, 10])
def test_autoencoder_forward(data_loaders, data_shapes, model_class, latent_dims):
    """
    Smoke test to check if the forward pass of the Autoencoder models works
    without throwing exceptions and returns the expected shape.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    # Instantiate the model
    if model_class == VAE_random_tracts:
        model = model_class(
            input_shape=in_channels, latent_dims=latent_dims, dropout=0.1
        )
    elif model_class == VariationalAutoencoder or model_class == Autoencoder:
        model = model_class(
            input_shape=sequence_length * in_channels, latent_dims=latent_dims
        )
    model.eval()  # Forward pass check, no training

    # Retrieve a single batch
    data_iter = iter(test_loader)
    x, _ = next(data_iter)

    batch_size = x.size(0)
    num_tracts = x.size(1)

    # Randomly select one tract per batch element
    tract_indices = np.random.randint(0, num_tracts, size=batch_size)
    batch_indices = np.arange(batch_size)

    # Extract the randomly chosen tracts
    tract_data = x[batch_indices, tract_indices, :]  # Shape: [batch_size, 100]

    # Add a channel dimension for consistency
    tract_data = tract_data.unsqueeze(1)  # Shape: [batch_size, 1, 100]

    # Perform forward pass
    with torch.no_grad():
        if model_class == VAE_random_tracts:
            output = model(tract_data)
        elif model_class == VariationalAutoencoder or model_class == Autoencoder:
            output = model(x)

    # Validate output shape
    if model_class == VAE_random_tracts:
        expected_shape = (x.size(0), in_channels)
    elif model_class == VariationalAutoencoder or model_class == Autoencoder:
        expected_shape = (x.size(0), sequence_length, in_channels)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}."


@pytest.mark.parametrize(
    "model_class", [Autoencoder, VariationalAutoencoder, VAE_random_tracts]
)
def test_autoencoder_train_loop(data_loaders, data_shapes, model_class):
    """
    Simple smoke test for the training loop of the Autoencoder models,
    checking for any exceptions.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    if model_class == VAE_random_tracts:
        model = model_class(input_shape=in_channels, latent_dims=10, dropout=0.1)
    elif model_class == VariationalAutoencoder or model_class == Autoencoder:
        model = model_class(input_shape=sequence_length * in_channels, latent_dims=10)
    model.train()

    try:
        if model_class == VAE_random_tracts:
            model.fit(train_loader, epochs=1, lr=0.001)
        elif model_class == VariationalAutoencoder or model_class == Autoencoder:
            model.fit(test_loader, epochs=1, lr=0.001)
    except Exception as e:
        pytest.fail(f"Model {model_class.__name__} failed with exception: {e}")
