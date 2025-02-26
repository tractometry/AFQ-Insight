import numpy as np
import pytest
import torch

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import Decoder, VariationalEncoder
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


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, input_shape, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = VariationalEncoder(input_shape, latent_dims, dropout)
        self.decoder = Decoder(input_shape, latent_dims)

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)

        z, mu, logvar, kl = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def fit(self, train_loader, epochs=1, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for x, _ in train_loader:
                batch_size = x.size(0)
                num_tracts = x.size(1)
                tract_indices = np.random.randint(0, num_tracts, size=batch_size)
                batch_indices = np.arange(batch_size)
                tract_data = x[batch_indices, tract_indices, :]
                tract_data = tract_data.flatten(start_dim=1)

                optimizer.zero_grad()
                z, mu, logvar, kl = self.encoder(tract_data)
                x_hat = self.decoder(z)

                recon_loss = torch.nn.functional.mse_loss(x_hat, tract_data)
                loss = recon_loss + kl

                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


@pytest.mark.parametrize("model_class", [VariationalAutoencoder])
@pytest.mark.parametrize("latent_dims", [2, 10])
def test_autoencoder_forward(
    data_loaders, data_shapes, model_class, latent_dims, device
):
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    model = model_class(input_shape=in_channels, latent_dims=latent_dims, dropout=0.1)
    model.to(device)
    model.eval()

    data_iter = iter(test_loader)
    x, _ = next(data_iter)
    x = x.to(device)

    batch_size = x.size(0)
    num_tracts = x.size(1)

    tract_indices = np.random.randint(0, num_tracts, size=batch_size)
    batch_indices = np.arange(batch_size)

    tract_data = x[batch_indices, tract_indices, :]
    tract_data = tract_data.to(device)

    with torch.no_grad():
        output = model(tract_data)

    expected_shape = (batch_size, in_channels)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


@pytest.mark.parametrize("model_class", [VariationalAutoencoder])
def test_autoencoder_train_loop(data_loaders, data_shapes, model_class, device):
    """
    Simple test for the training loop of the Autoencoder models,
    checking for any exceptions.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    model = model_class(input_shape=in_channels, latent_dims=10, dropout=0.1)
    model.to(device)
    model.train()

    try:
        model.fit(train_loader, epochs=1, lr=0.001)
    except Exception as e:
        pytest.fail(f"Model {model_class.__name__} failed with exception: {e}")


@pytest.mark.parametrize("latent_dims", [2, 10, 20])
def test_variational_encoder_outputs(data_loaders, data_shapes, latent_dims, device):
    """
    Test that the variational encoder returns the expected number of outputs
    with the correct shapes.
    """
    torch_dataset, train_loader, test_loader, val_loader = data_loaders
    gt_shape, sequence_length, in_channels = data_shapes

    encoder = VariationalEncoder(
        input_shape=in_channels, latent_dims=latent_dims, dropout=0.1
    )
    encoder.to(device)
    encoder.eval()

    data_iter = iter(test_loader)
    x, _ = next(data_iter)
    x = x.to(device)

    batch_size = x.size(0)
    num_tracts = x.size(1)

    tract_indices = np.random.randint(0, num_tracts, size=batch_size)
    batch_indices = np.arange(batch_size)

    tract_data = x[batch_indices, tract_indices, :]
    tract_data = tract_data.to(device)

    with torch.no_grad():
        z, mu, logvar, kl = encoder(tract_data)

    assert isinstance(z, torch.Tensor), "z should be a tensor"
    assert isinstance(mu, torch.Tensor), "mu should be a tensor"
    assert isinstance(logvar, torch.Tensor), "logvar should be a tensor"
    assert isinstance(kl, torch.Tensor), "kl should be a tensor"

    assert z.shape == (
        batch_size,
        latent_dims,
    ), f"Expected z shape {(batch_size, latent_dims)}, got {z.shape}"
    assert mu.shape == (
        batch_size,
        latent_dims,
    ), f"Expected mu shape {(batch_size, latent_dims)}, got {mu.shape}"
    assert logvar.shape == (
        batch_size,
        latent_dims,
    ), f"Expected logvar shape {(batch_size, latent_dims)}, got {logvar.shape}"
    assert kl.dim() == 0, f"KL should be a scalar tensor, got shape {kl.shape}"
