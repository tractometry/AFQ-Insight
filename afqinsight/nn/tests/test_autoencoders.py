import pytest
import torch

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import Conv1DVariationalAutoencoder, VariationalAutoencoder
from afqinsight.nn.utils import prep_pytorch_data


@pytest.fixture
def device():
    """Sets up the computing device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dataset():
    """Loads the AFQ dataset."""
    return AFQDataset.from_study("hbn")


@pytest.fixture
def data_loaders(dataset):
    """Prepare PyTorch datasets and data loaders."""
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
    "model_class", [VariationalAutoencoder, Conv1DVariationalAutoencoder]
)
def test_vae_training(data_loaders, data_shapes, model_class, device):
    """Test that the VAE can be trained without errors."""
    _, num_tracts, input_shape = data_shapes

    _, train_loader, _, _ = data_loaders

    if model_class == VariationalAutoencoder:
        model = model_class(input_shape=input_shape, latent_dims=10, dropout=0.1)
    else:
        model = model_class(num_tracts=num_tracts, latent_dims=10, dropout=0.1)
    model.to(device)

    try:
        model.fit(train_loader, epochs=1, lr=0.001)
    except Exception as e:
        pytest.fail(f"VAE training failed with exception: {e}")
