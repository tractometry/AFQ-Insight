import pytest
import torch

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import Conv1DVariationalAutoencoder, VariationalAutoencoder
from afqinsight.nn.utils import prep_first_tract_data, prep_pytorch_data


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
def first_tract_data_loaders(dataset):
    """Prepare PyTorch datasets and data loaders with only the first tract."""
    torch_dataset, train_loader, test_loader, val_loader = prep_first_tract_data(
        dataset
    )
    return torch_dataset, train_loader, test_loader, val_loader


@pytest.fixture
def data_shapes(data_loaders):
    """Fixture to compute shapes for input and target tensors."""
    torch_dataset = data_loaders[0]
    gt_shape = torch_dataset[0][1].size()[0]
    sequence_length = torch_dataset[0][0].size()[0]  # 48
    in_channels = torch_dataset[0][0].size()[1]  # 100
    return gt_shape, sequence_length, in_channels


@pytest.fixture
def first_tract_shapes(first_tract_data_loaders):
    """Fixture to compute shapes for first tract data."""
    sample_batch, _ = next(iter(first_tract_data_loaders[1]))

    if len(sample_batch.shape) == 3:
        input_shape = sample_batch.shape[1] * sample_batch.shape[2]
    else:
        input_shape = sample_batch.shape[1]

    return input_shape


@pytest.mark.parametrize(
    "model_class", [VariationalAutoencoder, Conv1DVariationalAutoencoder]
)
def test_vae_training(
    first_tract_data_loaders, first_tract_shapes, model_class, device
):
    """Test that the VAE can be trained without errors."""
    input_shape = first_tract_shapes

    _, train_loader, _, _ = first_tract_data_loaders

    if model_class == VariationalAutoencoder:
        model = model_class(input_shape=input_shape, latent_dims=10, dropout=0.1)
    else:
        model = model_class(latent_dims=10, dropout=0.1)
    model.to(device)

    try:
        model.fit(train_loader, epochs=1, lr=0.001)
    except Exception as e:
        pytest.fail(f"VAE training failed with exception: {e}")


@pytest.mark.parametrize(
    "model_class", [VariationalAutoencoder, Conv1DVariationalAutoencoder]
)
def test_vae_transform(
    first_tract_data_loaders, first_tract_shapes, model_class, device
):
    """Test the transform method of the Autoencoders."""
    input_shape = first_tract_shapes
    latent_dims = 10

    _, _, test_loader, _ = first_tract_data_loaders

    if model_class == VariationalAutoencoder:
        model = model_class(input_shape=input_shape, latent_dims=10, dropout=0.1)
    else:
        model = model_class(latent_dims=10, dropout=0.1)
    model.to(device)
    model.eval()

    data_iter = iter(test_loader)
    batch, _ = next(data_iter)

    if model_class == VariationalAutoencoder:
        flattened_batch = torch.flatten(batch, start_dim=1)
    elif model_class == Conv1DVariationalAutoencoder:
        flattened_batch = batch

    flattened_batch = flattened_batch.to(device)

    with torch.no_grad():
        z = model.transform(flattened_batch)

    batch_size = flattened_batch.shape[0]
    if model_class == VariationalAutoencoder:
        assert z.shape == (
            batch_size,
            latent_dims,
        ), f"Expected z shape {(batch_size, latent_dims)}, got {z.shape}"
    elif model_class == Conv1DVariationalAutoencoder:
        print("z_shape", z.shape)
        assert z.shape == (
            batch_size,
            latent_dims,
        ), f"Expected z shape {(batch_size, latent_dims)}, got {z.shape}"

    with torch.no_grad():
        transformed = model.transform(test_loader)

    if model_class == VariationalAutoencoder:
        num_samples = len(test_loader.dataset)
        assert transformed.shape == torch.Size(
            (num_samples, 1, latent_dims)
        ), f"Expected transform shape {(num_samples, latent_dims)},"
        f" got {transformed.shape}"
    elif model_class == Conv1DVariationalAutoencoder:
        num_samples = len(test_loader.dataset)
        print(transformed.shape)
        assert transformed.shape == torch.Size(
            (num_samples, latent_dims)
        ), f"Expected transform shape {(num_samples, latent_dims)},"
        f" got {transformed.shape}"
