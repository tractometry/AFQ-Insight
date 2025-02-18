import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.utils

from afqinsight import AFQDataset
from afqinsight.nn.autoencoder_experiment_utils import (
    train_first_tract_dropout_experiment,
    train_first_tract_latent_experiment,
)
from afqinsight.nn.pt_models import VAE_first_tract
from afqinsight.nn.utils import (
    prep_pytorch_data,
)

torch.manual_seed(0)
plt.rcParams["figure.dpi"] = 200
# Set up for Autoencoder Experiments
dataset = AFQDataset.from_study("hbn")
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(
    dataset, batch_size=64
)
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100
device = "mps"

# Dropout Experiment with the Linear Autoencoder, just looking at the first tract
fig1, fig2, fig3 = run_dropout_experiment(
    train_loader, val_loader, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], in_channels, device
)


def run_dropout_experiment(
    train_loader,
    val_loader,
    dropout_values,
    in_channels,
    device,
    latent_dims=20,
    epochs=100,
    lr=1e-3,
    num_selected_tracts=8,
):
    train_rmse_results = {}
    val_rmse_results = {}

    for dropout in dropout_values:
        print(f"Training with dropout = {dropout}")
        vae_first_tract = VAE_first_tract(
            in_channels, latent_dims=latent_dims, dropout=dropout
        ).to(device)

        train_rmse, val_rmse = train_first_tract_dropout_experiment(
            vae_first_tract,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            num_selected_tracts=num_selected_tracts,
        )
        train_rmse_results[dropout] = train_rmse
        val_rmse_results[dropout] = val_rmse

    fig1 = plt.figure(figsize=(18, 6))
    for dropout in dropout_values:
        plt.plot(
            range(1, len(train_rmse_results[dropout]) + 1),
            train_rmse_results[dropout],
            label=f"Train RMSE (Dropout = {dropout})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training RMSE for Different Dropout Values")
    plt.legend()
    plt.grid()

    fig2 = plt.figure(figsize=(18, 6))
    for dropout in dropout_values:
        plt.plot(
            range(1, len(val_rmse_results[dropout]) + 1),
            val_rmse_results[dropout],
            label=f"Val RMSE (Dropout = {dropout})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE for Different Dropout Values")
    plt.legend()
    plt.grid()

    fig3 = plt.figure(figsize=(18, 8))
    for dropout in dropout_values:
        plt.plot(
            range(1, len(train_rmse_results[dropout]) + 1),
            train_rmse_results[dropout],
            label=f"Train RMSE (Dropout = {dropout})",
        )
        plt.plot(
            range(1, len(val_rmse_results[dropout]) + 1),
            val_rmse_results[dropout],
            linestyle="--",
            label=f"Val RMSE (Dropout = {dropout})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Train vs. Validation RMSE for Different Dropout Values")
    plt.legend()
    plt.grid()

    return fig1, fig2, fig3


# Latent Bottleneck Dimension with the Linear Autoencoder, first tract
latent_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# plot 1, train rmse vs epoch
results = train_first_tract_latent_experiment(train_loader, val_loader)
plt.figure(figsize=(18, 6))
for result in results:
    plt.plot(
        range(1, len(result["train_rmse_per_epoch"]) + 1),
        result["train_rmse_per_epoch"],
        label=f"Latent Dim {result['latent_dim']}",
    )
plt.xlabel("Epoch")
plt.ylabel("Training RMSE")
plt.title("Training RMSE Over Epochs for Different Latent Dimensions")
plt.legend(title="Latent Dimension", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid()
plt.show()

# plot 1, val rmse vs epoch
plt.figure(figsize=(18, 6))
for result in results:
    plt.plot(
        range(1, len(result["val_rmse_per_epoch"]) + 1),
        result["val_rmse_per_epoch"],
        label=f"Latent Dim {result['latent_dim']}",
    )
plt.xlabel("Epoch")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE Over Epochs for Different Latent Dimensions")
plt.legend(title="Latent Dimension", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid()
plt.show()
