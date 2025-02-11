import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.utils

from afqinsight import AFQDataset
from afqinsight.nn.pt_models import VAE_one_tract
from afqinsight.nn.utils import prep_pytorch_data

torch.manual_seed(0)
plt.rcParams["figure.dpi"] = 200

# Set up for experiments
dataset = AFQDataset.from_study("hbn")
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(
    dataset, batch_size=64
)
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100

# Dropout Experiment with the Linear Autoencoder, just looking at the first tract
dropout_values = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
train_rmse_results = {}
val_rmse_results = {}

for dropout in dropout_values:
    print(f"Training with dropout = {dropout}")
    vae_one_tract = VAE_one_tract(in_channels, latent_dims=20, dropout=dropout).to(
        device
    )
    train_rmse, val_rmse = random_train_multiple_tracts_experiment(
        vae_one_tract,
        train_loader,
        val_loader,
        epochs=100,
        lr=1e-3,
        num_selected_tracts=8,
    )
    train_rmse_results[dropout] = train_rmse
    val_rmse_results[dropout] = val_rmse

# Plot 1: Training RMSE only
plt.figure(figsize=(18, 6))
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
plt.show()

# Plot 2: Validation RMSE only
plt.figure(figsize=(18, 6))
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
plt.show()

# Plot 3: Both Training and Validation RMSE
plt.figure(figsize=(18, 8))
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
plt.show()


# Latent Bottleneck Dimension with the Linear Autoencoder, first tract
latent_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# plot 1, train rmse vs epoch
results = train_multiple_latent_dimensions(train_loader, val_loader)
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
