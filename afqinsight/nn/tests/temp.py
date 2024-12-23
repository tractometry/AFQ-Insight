import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from afqinsight import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data

plt.rcParams["figure.dpi"] = 200


torch.manual_seed(0)
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps"


# %%
class ConvEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 6 * 13, latent_dim)  # Latent space mean
        self.fc_logvar = nn.Linear(
            128 * 6 * 13, latent_dim
        )  # Latent space log variance

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# %%
class ConvDecoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(
            latent_dim, 128 * 6 * 13
        )  # Map latent space back to flattened feature map

        # Define transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=(2, 2), padding=(1, 1), output_padding=(1, 0)
        )  # [6, 13] -> [12, 25]
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=(2, 2), padding=(1, 1), output_padding=(1, 1)
        )  # [12, 25] -> [24, 50]
        self.deconv3 = nn.ConvTranspose2d(
            32,
            input_channels,
            kernel_size=3,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )  # [24, 50] -> [48, 100]

    def forward(self, z):
        # Fully connected layer
        x = self.fc(z)
        x = x.view(
            z.size(0), 128, 6, 13
        )  # Reshape to match ConvTranspose2D input dimensions
        # print(f"After reshape: {x.shape}")

        # Deconvolution layers
        x = F.relu(self.deconv1(x))
        # print(f"After deconv1: {x.shape}")  # Should be [64, 64, 12, 25]
        x = F.relu(self.deconv2(x))
        # print(f"After deconv2: {x.shape}")  # Should be [64, 32, 24, 50]
        x = torch.sigmoid(self.deconv3(x))  # Normalize to [0, 1]
        # print(f"After deconv3: {x.shape}")  # Should be [64, 1, 48, 100]

        return x


# %%
class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(input_channels, latent_dim)
        self.decoder = ConvDecoder(input_channels, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # Reparameterization trick
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


# %%
def train(autoencoder, data_loader, epochs=10, lr=1e-3, beta=0.001):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    autoencoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        for x, _ in data_loader:
            x = x.to(device)

            optimizer.zero_grad()

            # Forward pass
            x_reconstructed, mu, logvar = autoencoder(x)

            # Compute reconstruction loss
            recon_loss = F.mse_loss(x_reconstructed, x, reduction="sum")

            # Compute KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            total_loss = recon_loss + beta * kl_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Total Loss: {epoch_loss / len(data_loader)}, "
            f"Recon Loss: {recon_loss_total / len(data_loader)}, "
            f" KL Loss: {kl_loss_total / len(data_loader)}"
        )
    return autoencoder


# %%
dataset = AFQDataset.from_study("hbn")
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(
    dataset, batch_size=64
)
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100


# %%


# %%
print(dataset)
print(gt_shape, sequence_length, in_channels)
print(torch_dataset[1866][0].size())

# %%
# latent_dims = 100
# input_shape = sequence_length * in_channels
# autoencoder = Autoencoder(input_shape, latent_dims).to(device) # GPU

# # data = torch.utils.data.DataLoader(
# #         torchvision.datasets.MNIST('./data',
# #                transform=torchvision.transforms.ToTensor(),
# #                download=True),
# #         batch_size=128,
# #         shuffle=True)

# autoencoder = train(autoencoder, train_loader)

# %%
print(train_loader.dataset[0][0].size())

# %%
latent_dims = 20
in_channels = 1

vae = ConvAutoencoder(in_channels, latent_dims).to(device)  # GPU
vae = train(vae, train_loader, epochs=20, lr=0.001)

# %%
output = vae.forward(test_loader.dataset[0][0].unsqueeze(0).to(device))
output2 = vae.forward(test_loader.dataset[1][0].unsqueeze(0).to(device))
output3 = vae.forward(test_loader.dataset[2][0].unsqueeze(0).to(device))
print(output[0].shape)


# %%
plt.plot(output[0].cpu().detach().numpy().flatten()[0:100])
plt.plot(dataset.X[0].flatten()[0:100])
plt.plot(output2[0].cpu().detach().numpy().flatten()[0:100])
plt.plot(dataset.X[1].flatten()[0:100])


# %%
def plot_latent(vautoencoder, data_loader, target_index, num_batches=100):
    plt.figure(figsize=(8, 6))

    for i, (x, y) in enumerate(data_loader):
        print(x.shape, y.shape)
        z = vautoencoder.encoder(x.to(device)).to("cpu").detach().numpy()
        print("Latent space shape:", z.shape)

        target = y[:, target_index].to("cpu").numpy()  # Shape: (batch_size,)
        print("Target column:", target.shape)

        plt.scatter(z[:, 0], z[:, 1], c=target, cmap="viridis", alpha=0.5)

        if i >= num_batches:
            break

    plt.colorbar(label=f"Target: {target_index}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"Latent Space Visualization (Target Index: {target_index})")

    plt.show()


# %%
plot_latent(vae, train_loader, 0)

# %%
plot_latent(vae, train_loader, 1)

# %%
plot_latent(vae, train_loader, 2)
