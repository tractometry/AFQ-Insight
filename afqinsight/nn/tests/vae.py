import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from afqinsight import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data

torch.manual_seed(0)


plt.rcParams["figure.dpi"] = 200

device = "cuda" if torch.cuda.is_available() else "cpu"


class VariationalEncoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_shape, 500)
        self.linear2 = nn.Linear(500, latent_dims)
        self.linear3 = nn.Linear(500, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Linear(500, input_shape),
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = self.model(z)
        return x.view((batch_size, 48, 100))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

    def forward(self, x):
        print(x.shape)
        z = self.encoder(x)
        print(z.shape)
        print(self.decoder(z).shape)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for x, y in data:
            print(y.shape)
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder


dataset = AFQDataset.from_study("hbn")
torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(dataset)
gt_shape = torch_dataset[0][1].size()[0]
sequence_length = torch_dataset[0][0].size()[0]  # 48
in_channels = torch_dataset[0][0].size()[1]  # 100

latent_dims = 2  # 2 dimension latent space, better for plotting?
input_shape = sequence_length * in_channels

vae = VariationalAutoencoder(input_shape, latent_dims).to(device)
vae = train(vae, train_loader)


def plot_latent(vautoencoder, data_loader, target_index, num_batches=100):
    plt.figure(figsize=(8, 6))

    for i, (x, y) in enumerate(data_loader):
        print(x.shape, y.shape)
        z = vautoencoder.encoder(x.to(device)).to("cpu").detach().numpy()
        print("Latent space shape:", z.shape)

        target = y[:, target_index].to("cpu").numpy()
        print("Target column:", target.shape)

        plt.scatter(z[:, 0], z[:, 1], c=target, cmap="viridis", alpha=0.5)

        if i >= num_batches:
            break

    plt.colorbar(label=f"Target: {target_index}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"Latent Space Visualization (Target Index: {target_index})")

    plt.show()


# this gets column 1 of the target, which is age
plot_latent(vae, train_loader, 0)

# this should get column 2 of the target, which should be sex
plot_latent(vae, train_loader, 1)

# this should get column 3 of the target, which is scan site id
plot_latent(vae, train_loader, 2)
