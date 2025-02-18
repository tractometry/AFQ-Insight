import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils

import afqinsight.augmentation as aug

torch.manual_seed(0)
plt.rcParams["figure.dpi"] = 200


# rename this to be train_first_tract_dropout_experiment
def train_first_tract_dropout_experiment(
    self, train_data, val_data, epochs=20, lr=0.001, num_selected_tracts=5, sigma=0.03
):
    opt = torch.optim.Adam(self.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=5, factor=0.5
    )
    train_rmse_per_epoch = []
    val_rmse_per_epoch = []

    for epoch in range(epochs):
        self.train()
        running_loss = 0
        running_rmse = 0
        items = 0

        for x, _ in train_data:  # x shape: (batch_size, 48, 100)
            tract_data = x[:, 0, :]

            tract_data = tract_data.to(torch.float32).numpy()
            tract_data = aug.jitter(tract_data, sigma=sigma)
            tract_data = torch.tensor(tract_data, dtype=torch.float32).to(device)

            opt.zero_grad()
            x_hat = self(tract_data)

            loss = reconstruction_loss(tract_data, x_hat, kl_div=0, reduction="sum")

            batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

            items += tract_data.size(0)
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * tract_data.size(0)
            loss.backward()
            opt.step()
        scheduler.step(running_loss / items)

        avg_train_rmse = running_rmse / items
        train_rmse_per_epoch.append(avg_train_rmse)

        self.eval()
        val_rmse = 0
        val_items = 0

        with torch.no_grad():
            for x, _ in val_data:
                tract_data = x[:, 0, :]

                tract_data = tract_data.to(torch.float32).numpy()
                tract_data = aug.jitter(tract_data, sigma=sigma)
                tract_data = torch.tensor(tract_data, dtype=torch.float32).to(device)

                x_hat = self(tract_data)

                batch_val_rmse = torch.sqrt(
                    F.mse_loss(tract_data, x_hat, reduction="mean")
                )

                val_items += tract_data.size(0)
                val_rmse += batch_val_rmse.item() * tract_data.size(0)

        avg_val_rmse = val_rmse / val_items
        val_rmse_per_epoch.append(avg_val_rmse)

        print(
            f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f},"
            f"Val RMSE: {avg_val_rmse:.4f}"
        )

    return train_rmse_per_epoch, val_rmse_per_epoch


def train_first_tract_latent_experiment(
    train_data, val_data, epochs=100, lr=0.001, sigma=0.03
):
    latent_dims = list(range(10, 101, 10))  # [10, 20, ..., 100]
    results = []

    for latent_dim in latent_dims:
        print(f"Training Autoencoder with Latent Dimension: {latent_dim}")

        model = Autoencoder(input_dim=100, latent_dim=latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=5, factor=0.5
        )

        train_rmse_per_epoch = []
        val_rmse_per_epoch = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            running_rmse = 0
            items = 0

            for x, _ in train_data:  # x shape: (batch_size, 48, 100)
                tract_data = x[:, 0, :].to(device)  # Shape: (batch_size, 100)

                opt.zero_grad()
                x_hat = model(tract_data)

                loss = reconstruction_loss(tract_data, x_hat, kl_div=0, reduction="sum")

                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

                items += tract_data.size(0)
                running_loss += loss.item()
                running_rmse += batch_rmse.item() * tract_data.size(0)
                loss.backward()
                opt.step()

            scheduler.step(running_loss / items)

            avg_train_rmse = running_rmse / items
            train_rmse_per_epoch.append(avg_train_rmse)

            model.eval()
            val_rmse = 0
            val_items = 0

            with torch.no_grad():
                for x, _ in val_data:
                    tract_data = x[:, 0, :].to(device)

                    x_hat = model(tract_data)

                    batch_val_rmse = torch.sqrt(
                        F.mse_loss(tract_data, x_hat, reduction="mean")
                    )

                    val_items += tract_data.size(0)
                    val_rmse += batch_val_rmse.item() * tract_data.size(0)

            avg_val_rmse = val_rmse / val_items
            val_rmse_per_epoch.append(avg_val_rmse)

            print(
                f"Latent Dim {latent_dim} - Epoch {epoch+1}, Train RMSE:"
                f"{avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}"
            )

        results.append(
            {
                "latent_dim": latent_dim,
                "train_rmse_per_epoch": train_rmse_per_epoch,
                "val_rmse_per_epoch": val_rmse_per_epoch,
            }
        )

    return results
