import logging
import math

import torch
import torch.nn.functional as F
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire

from afqinsight.nn.utils import (
    vae_loss,
)

torch_msg = (
    "To use afqinsight's pytorch models, you need to have pytorch "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[tf]`, or by separately installing pytorch with `pip install "
    "pytorch`."
)

torch, has_torch, _ = optional_package("torch", trip_msg=torch_msg)  # noqa F811
if has_torch:
    import torch.nn as nn
else:
    Input = TripWire(torch_msg)


class MLP4(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(MLP4, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(input_shape, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, n_classes),
        )

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

        if verbose:
            logging.info(self.model)

    def forward(self, x):
        x = self.model(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x


def mlp4_pt(input_shape, n_classes):
    mlp4_Model = MLP4(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return mlp4_Model


class CNN_LENET(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(CNN_LENET, self).__init__()
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            logging.info(f"Pooling layers: {self.n_conv_layers}")

        conv_layers = []
        seq_len = input_shape[0]
        in_channels = input_shape[1]

        for i in range(self.n_conv_layers):
            if i == 0:
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=6,
                    kernel_size=3,
                    padding=1,
                )
            else:
                conv = nn.Conv1d(
                    in_channels=6 + 10 * (i - 1),
                    out_channels=6 + 10 * i,
                    kernel_size=3,
                    padding=1,
                )
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            seq_len = math.floor((seq_len + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1)
            seq_len = math.floor(seq_len / 2)

        final_channels = 6 + 10 * (self.n_conv_layers - 1)

        in_features = final_channels * seq_len

        self.model = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(in_features, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, n_classes),
        )

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x = self.model(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x


def cnn_lenet_pt(input_shape, n_classes):
    cnn_lenet_Model = CNN_LENET(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return cnn_lenet_Model


class CNN_VGG(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation):
        super(CNN_VGG, self).__init__()

        n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)

        conv_layers = []
        in_channels = input_shape[0]

        for i in range(n_conv_layers):
            num_filters = min(64 * 2**i, 512)

            conv_layers.append(
                nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))

            conv_layers.append(
                nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))

            if i > 1:
                conv_layers.append(
                    nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
                )
                conv_layers.append(nn.ReLU(inplace=True))

            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = num_filters

        self.features = nn.Sequential(*conv_layers)

        def _compute_feature_size(input_shape):
            test_input = torch.zeros(1, *input_shape)
            with torch.no_grad():
                features = self.features(test_input)
                return features.numel()

        feature_size = _compute_feature_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
        )

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)

        features = self.features(x)

        output = self.classifier(features)

        output = self.output_activation(output)

        return output


def cnn_vgg_pt(input_shape, n_classes):
    cnn_vgg_Model = CNN_VGG(
        input_shape,
        n_classes,
        output_activation=torch.softmax,
    )
    return cnn_vgg_Model


class LSTM1V0(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(LSTM1V0, self).__init__()
        self.lstm = nn.LSTM(input_shape[1], 512, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        if self.output_activation:
            x = self.output_activation(x)
        return x


def lstm1v0_pt(input_shape, n_classes):
    lstm1v0_Model = LSTM1V0(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm1v0_Model


class LSTM1(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTM(input_shape[1], 100, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x


def lstm1_pt(input_shape, n_classes):
    lstm1_Model = LSTM1(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm1_Model


class LSTM2(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(LSTM2, self).__init__()
        self.lstm1 = nn.LSTM(input_shape[1], 100, batch_first=True)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(100, 100, batch_first=True)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(100, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu1(x)

        x, _ = self.lstm2(x)
        x = self.relu2(x)

        x = self.fc(x[:, -1, :])

        if self.output_activation:
            x = self.output_activation(x)
        return x


def lstm2_pt(input_shape, n_classes):
    lstm2_Model = LSTM2(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm2_Model


class BLSTM1(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(BLSTM1, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=100,
            bidirectional=True,
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(200, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.relu(x)
        x = self.fc(x[:, -1, :])
        if self.output_activation:
            x = self.output_activation(x)
        return x


def blstm1_pt(input_shape, n_classes):
    blstm1_Model = BLSTM1(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return blstm1_Model


class BLSTM2(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(BLSTM2, self).__init__()
        self.blstm1 = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=100,
            bidirectional=True,
            batch_first=True,
        )
        self.relu1 = nn.ReLU()

        self.blstm2 = nn.LSTM(
            input_size=200, hidden_size=100, bidirectional=True, batch_first=True
        )
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(200, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x, _ = self.blstm1(x)
        x = self.relu1(x)

        x, _ = self.blstm2(x)
        x = self.relu2(x)

        x = self.fc(x[:, -1, :])

        if self.output_activation:
            x = self.output_activation(x)
        return x


def blstm2_pt(input_shape, n_classes):
    blstm2_Model = BLSTM2(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return blstm2_Model


class LSTM_FCN(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(LSTM_FCN, self).__init__()

        timesteps, features = input_shape

        self.lstm = nn.LSTM(input_size=timesteps, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        self.conv1 = nn.Conv1d(
            in_channels=features, out_channels=128, kernel_size=8, padding=4
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128 + 128, n_classes)

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu1(x_conv)

        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = self.relu2(x_conv)

        x_conv = self.conv3(x_conv)
        x_conv = self.bn3(x_conv)
        x_conv = self.relu3(x_conv)

        x_conv = self.gap(x_conv)
        x_conv = x_conv.squeeze(-1)

        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.dropout(x_lstm)
        x_lstm = x_lstm[:, -1, :]

        x = torch.cat((x_conv, x_lstm), dim=1)

        x = self.fc(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x


def lstm_fcn_pt(input_shape, n_classes):
    lstm_fcn_Model = LSTM_FCN(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm_fcn_Model


class CNN_RESNET(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation="softmax"):
        super(CNN_RESNET, self).__init__()

        in_channels = input_shape[0]

        self.layers = nn.ModuleList(
            [
                ResNetBlock(in_channels, 64),
                ResNetBlock(64, 128),
                ResNetBlock(128, 128),
            ]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, n_classes)

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.global_avg_pool(x).squeeze(-1)

        x = self.fc(x)

        x = self.output_activation(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm1d(out_channels),
        )

        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv_path(x)

        x = x + residual

        x = self.final_relu(x)

        return x


def cnn_resnet_pt(input_shape, n_classes):
    cnn_resnet_Model = CNN_RESNET(input_shape, n_classes, output_activation="softmax")
    return cnn_resnet_Model


class BaseEncoder(nn.Module):
    def __init__(self, input_shape=100, latent_dims=20, dropout=0.2):
        super(BaseEncoder, self).__init__()
        self.linear1 = nn.Linear(input_shape, 50)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.latent_dims = latent_dims

    def _encode_base(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Encoder(BaseEncoder):
    def __init__(self, input_shape=100, latent_dims=20, dropout=0.2):
        super(Encoder, self).__init__(input_shape, latent_dims, dropout)
        self.linear2 = nn.Linear(50, latent_dims)

    def forward(self, x):
        x = self._encode_base(x)
        x = self.linear2(x)
        return x


class VariationalEncoder(BaseEncoder):
    def __init__(self, input_shape=100, latent_dims=20, dropout=0.2):
        super(VariationalEncoder, self).__init__(input_shape, latent_dims, dropout)
        self.mean = nn.Linear(50, latent_dims)
        self.logvar = nn.Linear(50, latent_dims)

    def forward(self, x):
        x = self._encode_base(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, input_shape=100, latent_dims=20):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, input_shape)

    def forward(self, z):
        x = self.linear1(z)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class BaseConv1DEncoder(nn.Module):
    def __init__(self, num_tracts=48, latent_dims=20, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_tracts, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.latent_dims = latent_dims

    def _encode_base(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        return x


class Conv1DEncoder(BaseConv1DEncoder):
    def __init__(self, num_tracts=48, latent_dims=20, dropout=0.2):
        super().__init__(num_tracts, latent_dims, dropout)
        self.conv4 = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self._encode_base(x)
        x = self.conv4(x)
        return x


class Conv1DVariationalEncoder(BaseConv1DEncoder):
    def __init__(self, num_tracts=48, latent_dims=20, dropout=0.2):
        super().__init__(num_tracts, latent_dims, dropout)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(64 * 13, latent_dims)
        self.fc_logvar = nn.Linear(64 * 13, latent_dims)

    def forward(self, x):
        x = self._encode_base(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Conv1DVariationalDecoder(nn.Module):
    def __init__(self, num_tracts=48, latent_dims=20):
        super().__init__()
        self.fc = nn.Linear(latent_dims, 64 * 13)

        self.deconv1 = nn.ConvTranspose1d(
            latent_dims, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose1d(
            32, 16, kernel_size=4, stride=2, padding=2, output_padding=1
        )
        self.deconv4 = nn.ConvTranspose1d(
            16, num_tracts, kernel_size=3, stride=2, padding=2, output_padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, 64, 13)
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class Conv1DDecoder(nn.Module):
    def __init__(self, num_tracts=48, latent_dims=20):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(
            latent_dims, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            64, 32, kernel_size=3, stride=2, padding=2, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose1d(
            32, 16, kernel_size=4, stride=2, padding=2, output_padding=1
        )
        self.deconv4 = nn.ConvTranspose1d(
            16, num_tracts, kernel_size=3, stride=2, padding=2, output_padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape=100, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = VariationalEncoder(input_shape, latent_dims, dropout=dropout)
        self.decoder = Decoder(input_shape, latent_dims)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def forward(self, x):
        mean, logvar = self.encoder(x)

        z = self.reparameterize(mean, logvar)

        x_hat = self.decoder(z)

        return x_hat, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def fit(self, train_data, epochs=500, lr=0.001, kl_weight=0.001):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=5, factor=0.5
        )

        train_rmse_per_epoch = []
        train_kl_per_epoch = []
        train_recon_per_epoch = []

        # This is the weight for the KL "annealing"
        beta_start = 0.0
        beta_end = 1.0
        slope = (beta_end - beta_start) / epochs

        for epoch in range(epochs):
            running_loss = 0.0
            running_rmse = 0.0
            running_kl = 0.0
            running_recon_loss = 0.0
            items = 0

            # The weight on KL changes linearly with epoch:
            beta = beta_start + slope * epoch

            for x, _ in train_data:
                batch_size = x.size(0)
                tract_data = x.to(self.device)

                opt.zero_grad()

                x_hat, mean, logvar = self(tract_data)

                loss, recon_loss, kl_loss = vae_loss(
                    tract_data, x_hat, mean, logvar, beta, reduction="sum"
                )

                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

                loss.backward()
                opt.step()

                items += batch_size
                running_loss += loss.item()
                running_rmse += batch_rmse.item() * batch_size
                running_kl += kl_loss.item()
                running_recon_loss += recon_loss.item()

            scheduler.step(running_loss / items)
            avg_train_rmse = running_rmse / items
            avg_train_kl = running_kl / items
            avg_train_recon_loss = running_recon_loss / items

            train_rmse_per_epoch.append(avg_train_rmse)
            train_kl_per_epoch.append(avg_train_kl)
            train_recon_per_epoch.append(avg_train_recon_loss)

            logging.info(
                f"""Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, KL:
                {avg_train_kl:.4f}, Recon Loss: {avg_train_recon_loss:.4f}"""
            )

        self.history = {
            "train_rmse_per_epoch": train_rmse_per_epoch,
            "train_kl_per_epoch": train_kl_per_epoch,
            "train_recon_per_epoch": train_recon_per_epoch,
        }

        return self

    def transform(self, x):
        self.eval()

        if isinstance(x, torch.utils.data.DataLoader):
            latent_vectors = []

            with torch.no_grad():
                for batch, _ in x:
                    batch = batch.to(torch.float32).to(self.device)
                    mean, _ = self.encoder(batch)
                    latent_vectors.append(mean.cpu())

            if latent_vectors:
                return torch.cat(latent_vectors, dim=0)
            return None

        else:
            with torch.no_grad():
                x = x.to(torch.float32).to(self.device)
                mean, _ = self.encoder(x)
                return mean

    def fit_transform(self, data, epochs=20, kl_weight=0.001):
        self.fit(data, epochs, kl_weight=kl_weight)
        return self.transform(data)


class Autoencoder(nn.Module):
    def __init__(self, input_shape=100, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(input_shape, latent_dims, dropout=dropout)
        self.decoder = Decoder(input_shape, latent_dims)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def fit(self, train_data, epochs=500, lr=0.001):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=5, factor=0.5
        )

        train_loss_per_epoch = []
        train_rmse_per_epoch = []
        train_recon_per_epoch = []

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_rmse = 0.0
            running_recon_loss = 0.0
            items = 0

            for x, _ in train_data:
                batch_size = x.size(0)
                data = x.to(self.device)

                opt.zero_grad()
                x_hat = self(data)

                recon_loss = F.mse_loss(data, x_hat, reduction="sum")
                loss = recon_loss

                batch_rmse = torch.sqrt(F.mse_loss(data, x_hat, reduction="mean"))

                loss.backward()
                opt.step()

                items += batch_size
                running_loss += loss.item()
                running_rmse += batch_rmse.item() * batch_size
                running_recon_loss += recon_loss.item()

            scheduler.step(running_loss / items)
            avg_train_loss = running_loss / items
            avg_train_rmse = running_rmse / items
            avg_train_recon_loss = running_recon_loss / items

            train_loss_per_epoch.append(avg_train_loss)
            train_rmse_per_epoch.append(avg_train_rmse)
            train_recon_per_epoch.append(avg_train_recon_loss)

            logging.info(
                f"""Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f},
                Recon Loss: {avg_train_recon_loss:.4f}"""
            )

        self.history = {
            "train_loss_per_epoch": train_loss_per_epoch,
            "train_rmse_per_epoch": train_rmse_per_epoch,
            "train_recon_per_epoch": train_recon_per_epoch,
        }

        return self

    def transform(self, x):
        self.eval()

        if isinstance(x, torch.utils.data.DataLoader):
            latent_vectors = []
            with torch.no_grad():
                for batch, _ in x:
                    z = self.encoder(batch.to(self.device))
                    latent_vectors.append(z.cpu())

            if latent_vectors:
                return torch.cat(latent_vectors, dim=0)
            return None

        else:
            with torch.no_grad():
                x = x.to(self.device)
                z = self.encoder(x)
                return z

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class Conv1DVariationalAutoencoder(nn.Module):
    def __init__(self, num_tracts=48, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = Conv1DVariationalEncoder(num_tracts, latent_dims, dropout)
        self.decoder = Conv1DVariationalDecoder(num_tracts, latent_dims)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        (
            mean,
            logvar,
        ) = self.encoder(x)

        z = self.reparameterize(mean, logvar)

        x_prime = self.decoder(z)
        return x_prime, mean, logvar

    def fit(self, train_data, epochs=500, lr=0.001, kl_weight=0.001):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=5, factor=0.5
        )

        train_rmse_per_epoch = []
        train_kl_per_epoch = []
        train_recon_per_epoch = []

        beta_start = 0.0
        beta_end = 1.0
        slope = (beta_end - beta_start) / epochs

        for epoch in range(epochs):
            running_loss = 0.0
            running_rmse = 0.0
            running_kl = 0.0
            running_recon_loss = 0.0
            items = 0

            beta = beta_start + slope * epoch

            for x, _ in train_data:
                batch_size = x.size(0)
                tract_data = x.to(self.device)

                opt.zero_grad()

                x_hat, mean, logvar = self(tract_data)

                loss, recon_loss, kl_loss = vae_loss(
                    tract_data, x_hat, mean, logvar, beta, reduction="sum"
                )

                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

                loss.backward()
                opt.step()

                items += batch_size
                running_loss += loss.item()
                running_rmse += batch_rmse.item() * batch_size
                running_kl += kl_loss.item()
                running_recon_loss += recon_loss.item()

            scheduler.step(running_loss / items)
            avg_train_rmse = running_rmse / items
            avg_train_kl = running_kl / items
            avg_train_recon_loss = running_recon_loss / items

            train_rmse_per_epoch.append(avg_train_rmse)
            train_kl_per_epoch.append(avg_train_kl)
            train_recon_per_epoch.append(avg_train_recon_loss)

            logging.info(
                f"""Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, KL:
                {avg_train_kl:.4f}, Recon Loss: {avg_train_recon_loss:.4f}"""
            )

        self.history = {
            "train_rmse_per_epoch": train_rmse_per_epoch,
            "train_kl_per_epoch": train_kl_per_epoch,
            "train_recon_per_epoch": train_recon_per_epoch,
        }

        return self

    def transform(self, x):
        self.eval()

        if isinstance(x, torch.utils.data.DataLoader):
            latent_vectors = []

            with torch.no_grad():
                for batch, _ in x:
                    batch = batch.to(torch.float32).to(self.device)
                    mean, _ = self.encoder(batch)
                    latent_vectors.append(mean.cpu())

            if latent_vectors:
                return torch.cat(latent_vectors, dim=0)
            return None

        else:
            with torch.no_grad():
                x = x.to(torch.float32).to(self.device)
                mean, _ = self.encoder(x)
                return mean

    def fit_transform(self, data, epochs=20, kl_weight=0.001):
        self.fit(data, epochs, kl_weight=kl_weight)
        return self.transform(data)


class Conv1DAutoencoder(nn.Module):
    def __init__(self, num_tracts=48, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = Conv1DEncoder(num_tracts, latent_dims, dropout)
        self.decoder = Conv1DDecoder(num_tracts, latent_dims)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime

    def fit(self, train_data, epochs=500, lr=0.001):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=5, factor=0.5
        )

        train_loss_per_epoch = []
        train_rmse_per_epoch = []
        train_recon_per_epoch = []

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_rmse = 0.0
            running_recon_loss = 0.0
            items = 0

            for x, _ in train_data:
                batch_size = x.size(0)
                data = x.to(self.device)

                opt.zero_grad()
                x_hat = self(data)

                recon_loss = F.mse_loss(data, x_hat, reduction="sum")
                loss = recon_loss

                batch_rmse = torch.sqrt(F.mse_loss(data, x_hat, reduction="mean"))

                loss.backward()
                opt.step()

                items += batch_size
                running_loss += loss.item()
                running_rmse += batch_rmse.item() * batch_size
                running_recon_loss += recon_loss.item()

            scheduler.step(running_loss / items)
            avg_train_loss = running_loss / items
            avg_train_rmse = running_rmse / items
            avg_train_recon_loss = running_recon_loss / items

            train_loss_per_epoch.append(avg_train_loss)
            train_rmse_per_epoch.append(avg_train_rmse)
            train_recon_per_epoch.append(avg_train_recon_loss)

            logging.info(
                f"""Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f},
                Recon Loss: {avg_train_recon_loss:.4f}"""
            )

        self.history = {
            "train_loss_per_epoch": train_loss_per_epoch,
            "train_rmse_per_epoch": train_rmse_per_epoch,
            "train_recon_per_epoch": train_recon_per_epoch,
        }

        return self

    def transform(self, x):
        self.eval()

        if isinstance(x, torch.utils.data.DataLoader):
            latent_vectors = []
            with torch.no_grad():
                for batch, _ in x:
                    z = self.encoder(batch.to(self.device))
                    latent_vectors.append(z.cpu())

            if latent_vectors:
                return torch.cat(latent_vectors, dim=0)
            return None

        else:
            with torch.no_grad():
                x = x.to(self.device)
                z = self.encoder(x)
                return z

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


"""
data = torch.Tensor(numpy_array)
data.shape == (batch, in)
"""
