import math

import numpy as np
import torch
import torch.nn.functional as F
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire

import afqinsight.augmentation as aug

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
            print(self.model)

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
            print(f"Pooling layers: {self.n_conv_layers}")

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


class VariationalEncoder_one_tract(nn.Module):
    def __init__(self, input_shape, latent_dims, dropout):
        super(VariationalEncoder_one_tract, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_shape, 50)
        self.linear2 = nn.Linear(50, latent_dims)
        self.linear3 = nn.Linear(50, latent_dims)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder_one_tract(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder_one_tract, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, input_shape)

    def forward(self, z):
        batch_size = z.size(0)
        x = self.linear1(z)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x.view(batch_size, -1)


class VariationalEncoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_shape, 500)
        self.linear2 = nn.Linear(500, latent_dims)
        self.linear3 = nn.Linear(500, latent_dims)
        self.activation = nn.ReLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_shape, 500)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(500, latent_dims)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 500)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(500, input_shape)

    def forward(self, z):
        batch_size = z.size(0)
        x = self.linear1(z)
        x = self.relu(x)
        x = self.linear2(x)
        return x.view((batch_size, 48, 100))


class Conv1dEncoder(nn.Module):
    def __init__(self, in_channels=48, latent_channels=12):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, latent_channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            latent_channels * 2, latent_channels, kernel_size=3, stride=1, padding=1
        )
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        return x


class Conv1dDecoder(nn.Module):
    def __init__(self, in_channels=12, out_channels=48):
        super().__init__()

        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose1d(
            in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        return x


class Conv1DEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        # Input shape: [batch, channels=48, length=50]
        self.conv1 = nn.Conv1d(
            in_channels=48, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=latent_dims, kernel_size=3, stride=2, padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class Conv1DDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=latent_dims,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=0,
        )
        # change output padding back to 1
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.dropout(x)
        x = self.relu(self.deconv2(x))
        x = self.dropout(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)
        return x


class VAE_multiple_tract(nn.Module):
    def __init__(self, input_shape, latent_dims, dropout):
        super(VAE_multiple_tract, self).__init__()
        self.encoder = VariationalEncoder_one_tract(
            input_shape, latent_dims, dropout=dropout
        )
        self.decoder = Decoder_one_tract(input_shape, latent_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def fit(self, data, epochs=20, lr=0.001, num_selected_tracts=5, sigma=0.03):
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0
            items = 0
            for x, _ in data:
                batch_size = x.size(0)  # 64
                num_tracts = x.size(1)  # 48

                # By the end, will have shape (batch_size, num_selected_tracts, 100)
                selected_tracts = []
                for _ in range(num_selected_tracts):
                    tract_indices = np.random.randint(0, num_tracts, size=batch_size)
                    batch_indices = np.arange(batch_size)
                    selected_tracts.append(x[batch_indices, tract_indices, :])

                tract_data = torch.stack(selected_tracts, dim=1)

                tract_data = tract_data.to(torch.float32).numpy()
                tract_data = aug.jitter(tract_data, sigma=sigma)
                tract_data = torch.tensor(tract_data, dtype=torch.float32).to(
                    self.device
                )

                tract_data = tract_data.view(-1, 100).to(self.device)

                opt.zero_grad()
                x_hat = self(tract_data)

                loss = F.mse_loss(tract_data, x_hat, reduction="sum")

                items += tract_data.size(0)
                running_loss += loss.item()
                loss.backward()
                opt.step()

            print(f"Epoch {epoch+1}, Loss: {running_loss/items:.2f}")

        return self

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def fit(self, data, epochs=20, lr=0.001):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0
            items = 0
            for x, _ in data:
                x = x.to(self.device)  # GPU
                opt.zero_grad()
                x_hat = self(x)
                loss = ((x - x_hat) ** 2).sum() + self.encoder.kl
                items += x.size(0)
                running_loss += loss.item()
                loss.backward()
                opt.step()
            print(f"Epoch {epoch+1}, Loss: {running_loss/items:.2f}")

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def fit(self, data, epochs=20, lr=0.001):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0
            items = 0
            for x, _ in data:
                x = x.to(self.device)  # GPU
                opt.zero_grad()
                x_hat = self(x)
                loss = ((x - x_hat) ** 2).sum()
                items += x.size(0)
                running_loss += loss.item()
                loss.backward()
                opt.step()
            print(f"Epoch {epoch+1}, Loss: {running_loss/items:.2f}")

        return self

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class Conv1dAutoencoder(nn.Module):
    def __init__(self, in_channels=48, latent_channels=12):
        super().__init__()
        self.encoder = Conv1dEncoder(
            in_channels=in_channels, latent_channels=latent_channels
        )
        self.decoder = Conv1dDecoder(
            out_channels=in_channels, in_channels=latent_channels
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def fit(
        self,
        model,
        train_loader,
        val_loader,
        patience=10,
        lr_patience=3,
        epochs=5,
        lr=1e-3,
        use_lr_scheduler=False,
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=lr_patience, verbose=True
            )
        loss_fn = nn.MSELoss()
        num_no_improve = 0
        best_val_loss = 10**10

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_items = 0
            for x, _ in train_loader:
                x = x.to(self.device)

                x_hat = model(x)
                loss = loss_fn(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_items += x.size(0)
            avg_loss = epoch_loss / num_items

            print(
                f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.6f},"
                f"lr = {optimizer.param_groups[0]['lr']}"
            )

            model.eval()
            val_loss = 0.0
            num_items = 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(self.device)
                    x_hat = model(x)
                    loss = loss_fn(x_hat, x).item()
                    val_loss += loss
                    num_items += x.size(0)

            val_loss /= num_items
            if use_lr_scheduler:
                scheduler.step(val_loss)

            if val_loss > best_val_loss:
                num_no_improve += 1
            else:
                num_no_improve = 0
                best_val_loss = val_loss

            if num_no_improve != 0:
                print(
                    f"Validation Loss = {val_loss:.6f},"
                    f"No improvement for {num_no_improve} epochs"
                )
            else:
                print(f"Validation Loss = {val_loss:.6f}")

            if num_no_improve == patience:
                print("Early stopping")
                break

        return model

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class VAE_one_tract(nn.Module):
    def __init__(self, input_shape, latent_dims, dropout):
        super(VAE_one_tract, self).__init__()
        self.encoder = VariationalEncoder_one_tract(
            input_shape, latent_dims, dropout=dropout
        )
        self.decoder = Decoder_one_tract(input_shape, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def fit(self, data, epochs=20, lr=0.001):
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0
            items = 0
            for x, _ in data:  # x shape: (batch_size, 48, 100)
                tract_data = (
                    x[:, 0, :].to(torch.float32).to(device)
                )  # Shape: (batch_size, 100)

                opt.zero_grad()
                x_hat = self(tract_data).to(device)

                loss = reconstruction_loss(tract_data, x_hat, kl_div=0, reduction="sum")

                items += tract_data.size(0)
                running_loss += loss.item()
                loss.backward()
                opt.step()

            print(f"Epoch {epoch+1}, Loss: {running_loss/items:.2f}")

        return self

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


class Conv1DAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        self.encoder = Conv1DEncoder_fa(latent_dims, dropout)
        self.decoder = Conv1DDecoder_fa(latent_dims, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime

    def fit(self, data, epochs=20, lr=0.001):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=10, factor=0.5, verbose=True
        )

        self.to(device)

        for epoch in range(epochs):
            running_loss = 0
            items = 0
            self.train()
            for x, _ in data:  # x shape: [batch, 48, 50]
                x = x.to(torch.float32).to(device)
                opt.zero_grad()
                x_hat = self(x)
                loss = reconstruction_loss(x, x_hat, kl_div=0, reduction="sum")
                loss.backward()
                opt.step()
                items += x.size(0)
                running_loss += loss.item()

            avg_loss = running_loss / items
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        return self

    def transform(self, x):
        self.forward(x)

    def fit_transform(self, data, epochs=20):
        self.fit(data, epochs)
        return self.transform(data)


"""
data = torch.Tensor(numpy_array)
data.shape == (batch, in)
"""
