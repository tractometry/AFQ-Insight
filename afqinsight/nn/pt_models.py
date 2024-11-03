import math

import torch
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWire

torch_msg = (
    "To use afqinsight's pytorch models, you need to have pytorch "
    "installed. You can do this by installing afqinsight with `pip install "
    "afqinsight[tf]`, or by separately installing pytorch with `pip install "
    "pytorch`."
)

torch, has_torch, _ = optional_package("torch", trip_msg=torch_msg)  # noqa F811
if has_torch:
    import torch.nn as nn
    import torch.nn.functional as F
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
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(CNN_VGG, self).__init__()
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")

        conv_layers = []

        for i in range(self.n_conv_layers):
            num_filters = min(64 * 2**i, 512)
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=input_shape[1],
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1,
                    )
                )
            else:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=num_filters,
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1,
                    )
                )
            conv_layers.append(nn.ReLU())
            conv_layers.append(
                nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=3,
                    padding=1,
                )
            )
            conv_layers.append(nn.ReLU())
            if i > 1:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=num_filters,
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1,
                    )
                )
            conv_layers.append(nn.MaxPool1d(kernel_size=2))

        self.model = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
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


def cnn_vgg_pt(input_shape, n_classes):
    cnn_vgg_Model = CNN_VGG(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return cnn_vgg_Model


class LSTM1V0(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(LSTM1V0, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 512, batch_first=True), nn.Linear(512, n_classes)
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
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100), nn.ReLU(), nn.Linear(100, n_classes)
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
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
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
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
            nn.ReLU(),
            nn.Linear(200, n_classes),
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
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
            nn.ReLU(),
            nn.LSTM(200, 100, bidirectional=True),
            nn.ReLU(),
            nn.Linear(200, n_classes),
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
        # Process CNN branch first if that's desired
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

        # Process LSTM branch second if that's the arrangement
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.dropout(x_lstm)
        x_lstm = x_lstm[:, -1, :]  # Select the last output of the LSTM layer

        # Concatenate the outputs
        x = torch.cat((x_conv, x_lstm), dim=1)

        # Final dense layer and activation
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

        length, channels = input_shape

        self.blocks = nn.ModuleList()
        self.res_convs = nn.ModuleList()

        num_filters = [64, 128, 128]
        in_channels = channels

        for i, nb_nodes in enumerate(num_filters):
            block = nn.Sequential(
                nn.Conv1d(in_channels, nb_nodes, kernel_size=8, padding=3),
                nn.BatchNorm1d(nb_nodes),
                nn.ReLU(inplace=True),
                nn.Conv1d(nb_nodes, nb_nodes, kernel_size=5, padding=2),
                nn.BatchNorm1d(nb_nodes),
                nn.ReLU(inplace=True),
                nn.Conv1d(nb_nodes, nb_nodes, kernel_size=3, padding=1),
                nn.BatchNorm1d(nb_nodes),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)

            if i < 2:
                res_conv = nn.Sequential(
                    nn.Conv1d(in_channels, nb_nodes, kernel_size=1, padding=0),
                    nn.BatchNorm1d(nb_nodes),
                )
                self.res_convs.append(res_conv)
            else:
                self.res_convs.append(None)

            in_channels = nb_nodes

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(num_filters[-1], n_classes)

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = x

        for i, block in enumerate(self.blocks):
            conv = block(x)

            if self.res_convs[i] is not None:
                residual = self.res_convs[i](x)
            conv += residual

            conv = F.relu(conv)

            residual = conv
            x = conv

        x = self.global_avg_pool(conv)
        x = x.squeeze(-1)

        x = self.fc(x)

        if self.output_activation:
            x = self.output_activation(x)
        return x


def cnn_resnet_pt(input_shape, n_classes):
    cnn_resnet_Model = CNN_RESNET(input_shape, n_classes, output_activation="softmax")
    return cnn_resnet_Model


"""
data = torch.Tensor(numpy_array)
data.shape == (batch, in)
"""
