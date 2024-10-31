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

        self.model = nn.Sequential(
            nn.LSTM(input_shape[0], 128),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Conv1d(input_shape[1], 128, 8, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(256, n_classes),
        )

        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
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
        conv_layers = []
        in_channel = input_shape[1]

        for i, nb_nodes in enumerate([64, 128, 128]):
            conv_layers.append(nn.Conv1d(in_channel, nb_nodes, 8, padding=1))
            in_channel = nb_nodes
            conv_layers.append(nn.BatchNorm1d(128))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv1d(in_channel, nb_nodes, 5, padding=1))
            conv_layers.append(nn.BatchNorm1d(128))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv1d(in_channel, nb_nodes, 3, padding=1))
            conv_layers.append(nn.BatchNorm1d(128))
            conv_layers.append(nn.ReLU())
            if i < 2:
                conv_layers.append(nn.Conv1d(in_channel, nb_nodes, 1, padding=1))
            conv_layers.append(nn.BatchNorm1d(128))
            conv_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            *conv_layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(128, n_classes),
        )

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
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
