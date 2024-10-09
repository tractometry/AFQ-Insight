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
    # Since all model building functions start with Input, we make Input the
    # tripwire instance for cases where pytorch is not installed.
    Input = TripWire(torch_msg)
    print("test")


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


def mlp4(input_shape, n_classes):
    mlp4_Model = mlp4(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return mlp4_Model


class CNN_LENET(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")

        conv_layers = []

        for i in range(self.n_conv_layers):
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=input_shape[1],
                        out_channels=6,
                        kernel_size=3,
                        padding=1,
                    )
                )
            else:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=6 + 10 * i,
                        out_channels=6 + 10 * i,
                        kernel_size=3,
                        padding=1,
                    )
                )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))

        self.model = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(input_shape, 120),
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

    def cnn_lenet(input_shape, n_classes):
        cnn_lenet_Model = cnn_lenet(
            input_shape, n_classes, output_activation=torch.softmax, verbose=False
        )
        return cnn_lenet_Model


class CNN_VGG(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(cnn_vgg, self).__init__()
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")

        # idea is to create a list that sequential can use to
        # create the model with the order of layers?
        conv_layers = []

        for i in range(self.n_conv_layers):
            num_filters = min(64 * 2**i, 512)
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
                        in_channels=input_shape[1],
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1,
                    )
                )
            else:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
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
            nn.Linear(input_shape, 4096),
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


def cnn_vgg(input_shape, n_classes):
    cnn_vgg_Model = cnn_vgg(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return cnn_vgg_Model


class LSTM1V0(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(lstm1v0, self).__init__()
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


def lstm1v0(input_shape, n_classes):
    lstm1v0_Model = lstm1v0(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm1v0_Model


class LSTM1(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(lstm1, self).__init__()
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


def lstm1(input_shape, n_classes):
    lstm1_Model = lstm1(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm1_Model


class LSTM2(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(lstm2, self).__init__()
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


def lstm2(input_shape, n_classes):
    lstm2_Model = lstm2(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return lstm2_Model


class BLSTM1(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(blstm1, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
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


def blstm1(input_shape, n_classes):
    blstm1_Model = blstm1(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return blstm1_Model


class BLSTM2(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(blstm2, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
            nn.ReLU(),
            nn.LSTM(100, 100, bidirectional=True),
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


def blstm2(input_shape, n_classes):
    blstm2_Model = blstm2(
        input_shape, n_classes, output_activation=torch.softmax, verbose=False
    )
    return blstm2_Model


class LSTM_FCN(nn.Module):
    def __init__(
        self, input_shape, n_classes, output_activation=torch.softmax, verbose=False
    ):
        super(lstm_fcn, self).__init__()

        self.model = nn.Sequential(
            # what is the input shape, confused how permute translates
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


def lstm_fcn(input_shape, n_classes):
    lstm_fcn_Model = lstm_fcn(
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

    def cnn_resnet(input_shape, n_classes):
        cnn_resnet_Model = CNN_RESNET(
            input_shape, n_classes, output_activation="softmax"
        )
        return cnn_resnet_Model


"""
data = torch.Tensor(numpy_array)
data.shape == (batch, in)
"""
