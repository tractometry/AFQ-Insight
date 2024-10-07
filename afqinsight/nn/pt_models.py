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

torch, has_torch, _ = optional_package("torch", trip_msg=torch_msg)
if has_torch:
    import torch.nn as nn
else:
    # Since all model building functions start with Input, we make Input the
    # tripwire instance for cases where pytorch is not installed.
    Input = TripWire(torch_msg)
    print("test")

class mlp4(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation=torch.softmax, verbose=False):
        super(mlp4, self).__init__()
        
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
            nn.Linear(500, n_classes)
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
    
def MLP4(input_shape, n_classes):
    mlp4 = mlp4(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return mlp4
    
class cnn_lenet(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")
        
        #idea is to create a list that sequential can use to
            #create the model with the order of layers?
        conv_layers = []

        for i in range(self.n_conv_layers): 
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
                        in_channels= input_shape[1],
                        out_channels=6,
                        kernel_size=3,
                        padding=1, # because padding in tensorflow is 'same' and strides = 1
                    )
                )
            else:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
                        in_channels= 6 + 10 * i,
                        out_channels=6 + 10 * i,
                        kernel_size=3,
                        padding=1, # because padding in tensorflow is 'same' and strides = 1
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
        cnn_lenet = cnn_lenet(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
        return cnn_lenet

class cnn_vgg(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(cnn_vgg, self).__init__()
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")
        
        #idea is to create a list that sequential can use to
            #create the model with the order of layers?
        conv_layers = []

        for i in range(self.n_conv_layers):
            num_filters = min(64 * 2 ** i, 512)
            if i == 0:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
                        in_channels= input_shape[1],
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1, # because padding in tensorflow is 'same' and strides = 1
                    )
                )
            else:
                conv_layers.append(
                    nn.Conv1d(
                        # what is in channels?
                        in_channels= num_filters,
                        out_channels=num_filters,
                        kernel_size=3,
                        padding=1, # because padding in tensorflow is 'same' and strides = 1
                    )
                )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                padding=1
            ))
            conv_layers.append(nn.ReLU())
            if i > 1:
                conv_layers.append(nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=3,
                    padding=1
                ))
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
    cnn_vgg = cnn_vgg(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return cnn_vgg
    
class lstm1v0(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(lstm1v0, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 512, batch_first=True),
            nn.Linear(512, n_classes)
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
    lstm1v0 = lstm1v0(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return lstm1v0
    
class lstm1(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):   
        super(lstm1, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
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
    lstm1 = lstm1(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return lstm1
    
class lstm2(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(lstm2, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100),
            nn.ReLU(),
            nn.LSTM(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
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
    lstm2 = lstm2(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return lstm2
    
class blstm1(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(blstm1, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
            nn.ReLU(),
            nn.LSTM(100, 100, bidirectional=True),
            nn.ReLU(),
            nn.Linear(100, n_classes)
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
    blstm1 = blstm1(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return blstm1
    
class blstm2(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(blstm2, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_shape[1], 100, bidirectional=True),
            nn.ReLU(),
            nn.LSTM(100, 100, bidirectional=True),
            nn.ReLU(),
            nn.Linear(100, n_classes)
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
    blstm2 = blstm2(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return blstm2

class lstm_fcn(nn.Module):  
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
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
            nn.Linear(256, n_classes)
        )
        
        if output_activation == torch.softmax:
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.model(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

def lstm_fcn(input_shape, n_classes):
    lstm_fcn = lstm_fcn(input_shape, n_classes, output_activation = torch.softmax, verbose=False)
    return lstm_fcn

class cnn_resnet(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation='softmax'):
        super(cnn_resnet, self).__init__()
        steps, features = input_shape 

        self.model = nn.Sequential(
            for i, nb_nodes in enumerate([64, 128, 128]):
                nn.Conv1d(input_shape[1], 128, 8, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
           

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(128, n_classes),
        )

        if output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.model(x)
        x = self.output_activation(x)
        return x

'''
data = torch.Tensor(numpy_array)
data.shape == (batch, in)
'''