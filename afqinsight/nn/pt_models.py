import math
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

class mlp4_torch(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        super(mlp4_torch, self).__init__()
        self.flatten = nn.flatten()
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.3)

        self.l0 = nn.Linear(input_shape, 500)
        self.l1 = nn.ReLU(500,500)
        self.l2 = nn.ReLU(500,500)
        self.l3 = nn.ReLU(500,500)
        self.l4 = nn.softmax(500,n_classes)

        self.relu = nn.relu()
        
        if output_activation == torch.softmax:
            self.output = nn.softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout_1(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.dropout_2(x)
        x = self.l2(x)
        x = self.dropout_2(x)
        x = self.l3(x)
        x = self.dropout_3(x)
        x = self.l4(x)
        x = self.output(x)

        return x
    
class cnn_lenet_torch(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")
        
        #idea is to create a list that sequential can use to
            #create the model with the order of layers?
        conv_layers = []

        for i in range(self.n_conv_layers):
            conv_layers.append(
                nn.Conv1d(
                    # what is in channels?
                    in_channels= input_shape[i],
                    out_channels=6 + 10 * i,
                    kernel_size=3,
                    padding=1, # because padding in tensorflow is 'same' and strides = 1
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.flatten = nn.flatten()
        self.fc1 = nn.ReLU(input_shape, 120)
        self.fc2 = nn.ReLU(120, 84)
        self.fc3 = nn.softmax(84, n_classes)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

        if output_activation == torch.softmax:
            self.output = nn.softmax()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x) 
       
        x = self.fc1(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        x = self.dropout_2(x)
        x = self.fc3(x)  
        x = self.output(x)
        return x

class cnn_vgg_torch(nn.Module):
    def __init__(self, input_shape, n_classes, output_activation = torch.softmax, verbose=False):
        self.n_conv_layers = int(round(math.log(input_shape[0], 2)) - 3)
        if verbose:
            print(f"Pooling layers: {self.n_conv_layers}")
        
        #idea is to create a list that sequential can use to
            #create the model with the order of layers?
        conv_layers = []

        for i in range(self.n_conv_layers):
            num_filters = min(64 * 2 ** i, 512)
            conv_layers.append(
                nn.Conv1d(
                    # what is in channels?
                    in_channels= input_shape[i],
                    out_channels=num_filters,
                    kernel_size=3,
                    padding=1, # because padding in tensorflow is 'same' and strides = 1
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv1d(
                in_channels=6 + 10 * i,
                out_channels=num_filters,
                kernel_size=3,
                padding=1
            ))
            conv_layers.append(nn.ReLU())
            if i > 1:
                conv_layers.append(nn.Conv1d(
                    in_channels=6 + 10 * i,
                    out_channels=num_filters,
                    kernel_size=3,
                    padding=1
                ))
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.flatten = nn.flatten()
        self.fc1 = nn.ReLU(input_shape, 4096)
        self.fc2 = nn.ReLU(4096, 4096)
        self.fc3 = nn.softmax(4096, n_classes)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

        if output_activation == torch.softmax:
            self.output = nn.softmax()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x) 
       
        x = self.fc1(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        x = self.dropout_2(x)
        x = self.fc3(x)  
        x = self.output(x)
        return x
'''
data = torch.Tensor(numpy_array)
data.shape == (batch, in
'''