from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Conv1d, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d, ConvTranspose1d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax, interpolate


class base_cnn(nn.Module):
    def __init__(self, hyperparameter_dict):
        super(base_cnn, self).__init__()

        # Load hyperparameters
        self.in_features = hyperparameter_dict["in_features"]
        self.out_features = hyperparameter_dict["out_features"]
        
        self.strides = hyperparameter_dict["strides"]
        self.kernel_lenghts = hyperparameter_dict["kernel_lenghts"]
        self.channels = [self.in_features] + hyperparameter_dict["channels"] + [self.out_features] 
        self.paddings = hyperparameter_dict["paddings"]

        self.use_dropout = hyperparameter_dict['use_dropout']
        self.dropout = hyperparameter_dict['dropout']
        self.use_batch_normalization = hyperparameter_dict["use_batch_normalization"]
        self.activation_function = hyperparameter_dict["activation_function"]
        act_func = nn.ReLU()
        if self.activation_function == "elu":
            act_func = nn.ELU()
        elif self.activation_function == "leakyrelu":
            act_func = nn.LeakyReLU()
        elif self.activation_function == "rrelu":
            act_func = nn.RReLU()
        
        # Check input
        if len(self.strides) != len(self.kernel_lenghts) != len(self.channels)-1 != len(self.paddings):
            print("ERROR: stides kernel_lenghts channels paddings length does not match")
        self.num_conv_layers = len(self.strides)
        
        # Batch normalization layer
        self.normalize_input = BatchNorm1d(self.in_features)
        
        # Define hidden convolutional layers
        self.conv_layers = []

        # Dropout on input
        if self.use_dropout:
            self.conv_layers.append(Dropout(self.dropout[0]))

        for i in range(self.num_conv_layers-1):
            self.conv_layers.append(Conv1d(in_channels=self.channels[i], 
                                           out_channels=self.channels[i+1], 
                                           kernel_size=self.kernel_lenghts[i],
                                           stride=self.strides[i], 
                                           padding=self.paddings[i]))
            
            if self.use_batch_normalization:
                self.conv_layers.append(BatchNorm1d(self.channels[i+1]))
            
            if self.activation_function == "prelu":
                self.conv_layers.append(nn.PReLU(self.channels[i+1]))
            else:
                self.conv_layers.append(act_func)

            if self.use_dropout:
                self.conv_layers.append(Dropout(self.dropout[i+1]))
        
        self.conv_hidden_layers = nn.Sequential(*self.conv_layers)

        # Output conv layer
        self.conv_out_layer = Conv1d(in_channels=self.channels[-2], 
                                     out_channels=self.out_features, 
                                     kernel_size=self.kernel_lenghts[-1],
                                     stride=self.strides[-1], 
                                     padding=self.paddings[-1])

    def forward(self, x):
        x = self.normalize_input(x)
        x = self.conv_hidden_layers(x)
        x = self.conv_out_layer(x)
        return x

class pinn_cnn(base_cnn):
    def __init__(self, hyperparameter_dict, states, irregularities):
        super(base_cnn, self).__init__(hyperparameter_dict)
        self.states = states # u from ode
        self.irregularities = irregularities

    def get_data(self,):
        """ Get the data matrix for each time
        """
        pass