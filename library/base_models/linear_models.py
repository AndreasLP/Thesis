import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch.nn import Linear #, Conv1d, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d, ConvTranspose1d
# from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax, interpolate

class constant_model(nn.Module):
    def __init__(self, hyperparameter_dict):
        super(constant_model, self).__init__()
        
        self.out_features = hyperparameter_dict["out_features"]

        self.bias = nn.Parameter(torch.arange(self.out_features).reshape(1,self.out_features,1).to(torch.float))

    def forward(self, x):
        return torch.zeros((x.shape[0],self.out_features,x.shape[2])) + self.bias


class linear_model(nn.Module):
    def __init__(self, hyperparameter_dict):
        super(linear_model, self).__init__()
        
        self.in_features = hyperparameter_dict["in_features"]
        self.out_features = hyperparameter_dict["out_features"]
        # self.burn_in = hyperparameter_dict["burn_in"]
        # self.burn_out = hyperparameter_dict["burn_out"]

        self.linear = Linear(in_features=self.in_features, 
                             out_features=self.out_features, 
                             bias=True)

    def forward(self, x):
        x = x.permute(0,2,1)
        y = self.linear(x).permute(0,2,1)
        return y
