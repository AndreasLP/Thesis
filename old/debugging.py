import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
import library.functions.misc_helper_functions as misc_helpers
from library.functions import model_parameters
import library.data.data_helper_functions as data_helpers
import time

args = misc_helpers.parse_args()
args.training_data_file = 'Interpolated_data_training.pt'
params = {'batch_size': args.batch_size,
            # 'shuffle': True,
            'drop_last': False}
hyperparameters, model = model_parameters.parameters(args)

torch.random.manual_seed(args.seed)
net = model(hyperparameters).to(device=args.device, dtype=args.data_type)

train_dataset, val_dataset = data_helpers.load_data_training(args=args, net=net)
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
val_loader = torch.utils.data.DataLoader(val_dataset, **params)

X, y = list(train_loader)[0]