import os
import torch

data_folder = '/work1/s174505/Thesis/Data/'
(X_training, y_training) = torch.load(os.path.join(data_folder, 'Real_data_training.pt'))
coefs = torch.randn((y_training.shape[1], X_training.shape[1]))
y_dummy = torch.matmul(coefs, X_training)

torch.save((X_training, y_dummy, 'linear', coefs), os.path.join(data_folder, 'Debugging_data.pt'))