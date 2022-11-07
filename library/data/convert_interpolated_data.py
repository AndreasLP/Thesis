import os
import scipy.io
import torch


data_folder = '/work1/s174505/Thesis/Data/'
data = scipy.io.loadmat(os.path.join(data_folder, 'Interpolated_data.mat'))
data_v2 = scipy.io.loadmat(os.path.join(data_folder, 'Interpolated_data_v2.mat'))

X_training = torch.from_numpy(data['X_train']).to(torch.float64)
y_training = torch.from_numpy(data['y_train']).to(torch.float64)
X_training_v2 = torch.from_numpy(data_v2['X_train']).to(torch.float64)
y_training_v2 = torch.from_numpy(data_v2['y_train']).to(torch.float64)
X_test = torch.from_numpy(data['X_test']).to(torch.float64)
y_test = torch.from_numpy(data['y_test']).to(torch.float64)

torch.save((X_training, y_training), os.path.join(data_folder, 'Interpolated_data_training.pt'))
torch.save((X_training_v2, y_training_v2), os.path.join(data_folder, 'Interpolated_data_v2_training.pt'))
torch.save((X_test, y_test), os.path.join(data_folder, 'Interpolated_data_testing.pt'))