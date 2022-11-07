import os
import scipy.io
import torch
import numpy as np


"""
    "Km                 " "Velocità[km/h]     " "SSCount [n.]       " "Scartamento [mm]   " (0:disregard, 1:disregard, 2:disregard, 3:disregard 
    "ScartMedio [mm]    " "Curvatura [1/m]    " "Sopraelev. [mm]    " "VarSopraelev. [mm] " (4:disregard, 5:disregard, 6:disregard(lin. func of curv), 7:disregard)
    "ScartStdDev [mm]   " "Dif.Sopraelev. [mm]" "ScartLivTrasv [mm] " "LivTrasvStdDev [mm]" (8:disregard, 9:disregard, 10:disregard, 11:disregard)
    "Sghembo3m [°/oo]   " "Sghembo9m [°/oo]   " "LivLongSxD1 [mm]   " "LivLongDxD1 [mm]   " (12:disregard, 13:disregard, 14:keep, 15:keep)
    "AllinSxD1 [mm]     " "AllinDxD1 [mm]     " "LivLongSx10m [mm]  " "LivLongDx10m [mm]  " (16:keep, 17:keep, 18:disregard, 19:disregard)
    "AllinSx10m [mm]    " "AllinDx10m [mm]    " "AllinSxPkPk [mm]   " "AllinDxPkPk [mm]   " (20:disregard, 21:disregard, 22:disregard, 23:disregard)
    "AllinStdDev [mm]   " "LivLongStdDev [mm] " "LivLongSxD2 [mm]   " "LivLongDxD2 [mm]   " (24:disregard, 25:disregard, 26:keep, 27:keep)
    "AllinSxD2 [mm]     " "AllinDxD2 [mm]     " "LivLongSxD3 [mm]   " "LivLongDxD3 [mm]   " (28:keep, 29:keep, 30:keep, 31:keep)
    "AllinSxD3 [mm]     " "AllinDxD3 [mm]     "                                             (32:keep , 33:keep)
"""
data_folder = '/work1/s174505/Thesis/Data/'
geometry = scipy.io.loadmat(os.path.join(data_folder, 'raw_data/20150708_162643_tgipar.mat'))
geometry = torch.from_numpy(geometry['datigeo']).to(torch.float64)

dynamics = scipy.io.loadmat(os.path.join(data_folder, 'raw_data/20150708_163106_dypar.mat'))
dynamics = torch.from_numpy(dynamics['dati']).to(torch.float64)
dynamics = dynamics[~dynamics[:,0].isnan()]

X = dynamics[:, np.r_[1, 3:39, 67]]
y = geometry[:, np.r_[14:18, 26:34]]

# Fix (known) data issues
position = torch.arange(len(geometry[:, 0]))/2
curv = geometry[:, 5]
X[:,-1] = curv
X[128933:128946, 2] = 0
X[216105:216118, 6] = 0

X_training = list()
for xtmp in [x.split(1000) for x in torch.stack((X[23942:93942,:], X[117884:187884,:], X[211826:281826, :]))]:
    X_training.extend(xtmp)
y_training = list()
for ytmp in [y.split(1000) for y in torch.stack((y[23942:93942,:], y[117884:187884,:], y[211826:281826, :]))]:
    y_training.extend(ytmp)

X_training = torch.stack(X_training).permute(0,2,1)
y_training = torch.stack(y_training).permute(0,2,1)

X_test = (X[:23942,:], X[93942:117884,:], X[187884:211826,:])
y_test = (y[:23942,:], y[93942:117884,:], y[187884:211826,:])

X_test = torch.stack(X_test).permute(0,2,1)
y_test = torch.stack(y_test).permute(0,2,1)

torch.save((X, y, position), os.path.join(data_folder, 'Real_data.pt'))
torch.save((X_training, y_training), os.path.join(data_folder, 'Real_data_training.pt'))
torch.save((X_test, y_test), os.path.join(data_folder, 'Real_data_testing.pt'))