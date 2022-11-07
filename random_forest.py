import time
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn import ensemble
import torch 
import library.data.data_helper_functions as data_helpers
from sklearn.utils import parallel_backend

class ARGS(object):
    def __init__(self):
        self.batch_size = 1
        self.data_type = torch.float
        self.validation_fraction = 0.3
        self.seed = 10
        self.device = torch.device('cpu')
        self.data_folder = '/work1/s174505/Thesis/Data/'
        self.training_data_file = 'Real_data_training.pt'
        self.debugging = False
        self.use_cv = False
args = ARGS()

train_dataset, val_dataset = data_helpers.load_data_training(args)

params = {'batch_size': args.batch_size,
        'shuffle': False,
        'drop_last': False}
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
val_loader = torch.utils.data.DataLoader(val_dataset, **params)

train_X = [X[0] for X,_ in train_loader]
train_y = [y[0] for _,y in train_loader]
val_X = [X[0] for X,_ in val_loader]
val_y = [y[0] for _,y in val_loader]

Atrain = torch.cat(train_X,axis=1).permute(1,0).numpy()
ytrain = torch.cat(train_y,axis=1).permute(1,0).numpy()
Aval = torch.cat(val_X,axis=1).permute(1,0).numpy()
yval = torch.cat(val_y,axis=1).permute(1,0).numpy()

n_attributes = ytrain.shape[1]

res_train_me = {}
res_train_max = {}
res_train_q95 = {}
res_val_me = {}
res_val_max = {}
res_val_q95 = {}

if len(sys.argv) > 1:
    attributes = [int(i)-1 for i in sys.argv[1:]]
    save = False
else:
    attributes = range(n_attributes)
    save = True

with parallel_backend('loky', n_jobs=-1):
        for factor in [100]:#[10, 50, 100, 500]:
                for n_trees in [5000]: # [100, 500, 1000, 5000]
                        print('# trees:', n_trees)
                        attrs_train_ME = []
                        attrs_val_ME = []
                        for k in attributes:
                                s = time.time()
                                print('Attribute', k)

                                random_forest = ensemble.RandomForestRegressor(
                                        n_estimators=n_trees, 
                                        max_samples=min(factor/n_trees,1.))
                                random_forest.fit(Atrain, ytrain[:,k])
                                
                                yhat_train_forest = random_forest.predict(Atrain)
                                yhat_val_forest = random_forest.predict(Aval)
                                
                                err_train_forest = yhat_train_forest - ytrain[:,k]
                                err_val_forest = yhat_val_forest - yval[:,k]
                                np.save(f'results/random_forest/rf_best_train_{k}', err_train_forest)
                                np.save(f'results/random_forest/rf_best_val_{k}', err_val_forest)
                                attr_train_ME = np.mean(np.abs(err_train_forest))
                                attr_val_ME = np.mean(np.abs(err_val_forest))
                                print('Train ME:     ', attr_train_ME)
                                print('Validation ME:', attr_val_ME)
                                attrs_train_ME.append(attr_train_ME)
                                attrs_val_ME.append(attr_val_ME)
                                print('Time:', time.time() - s)
                                s = time.time()
                                
                                
                                
                        res_train_me[f"# trees {n_trees}"] = attrs_train_ME
                        res_val_me[f"# trees {n_trees}"] = attrs_val_ME

                df_train_me = pd.DataFrame(res_train_me)
                df_val_me = pd.DataFrame(res_val_me)
                if save:
                    df_train_me.to_pickle(f'results/random_forest/Random_forest_train_results_large_{factor}.pkl')
                    df_val_me.to_pickle(f'results/random_forest/Random_forest_val_results_large_{factor}.pkl')
