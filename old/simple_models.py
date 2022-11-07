import time
from typing import Union
from typing_extensions import TypedDict
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

import numpy as np
import pandas as pd
from sklearn import linear_model as sk_linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.pipeline import Pipeline
import torch 
import library.data.data_helper_functions as data_helpers
import matplotlib.pyplot as plt
import mapie
from sklearn.utils import parallel_backend

from IPython.display import display, display_latex

args = {'batch_size' : 1,
        'data_type' : torch.float,
        'validation_fraction' : 0.3,
        'seed' : 10,
        'device' : torch.device('cpu')}
X, y = torch.load('Library/Data/Real_data_training.pt', map_location=torch.device("cpu"))
X = X.to(dtype=args['data_type'])
y = y.to(dtype=args['data_type'])

num_batches = X.shape[0]

val_batches = int(args['validation_fraction'] * num_batches)
train_batches = num_batches - val_batches

data = data_helpers.MyDataset(X, y, device=args['device'], data_type=args['data_type'])
torch.random.manual_seed(args['seed'])
 
train_dataset, val_dataset = torch.utils.data.dataset.random_split(data, [train_batches, val_batches])

params = {'batch_size': args['batch_size'],
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

train_mean = ytrain.mean(0)
err_train_mean = train_mean - ytrain
err_val_mean = train_mean - yval
print('Train MSE')
print((err_train_mean**2).mean(0))
print((err_train_mean**2).mean())
print('Validation MSE')
print((err_val_mean**2).mean(0))
print(np.sqrt((err_val_mean**2).mean()))

constant_ME_train = np.sqrt((err_train_mean**2).mean(0))
constant_ME_val = np.sqrt((err_val_mean**2).mean(0))


# Linear model

scalar = preprocessing.StandardScaler()
scalar.fit(Atrain)
Atrain_trans = scalar.transform(Atrain)
Aval_trans = scalar.transform(Aval)
reg = sk_linear_model.LinearRegression()
reg.fit(Atrain_trans, ytrain)

yhat_train = reg.predict(Atrain_trans)
yhat_val = reg.predict(Aval_trans)

err_train = yhat_train - ytrain
err_val = yhat_val - yval
print('Train ME')
print(np.sqrt((err_train**2).mean(0)))
print(np.sqrt((err_train**2).mean()))
print('Validation ME')
print(np.sqrt((err_val**2).mean(0)))
print(np.sqrt((err_val**2).mean()))

linear_ME_train = np.sqrt((err_train**2).mean(0))
linear_ME_val = np.sqrt((err_val**2).mean(0))


# Tree models

regr_tree_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_tree_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_tree_3 = tree.DecisionTreeRegressor(max_depth=8)

regr_tree_1.fit(Atrain, ytrain)
regr_tree_2.fit(Atrain, ytrain)
regr_tree_3.fit(Atrain, ytrain)


yhat_train_tree_1 = regr_tree_1.predict(Atrain)
yhat_train_tree_2 = regr_tree_2.predict(Atrain)
yhat_train_tree_3 = regr_tree_3.predict(Atrain)
yhat_val_tree_1 = regr_tree_1.predict(Aval)
yhat_val_tree_2 = regr_tree_2.predict(Aval)
yhat_val_tree_3 = regr_tree_3.predict(Aval)

err_train_tree_1 = yhat_train_tree_1 - ytrain
err_train_tree_2 = yhat_train_tree_2 - ytrain
err_train_tree_3 = yhat_train_tree_3 - ytrain
err_val_tree_1 = yhat_val_tree_1 - yval
err_val_tree_2 = yhat_val_tree_2 - yval
err_val_tree_3 = yhat_val_tree_3 - yval

tree1_ME_train = np.sqrt((err_train_tree_1**2).mean(0))
tree2_ME_train = np.sqrt((err_train_tree_2**2).mean(0))
tree3_ME_train = np.sqrt((err_train_tree_3**2).mean(0))
tree1_ME_val = np.sqrt((err_val_tree_1**2).mean(0))
tree2_ME_val = np.sqrt((err_val_tree_2**2).mean(0))
tree3_ME_val = np.sqrt((err_val_tree_3**2).mean(0))


adaboost_ME_train, adaboost_ME_val = [], []
random_forest_ME_train, random_forest_ME_val = [], []
gradboost_ME_train, gradboost_ME_val = [], []

with parallel_backend('loky', n_jobs=-3):
    for k in range(24):
        print('Attribute:', k)
        
        adaboost_1 = ensemble.AdaBoostRegressor(n_estimators=20)

        adaboost_1.fit(Atrain, ytrain[:,k])

        yhat_train_adaboost_1 = adaboost_1.predict(Atrain)
        yhat_val_adaboost_1 = adaboost_1.predict(Aval)

        err_train_adaboost_1 = yhat_train_adaboost_1 - ytrain[:,k]
        err_val_adaboost_1 = yhat_val_adaboost_1 - yval[:,k]

        adaboost_ME_train.append(np.sqrt((err_train_adaboost_1**2).mean(0)))
        adaboost_ME_val.append(np.sqrt((err_val_adaboost_1**2).mean(0)))

        random_forest_2 = ensemble.RandomForestRegressor(n_estimators=200, max_samples=5/200)

        random_forest_2.fit(Atrain, ytrain[:,k])

        yhat_train_forest_2 = random_forest_2.predict(Atrain)
        yhat_val_forest_2 = random_forest_2.predict(Aval)

        err_train_forest_2 = yhat_train_forest_2 - ytrain[:,k]
        err_val_forest_2 = yhat_val_forest_2 - yval[:,k]

        random_forest_ME_train.append(np.sqrt((err_train_forest_2**2).mean(0)))
        random_forest_ME_val.append(np.sqrt((err_val_forest_2**2).mean(0)))

        gradboost_1 = ensemble.HistGradientBoostingRegressor(loss='squared_error')

        gradboost_1.fit(Atrain, ytrain[:, k])

        yhat_train_gradboost_1 = gradboost_1.predict(Atrain)
        yhat_val_gradboost_1 = gradboost_1.predict(Aval)

        err_train_gradboost_1 = yhat_train_gradboost_1 - ytrain[:,k]
        err_val_gradboost_1 = yhat_val_gradboost_1 - yval[:,k]

        gradboost_ME_train.append(np.sqrt((err_train_gradboost_1**2).mean(0)))
        gradboost_ME_val.append(np.sqrt((err_val_gradboost_1**2).mean(0)))



df_train_simple_current = pd.DataFrame({
    'Constant (mean)' : constant_ME_train,
    'Linear' : linear_ME_train,
    'Tree 1' : tree1_ME_train,
    'Tree 2' : tree2_ME_train,
    'Tree 3' : tree3_ME_train,
    'Random forest (5)' : random_forest_ME_train,
    'Adaboost (20)' : adaboost_ME_train,
    'Gradboost' : gradboost_ME_train
})

df_val_simple_current = pd.DataFrame({
    'Constant (mean)' : constant_ME_val,
    'Linear' : linear_ME_val,
    'Tree (5)' : tree2_ME_val, 
    'Random forest (200)' : random_forest_ME_val,
    'Adaboost (20)' : adaboost_ME_val,
    'Gradboost' : gradboost_ME_val
}, index=['Scartamento [mm]', 'ScartMedio [mm]', 'Curvatura [1/m]', 'Sopraelev. [mm]', 'VarSopraelev. [mm]', 'ScartStdDev [mm]', 
    'ScartLivTrasv [mm]', 'LivTrasvStdDev [mm]', 'Sghembo3m [°/oo]', 'Sghembo9m [°/oo]', 'LivLongSxD1 [mm]', 'LivLongDxD1 [mm]',
    'AllinSxD1 [mm]', 'AllinDxD1 [mm]', 'AllinStdDev [mm]', 'LivLongStdDev [mm]', 'LivLongSxD2 [mm]', 'LivLongDxD2 [mm]', 'AllinSxD2 [mm]', 
    'AllinDxD2 [mm]', 'LivLongSxD3 [mm]', 'LivLongDxD3 [mm]', 'AllinSxD3 [mm]', 'AllinDxD3 [mm]'])
df_train_simple_current.index = df_val_simple_current.index

display(df_val_simple_current)

df_train_simple_current.to_csv('df_train_simple_current.csv')
df_val_simple_current.to_csv('df_val_simple_current.csv')


# In[14]:


df_val_simple_current.to_latex(float_format="%.3g")


# In[23]:


model = Pipeline([('linear',sk_linear_model.LinearRegression())])

Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
STRATEGIES = {
    "cv": Params(method="base", cv=3),
    "cv_plus": Params(method="plus", cv=3),
    "cv_minmax": Params(method="minmax", cv=3),
    "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=50)),
    "jackknife_minmax_ab": Params(method="minmax", cv=Subsample(n_resamplings=50)),
}
y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    print(strategy)
    mapie = MapieRegressor(model, **params)
    mapie.fit(Atrain, ytrain[:,0])
    y_pred[strategy], y_pis[strategy] = mapie.predict(Aval, alpha=0.05)


# In[ ]:




