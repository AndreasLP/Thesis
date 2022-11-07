from os.path import join as path_join
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def load_data_training(args, net=None):
    """
    Return two datasets for train and val
    """
    if args.debugging:
        X, y, model_type, model_coef = torch.load(path_join(args.data_folder, args.training_data_file), 
            map_location=torch.device("cpu"))
    else:
        X, y = torch.load(path_join(args.data_folder, args.training_data_file), 
            map_location=torch.device("cpu"))
    
    if "ODE_simulation" in args.training_data_file:
        X = X.reshape(1,-1,38).permute(0,2,1)
        y = y.reshape(1,-1,12).permute(0,2,1)
        Xtrain = X[:,:,0:-210000]
        ytrain = crop_data(Xtrain, y[:,:,0:-70000], net, args)
        
        Xval = X[:,:,-210000:]
        yval = crop_data(Xval, y[:,:,-70000:], net, args)
        dataset_train = MyDataset(Xtrain, ytrain, device=args.device, data_type=args.data_type)
        dataset_val = MyDataset(Xval, yval, device=args.device, data_type=args.data_type)
        return dataset_train, dataset_val
        
        
    if net is not None and not args.training_data_file.endswith("_v2_training.pt"):
        y = crop_data(X, y, net, args)
    
    data = MyDataset(X, y, device=args.device, data_type=args.data_type)

    num_batches = X.shape[0]
    val_batches = int(args.validation_fraction * num_batches)
    train_batches = num_batches - val_batches
        
    if args.use_cv and not args.training_data_file.endswith("_v2_training.pt"):
        split = KFold(args.cv_folds, shuffle=True, random_state=args.seed)
        num_batches = X.shape[0]
        train_idx, val_idx = list(split.split(np.arange(num_batches, dtype=int)))[args.cv_index]
        return data, train_idx, val_idx
    elif args.use_cv :
        assert args.cv_folds == 21
        assert args.cv_folds%3 == 0
        assert args.cv_index < args.cv_folds
        assert args.cv_index >= 0
        cv_folds = args.cv_folds
        cv_index = args.cv_index
        cv_folds_per_area = cv_folds//3
        cv_area_index = cv_index%cv_folds_per_area

        val_area = cv_index//cv_folds_per_area
        X_splitted = X#.split(1)
        y_splitted = y#.split(1)
        val_x_area = X_splitted[val_area].split(X.shape[-1]//cv_folds_per_area,dim=-1)
        val_y_area = y_splitted[val_area].split(y.shape[-1]//cv_folds_per_area,dim=-1)

        X_before_validation = [val_x_area[i] for i in range(cv_area_index)]
        X_after_validation = [val_x_area[i] for i in range(cv_area_index+1,len(val_x_area))]
        y_before_validation = [val_y_area[i] for i in range(cv_area_index)]
        y_after_validation = [val_y_area[i] for i in range(cv_area_index+1,len(val_y_area))]

        train_samples = [(X_splitted[i],y_splitted[i]) for i in range(3) if i != val_area]
        val_sample = [(val_x_area[cv_area_index], val_y_area[cv_area_index])]
        if len(X_before_validation) > 0:
            train_samples.append( (torch.hstack(X_before_validation), torch.hstack(y_before_validation)) )
        if len(X_after_validation) > 0:
            train_samples.append( (torch.hstack(X_after_validation), torch.hstack(y_after_validation)) )
        
        train_samples = crop_data_v2(train_samples, net, args)
        val_sample    = crop_data_v2(val_sample,    net, args)
        dataset_train = MyDataset_v2(train_samples, args.device, args.data_type)
        dataset_val = MyDataset_v2(val_sample, args.device, args.data_type)
        return dataset_train, dataset_val

        
    elif args.debugging:
        torch.random.manual_seed(args.seed)
        return torch.utils.data.dataset.random_split(data, [train_batches, val_batches]), model_type, model_coef, (X,y,data)
    else:
        torch.random.manual_seed(args.seed)
        return torch.utils.data.dataset.random_split(data, [train_batches, val_batches])


def load_data_testing(args, net=None):
    """
    Return a dataset for test
    """
    X, y = torch.load(path_join(args.data_folder, args.testing_data_file), 
        map_location=torch.device("cpu"))
    if net is not None:
        y = crop_data(X, y, net, args)

    data = MyDataset(X, y, device=args.device, data_type=args.data_type)
    return data


class MyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, device=torch.device('cpu'), data_type=torch.float):
        self.X = X.to(device=device, dtype=data_type)
        self.y = y.to(device=device, dtype=data_type)
        self.device = device
        self.data_type = data_type

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index, :, :], self.y[index, :, :]

class MyDataset_v2(Dataset):
    def __init__(self, samples, device=torch.device('cpu'), data_type=torch.float):
        self.samples = [(X.to(device=device, dtype=data_type),y.to(device=device, dtype=data_type)) for (X,y) in samples] 
        self.device = device
        self.data_type = data_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index:int):
        (X,y) = self.samples[index]
        return X, y

def crop_data(X, y, net, args):
    y_shape = net(torch.zeros((2, X.shape[1], X.shape[2]), dtype=args.data_type, device=args.device)).shape
    reduction = y.shape[-1] - y_shape[-1]
    print(reduction)
    if reduction==0:
        return y
    else:
        return y[:, :, int(reduction/2):(-int(reduction/2))]

def crop_data_v2(samples, net, args):
    samples_out = []
    for (X,y) in samples:
        y_shape = net(torch.zeros((1, X.shape[-2], X.shape[-1]), dtype=args.data_type, device=args.device)).shape
        reduction = y.shape[-1] - y_shape[-1]
        if reduction>0:
            y = y[:, int(reduction/2):(-int(reduction/2))]
        samples_out.append((X,y))
    return samples_out