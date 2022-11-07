from logging import warning
from ..base_models.linear_models import linear_model, constant_model
from ..cnn.cnn_models import base_cnn
import numpy as np

def parameters(args):
    str_to_model = {
        "constant_model" : constant_model,
        "linear_model" : linear_model,
        "base_cnn" : base_cnn,
    }

    if args is None:
        hyperparameters = {
            "in_features": 37, 
            "out_features": 14, 
            "lr" : 1e-4,
            "use_lr_schedule" : False,
            "weight_decay" : 0.,
            "validation_fraction" : 0.25,
            "seed" : 10
        }

        model = str_to_model["linear_model"]
    else:
        if not hasattr(args, 'use_batch_normalization'):
            use_batch_normalization = False
        else:
            use_batch_normalization = args.use_batch_normalization
        if not hasattr(args, 'activation_function'):
            activation_function = "elu"
        else:
            activation_function = args.activation_function

        hyperparameters = {
            "data_type": str(args.data_type), 
            "device": str(args.device), 
            "data_folder": args.data_folder, 
            "training_data_file": args.training_data_file, 
            "testing_data_file": args.testing_data_file, 
            
            "validation_fraction" : args.validation_fraction,
            "in_features": args.in_features, 
            "out_features": args.out_features, 
            "batch_size": args.batch_size, 
            
            "seed" : args.seed,
            "save_model_freq" : args.save_model_freq,
            "run_id" : args.run_id,
            "run_index" : args.run_index,
            "model_savepoint_folder" : args.model_savepoint_folder,
            
            "max_epochs" : args.max_epochs,
            
            "debugging" : args.debugging,
            
            "use_cv" : args.use_cv, 
            "cv_folds" : args.cv_folds, 
            "cv_index" : args.cv_index, 
            
            "optimizer" : args.optimizer,
            "lr" : args.lr,
            "use_lr_schedule" : args.use_lr_schedule,
            "weight_decay" : args.weight_decay,
            "momentum" : args.momentum,
            
            "model": args.model,
            "strides" : [],
            "kernel_lenghts" : [],
            "channels" : [],
            "paddings" : [],

            "activation_function" : activation_function,

            "use_batch_normalization" : use_batch_normalization,
        }
        model = str_to_model[args.model]

        if args.model == 'base_cnn':
            
            if args.training_data_file == 'Real_data_training.pt':
                if args.parameter_settings == 1:
                    pass
                else:
                    hyperparameters['strides'] =        [1, 1, 1]  + [1]
                    hyperparameters['paddings'] =       [0, 0, 0]  + [0]
                    hyperparameters['kernel_lenghts'] = [301, 7, 7] + [3]
                    hyperparameters['channels'] =       [128, 64, 32]

            else:

                if args.parameter_settings == 1:
                    hyperparameters['strides'] =        [3,       1,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     256, 64,  32 ]
                elif args.parameter_settings == 2:
                    hyperparameters['strides'] =        [1,       1,   3,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     256, 64,  32 ]
                elif args.parameter_settings == 3:
                    hyperparameters['strides'] =        [1,       1,   1,   3  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     256, 64,  32 ]
                elif args.parameter_settings == 4:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     128, 64,  32 ]
                elif args.parameter_settings == 5:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     128, 128,  32 ]
                elif args.parameter_settings == 6:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [128,     64,  32,  16 ]
                elif args.parameter_settings == 7:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 8:
                    hyperparameters['strides'] =        [1,       3  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [128,     256]
                elif args.parameter_settings == 9:
                    hyperparameters['strides'] =        [1,       3  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [128,     128]
                elif args.parameter_settings == 10:
                    hyperparameters['strides'] =        [1,       3,   1 ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0 ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7 ] + [3]
                    hyperparameters['channels'] =       [128,     256, 64]
                elif args.parameter_settings == 11:
                    hyperparameters['strides'] =        [1,       3,   1 ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0 ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9,   7 ] + [3]
                    hyperparameters['channels'] =       [128,     128, 64]
                elif args.parameter_settings == 12:
                    hyperparameters['strides'] =        [1,       3  ] + [1]
                    hyperparameters['paddings'] =       [0,       0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [64,      32 ]
                elif args.parameter_settings == 13:
                    hyperparameters['strides'] =        [1,       3  ] + [1]
                    hyperparameters['paddings'] =       [0,       0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [64,      64 ]
                elif args.parameter_settings == 14:
                    hyperparameters['strides'] =        [3,       1  ] + [1]
                    hyperparameters['paddings'] =       [0,       0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [64,      64 ]
                elif args.parameter_settings == 15:
                    hyperparameters['strides'] =        [3,       1  ] + [1]
                    hyperparameters['paddings'] =       [0,       0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 9  ] + [3]
                    hyperparameters['channels'] =       [64,      32 ]
                elif args.parameter_settings == 16:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 17:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [800*3+1, 9,   7,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 18:
                    hyperparameters['strides'] =        [1,       3,   1 ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0 ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 9,   7 ] + [3]
                    hyperparameters['channels'] =       [256,     256, 64]
                elif args.parameter_settings == 19:
                    hyperparameters['strides'] =        [1,       3,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 9,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256, 128]
                elif args.parameter_settings == 20:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [400*3+1, 21,  7,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 21:
                    hyperparameters['strides'] =        [1,       3,       1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,       0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 600*3+1, 7,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256,     64,  32 ]
                elif args.parameter_settings == 22:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 9,   9,   9  ] + [1]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 23:
                    hyperparameters['strides'] =        [1,       3,   1,   1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,   0,   0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 21,  7,   7  ] + [7]
                    hyperparameters['channels'] =       [256,     256, 64,  32 ]
                elif args.parameter_settings == 24:
                    hyperparameters['strides'] =        [1,       3,         1  ] + [1]
                    hyperparameters['paddings'] =       [200*3,   0,         0  ] + [0]
                    hyperparameters['kernel_lenghts'] = [600*3+1, 600*3+1,   7  ] + [3]
                    hyperparameters['channels'] =       [256,     256,       128]
                else:
                    warning("Parameter settings not found")
                    return
            
            if args.use_dropout:
                hyperparameters['use_dropout'] = True
                hyperparameters['dropout'] =     [0.00, 0.10, 0.10, 0.00, 0.0]
            else:
                hyperparameters['use_dropout'] = False
                hyperparameters['dropout'] =     [0] * (len(hyperparameters['channels'])+1)
            
            index_to_lr = {
                1 : 5e-5, 2 : 1e-4, 3 : 2e-4, 4 : 6e-4, 5 : 1e-3, 6 : 2e-3,
            }
            index_to_activation_function = {
                1  : "relu", 2  : "elu", 3  : "leakyrelu", 4  : "rrelu", 5  : "prelu"
            }
            index_to_optimizer = {
                1  : "SGD", 2  : "nesterov", 3  : "AdamW", 4  : "AMSGrad", 5  : "Adadelta",
                6  : "Adagrad", 7  : "Adamax", 8  : "ASGD", 9  : "Adam", 10 : "NAdam",
                11 : "RAdam", 12 : "RMSprop", 13 : "Rprop", 14 : "LBFGS",
            }
            index_to_dropout = {
                1  : [0.50, 0.40, 0.40, 0.40, 0.4], 2  : [0.20, 0.40, 0.40, 0.40, 0.2],
                3  : [0.00, 0.40, 0.40, 0.40, 0.0], 4  : [0.50, 0.40, 0.40, 0.50, 0.2],
                5  : [0.20, 0.40, 0.40, 0.50, 0.2], 6  : [0.00, 0.40, 0.40, 0.50, 0.2],
                7  : [0.50, 0.40, 0.40, 0.30, 0.5], 8  : [0.20, 0.40, 0.40, 0.30, 0.5],
                9  : [0.00, 0.40, 0.40, 0.30, 0.5], 10 : [0.50, 0.40, 0.40, 0.30, 0.2],
                11 : [0.20, 0.40, 0.40, 0.30, 0.2], 12 : [0.00, 0.40, 0.40, 0.30, 0.2],
                13 : [0.00, 0.20, 0.20, 0.20, 0.2], 14 : [0.00, 0.30, 0.30, 0.10, 0.1],
                15 : [0.00, 0.40, 0.30, 0.20, 0.2], 16 : [0.00, 0.30, 0.20, 0.30, 0.0],
                17 : [0.00, 0.20, 0.20, 0.20, 0.0], 18 : [0.00, 0.10, 0.10, 0.10, 0.0],
                19 : [0.00, 0.10, 0.10, 0.00, 0.0], 20 : [0.00, 0.10, 0.00, 0.10, 0.0],
                21 : [0.00, 0.30, 0.30, 0.00, 0.0], 22 : [0.00, 0.05, 0.05, 0.00, 0.0],
                23 : [0.00, 0.50, 0.50, 0.00, 0.0], 24 : [0.00, 0.55, 0.55, 0.20, 0.0],
                25 : [0.00, 0.60, 0.60, 0.10, 0.0], 26 : [0.05, 0.45, 0.45, 0.30, 0.0],
                27 : [0.05, 0.50, 0.50, 0.00, 0.0], 28 : [0.00, 0.45, 0.45, 0.40, 0.0],
            }
            
            if not hasattr(args, "tuning_round"):
                args.tuning_round = 0
            
            if args.tuning_round==3:
                hyperparameters["lr"] = index_to_lr[args.run_index]
                args.lr = index_to_lr[args.run_index]
            
            if args.tuning_round==4:
                hyperparameters["dropout"] = index_to_dropout[args.run_index]
            
            if args.tuning_round==5:
                hyperparameters["lr"] *= 2.7**(args.run_index)
                args.lr *= 2**(args.run_index)
            
            if args.tuning_round==8:
                hyperparameters["activation_function"] = index_to_activation_function[args.run_index]
                args.activation_function = index_to_activation_function[args.run_index]
            
            if args.tuning_round==9:
                hyperparameters["weight_decay"] = np.logspace(-7, 0, 8)[args.run_index-1]
            
            if args.tuning_round==11:
                hyperparameters["optimizer"] = index_to_optimizer[args.run_index]
                args.optimizer = index_to_optimizer[args.run_index]

            if args.run_index > 1 and args.tuning_round==13:
                hyperparameters["lr"] *= 1.5**(args.run_index-1)
            

    return hyperparameters, model
