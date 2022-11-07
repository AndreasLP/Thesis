import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time, os, sys, argparse

def get_data(x):
    if torch.is_tensor(x):
        return x.to(torch.device('cpu')).data.numpy()
    else:
        return x

def save_model_and_optim(epoch, model, val_errors, optimiser, parameters, args, t, final=False):
    if final is True:
        PATH = os.path.join(args.model_savepoint_folder, type(model).__name__ + 
                            "_{:0>2}-{:0>2}-{}_{:0>2}:{:0>2}_{}_final.pty".format(
                                t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, args.run_id))
    else:
        PATH = os.path.join(args.model_savepoint_folder, type(model).__name__ + 
                            "_{:0>2}-{:0>2}-{}_{:0>2}:{:0>2}_{}_best_seen.pty".format(
                                t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, args.run_id))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'parameters': parameters,
            'args': args,
            'val_errors' : val_errors,
            }, PATH)

def log_epoch_loss(loss_log: str, epoch: int, train_loss: torch.tensor, val_loss: torch.tensor, labels: list, 
        writer_signed: SummaryWriter=None, writer_unsigned: SummaryWriter=None):
    """
    Append the loss in train_loss or val_loss to the loss_log file for the given epoch.
    Input:
        loss_log - path to loss log
        epoch - epoch number
        train_loss - 2d tensor with losses. Rows correspond to 
    """
    timer = time.time()
    train_loss_signed   = train_loss.detach().clone()
    train_loss_unsigned = train_loss.detach().clone().abs()

    val_loss_signed     = val_loss.detach().clone()
    val_loss_unsigned   = val_loss.detach().clone().abs()

    if (epoch%2) == 0:
        with open(loss_log, "a") as f:
            train_loss = train_loss.abs()

            epoch_train_loss_means = get_data(torch.mean(train_loss, [0, 2]))
            # Write to file
            f.write("train losses {:6}:\t".format(epoch))
            for i in range(train_loss.shape[1]):
                f.write("{:6.4}".format(epoch_train_loss_means[i]) + "\t") 
            
            val_loss = val_loss.abs()
            epoch_val_loss_means = get_data(torch.mean(val_loss, [0, 2]))
            epoch_val_loss_maxs = get_data(torch.tensor([torch.max(val_loss[:,i,:]) for i in range(val_loss.shape[1])]))
            epoch_val_loss_q95s = get_data(torch.tensor([torch.quantile(val_loss[:,i,:], 0.95) for i in range(val_loss.shape[1])]))
            # epoch_val_loss_q95s = [-1.]*len(epoch_val_loss_maxs) 
            epoch_val_loss_mean = get_data(torch.mean(val_loss))
            epoch_val_loss_max = get_data(torch.max(val_loss))
            epoch_val_loss_q95 = get_data(torch.quantile(val_loss, 0.95))
            # Write to file
            f.write("\n")
            # Print means
            f.write("val losses   {:6}:\t".format(epoch))
            for i in range(val_loss.shape[1]):
                f.write("{:6.4}".format(epoch_val_loss_means[i]) + "\t") 
            f.write(f" | {epoch_val_loss_mean:.4} mean\n")
            # Print quantiles
            f.write("                     \t")
            for i in range(val_loss.shape[1]):
                f.write("{:6.4}".format(epoch_val_loss_q95s[i]) + "\t") 
            f.write(f" | {epoch_val_loss_q95:.4} quantile\n")
            # Print maxs
            f.write("                      \t")
            for i in range(val_loss.shape[1]):
                f.write("{:6.4}".format(epoch_val_loss_maxs[i]) + "\t") 
            f.write(f" | {epoch_val_loss_max:.4} max\n")
            
            f.write("\n")

    for writer, train_loss, val_loss in zip([writer_signed, writer_unsigned], 
        [train_loss_signed, train_loss_unsigned], [val_loss_signed, val_loss_unsigned]):
        if writer is not None:
            
            epoch_train_loss_mins = {label : torch.min(train_loss[:,i,:]) for i,label in enumerate(labels)}
            epoch_train_loss_means = {label : t for label,t in zip(labels, torch.mean(train_loss, [0, 2]))}
            epoch_train_loss_q95s = {label : torch.quantile(train_loss[:,i,:], 0.95) for i,label in enumerate(labels)}
            # epoch_train_loss_q95s = {label : -1. for i,label in enumerate(labels)}
            epoch_train_loss_maxs = {label : torch.max(train_loss[:,i,:]) for i,label in enumerate(labels)}
            
            epoch_train_loss_min = torch.min(train_loss)
            epoch_train_loss_mean = torch.mean(train_loss)
            epoch_train_loss_q95 = torch.quantile(train_loss, 0.95)
            epoch_train_loss_max = torch.max(train_loss)
            
            epoch_val_loss_mins = {label : torch.min(val_loss[:,i,:]) for i,label in enumerate(labels)}
            epoch_val_loss_means = {label : t for label,t in zip(labels, torch.mean(val_loss, [0, 2]))}
            # epoch_val_loss_q95s = {label : -1. for i,label in enumerate(labels)}
            epoch_val_loss_q95s = {label : torch.quantile(val_loss[:,i,:], 0.95) for i,label in enumerate(labels)}
            epoch_val_loss_maxs = {label : torch.max(val_loss[:,i,:]) for i,label in enumerate(labels)}
            
            epoch_val_loss_min = torch.min(val_loss)
            epoch_val_loss_mean = torch.mean(val_loss)
            epoch_val_loss_q95 = torch.quantile(val_loss, 0.95)
            epoch_val_loss_max = torch.max(val_loss)

            for (stat_type,stat_loss) in zip(
                    ["min", "mean", "q95", "max"],
                    [epoch_train_loss_mins, epoch_train_loss_means, epoch_train_loss_q95s, epoch_train_loss_maxs]
                ):
                for label in stat_loss.keys():
                    writer.add_scalar(f"train/{stat_type}/{label}",      stat_loss[label],    epoch)

            writer.add_scalar('train/min/aggregate',      epoch_train_loss_min,   epoch)
            writer.add_scalar('train/mean/aggregate',     epoch_train_loss_mean,  epoch)
            writer.add_scalar('train/q95/aggregate',      epoch_train_loss_q95,   epoch)
            writer.add_scalar('train/max/aggregate',      epoch_train_loss_max,   epoch)
            
            for (stat_type,stat_loss) in zip(
                    ["min", "mean", "q95", "max"],
                    [epoch_val_loss_mins, epoch_val_loss_means, epoch_val_loss_q95s, epoch_val_loss_maxs]
                ):
                for label in stat_loss.keys():
                    writer.add_scalar(f"val/{stat_type}/{label}",      stat_loss[label],    epoch)
            
            writer.add_scalar('val/min/aggregate',        epoch_val_loss_min,     epoch)
            writer.add_scalar('val/mean/aggregate',       epoch_val_loss_mean,    epoch)
            writer.add_scalar('val/q95/aggregate',        epoch_val_loss_q95,     epoch)
            writer.add_scalar('val/max/aggregate',        epoch_val_loss_max,     epoch)


def get_class(class_name, module="models"):
    return getattr(sys.modules[module], class_name)

def parse_args():
    
    string_to_dtype = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "float64" : torch.float64
    }

    parser = argparse.ArgumentParser()
    # environment

    # Data
    parser.add_argument('--data_type', default='float32')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_folder', default='/work1/s174505/Thesis/Data/')
    parser.add_argument('--training_data_file', default='Interpolated_data_v2_training.pt')
    parser.add_argument('--testing_data_file', default='Interpolated_data_testing.pt')

    parser.add_argument('--validation_fraction', default=0.3, type=float)
    parser.add_argument('--in_features', default=38, type=int)
    parser.add_argument('--out_features', default=12, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    # Run specifications
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--save_model_freq', default=1200, type=int) 
    parser.add_argument('--run_id', default='default')
    parser.add_argument('--run_index', default='None')
    parser.add_argument('--model_savepoint_folder', default='/work1/s174505/Thesis/Models/')

    parser.add_argument('--max_epochs', default=300, type=int) 
    
    parser.add_argument('--debugging', default=False, type=bool) 

    # Cross validation
    parser.add_argument('--use_cv', default=False, type=bool) 
    parser.add_argument('--cv_folds', default=21, type=int) 
    parser.add_argument('--cv_index', default=0, type=int) 

    # Model specifications
    parser.add_argument('--model', default='base_cnn') 
    parser.add_argument('--parameter_settings', default=0, type=int) 
    parser.add_argument('--activation_function', default="relu", type=str) 
    parser.add_argument('--pretrained_model', default="no", type=str) 
    parser.add_argument('--tuning_round', default=0, type=int) 

    # Optimization settings
    parser.add_argument('--optimizer', default="Adam", type=str) 
    parser.add_argument('--lr', default=1e-4, type=float) 
    parser.add_argument('--use_lr_schedule', default=False, type=bool) 
    parser.add_argument('--weight_decay', default=0., type=float) 
    parser.add_argument('--momentum', default=0., type=float)
    parser.add_argument('--use_dropout', default=False, type=bool) 
    parser.add_argument('--use_batch_normalization', default=False, type=bool) 
    

    args = parser.parse_args()

    args.data_type = string_to_dtype[args.data_type]

    if 'cuda' in args.device and torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

    if args.run_index != "None":
        args.run_id = "_".join([args.run_id, args.run_index]) 
        args.run_index = int(args.run_index) 
    else:
        args.run_index = 0
    if args.use_cv:
        args.run_id += '_cv-{}-{}'.format(args.cv_index, args.cv_folds)

    

    return args
