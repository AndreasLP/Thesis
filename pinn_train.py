import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import library.functions.misc_helper_functions as misc_helpers
from library.functions import model_parameters
import library.data.data_helper_functions as data_helpers
import time

if __name__ == '__main__':

    args = misc_helpers.parse_args()

    # Hyperparameters - Data
    device = args.device
    print("Device:", device)
    
    params = {'batch_size': args.batch_size,
              # 'shuffle': True,
              'drop_last': False}

    labels = [
        'LivLongSxD1 [mm]', 'LivLongDxD1 [mm]', 'AllinSxD1 [mm]', 'AllinDxD1 [mm]',
        'LivLongSxD2 [mm]', 'LivLongDxD2 [mm]', 'AllinSxD2 [mm]', 'AllinDxD2 [mm]', 
        'LivLongSxD3 [mm]', 'LivLongDxD3 [mm]', 'AllinSxD3 [mm]', 'AllinDxD3 [mm]']

    # Load hyperparameters
    hyperparameters, model = model_parameters.parameters(args)

    torch.random.manual_seed(args.seed)
    net = model(hyperparameters).to(device=args.device, dtype=args.data_type)

    print("Hyperparameters:")
    for (key, value) in hyperparameters.items():
        print("\t", key, " : ", value, sep="")
    print(net)

    if args.use_cv:
        dataset, train_idx, val_idx = data_helpers.load_data_training(args=args, net=net)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **params)
        val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, **params)
    else:
        train_dataset, val_dataset = data_helpers.load_data_training(args=args, net=net)
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        val_loader = torch.utils.data.DataLoader(val_dataset, **params)

    # Setup save files
    t = time.localtime()
    run_name = type(net).__name__+"_" + args.training_data_file[:-12] + "_{:0>2}-{:0>2}-{}_{:0>2}:{:0>2}_{}".format(
        t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, args.run_id)
    loss_log = "results/epoch_logs/" + run_name + ".log"

    writer_signed = SummaryWriter('/work1/s174505/Thesis/runs/' + run_name + '_signed')#.add_hparams(hyperparameters)
    writer_unsigned = SummaryWriter('/work1/s174505/Thesis/runs/' + run_name + '_unsigned')#.add_hparams(hyperparameters)

    # Hyperparameters - optimiser and loss function
    if args.optimizer == "SGD":
        optimiser = torch.optim.SGD(
            net.parameters(), lr=hyperparameters["lr"])
    elif args.optimizer == "nesterov": 
        optimiser = torch.optim.SGD(
            net.parameters(), lr=hyperparameters["lr"], momentum=args.momentum)
    elif args.optimizer == "AdamW": 
        optimiser = torch.optim.AdamW(
            net.parameters(), lr=hyperparameters["lr"])
    elif args.optimizer == "AMSGrad": 
        optimiser = torch.optim.Adam(
            net.parameters(), lr=hyperparameters["lr"], amsgrad=True)
    else:
        optimiser = torch.optim.Adam(
            net.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
    
    criterion = torch.nn.MSELoss(reduction="mean")

    # Training hyperparameters
    max_epochs = args.max_epochs

    train_mse   = np.zeros((max_epochs, len(labels)))
    val_mse     = []

    train_batches = len(train_loader)
    val_batches = len(val_loader)

    print("Train batches:", train_batches, "\tValidation batches:", val_batches)

    # Use LR scheduler
    if args.use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.7)

    torch.backends.cudnn.benchmark = True
    training_since_save = time.time()
    epoch_start_time = time.time()
    for epoch in range(max_epochs):
        if epoch%100 == 0:
            print("Starting epoch:", epoch)

        train_error_epoch = [None] * len(train_loader)

        
        # Train loop
        net.train()
        for idx, (x, y) in enumerate(train_loader):
            yhat = net(x)

            batch_mse = criterion(yhat, y) 
            optimiser.zero_grad()
            batch_mse.backward()
            optimiser.step()

            train_error_epoch[idx] = yhat - y
        train_error_epoch = torch.concat(train_error_epoch, 0)

        # Validation loop
        val_mse_epoch = torch.zeros((args.out_features, val_batches))

        net.eval()
        with torch.no_grad():
            val_error_epoch = torch.concat([net(x) - y for x,y in val_loader], 0)
            
        misc_helpers.log_epoch_loss(loss_log=loss_log, epoch=epoch, 
            train_loss=train_error_epoch, val_loss=val_error_epoch, labels=labels, 
            writer_signed=writer_signed, writer_unsigned=writer_unsigned)

        if (epoch+1)%100 == 0:
            print("Epochs time: {:.2f} s".format(time.time() - epoch_start_time))
            epoch_start_time = time.time()
        
        if args.use_lr_schedule:
            scheduler.step()
        
        # End of final epoch
        if epoch == max_epochs - 1:
            # Calculate errors
            net.eval()
            with torch.no_grad():
                val_error_epoch = torch.concat([net(x) - y for x,y in val_loader], 0)
            misc_helpers.save_model_and_optim(epoch, net, val_error_epoch, optimiser, hyperparameters, args, final=True)
        
        elif (time.time() - training_since_save) >= args.save_model_freq: # else save checkpoint
            misc_helpers.save_model_and_optim(epoch, net, None, optimiser, hyperparameters, args, final=False)
