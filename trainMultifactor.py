import argparse
from collections import OrderedDict
import logging
# import sys
# from pathlib import Path
import time
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from net import build_net
from utils.stock_dataset import StockDatasetAll, StockDatabase
from utils import options
from utils.metrics import evaluate_single
from utils.optim import CosineAnnealingLR_WarmUpRestart
from utils.log import get_logger, get_tb_logger


'''
Train the network.
NOTE: 
220727 AlphanetV1 multifactor train
'''
## ----------------
# train defaults
DEFAULTS = dict(
    # name
    name = 'Alphanetv1-multifactor',

    ## dataset 
    dir_folder = r'D:\syx-working\quant\stockdata\wukong',
    # dir_folder = r'D:\syx-working\quant\stockdata\sample',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/wukong',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/test',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/sample',
    # dir_folder = './data-sample/test/'

    split_ratio = [4, 1, 0], # train, valid, test split
    time_length = 30,
    pred_day = 10,

    ## network 
    # in_features = 9 ,
    netname = "Factor",
    net_opt = dict(
        in_features = 32,
        out_features = 1,
        time_length = 30,
    ),

    ## train setting
    batch_size = 2000,
    total_epoch = 50,
    lr = 1e-4,
    each_Tmax=25,

    ## log setting
    exp_dir = "./experiments/",
    resume = None,
)


# TODO:
# 3 后续增加第二版和第三版

regularization_lambda = 1e-3



## ----
def get_args():
    parser = argparse.ArgumentParser(description='Train the network')

    # parser.add_argument('--label', type=str, default=None, help='Experiment label')
    parser.add_argument('-opt', type=str, default=None, help='Read options from file.')
    parser.add_argument('-resume', type=str, default=None, help='Resume training state from file.')
    parser.add_argument('-name', type=str, default=None, help='Train Exp Name.')
    
    # DDP arguments
    parser.add_argument('--local_rank',type=int, default=-1)
    parser.add_argument("--local_world_size", type=int, default=1)

    args = parser.parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def train_model():
    #### Initialize
    args = get_args()
    if args.opt:
        option = DEFAULTS
        option_new = options.read_fromyaml(args.opt)
        option = options.copyvalues(option, option_new)
        del(option_new)
    else:
        option = DEFAULTS
    # set seed 
    seed = option.get("seed", 2022)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = args.local_rank
    if rank < 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        device = torch.device(f'cuda:{args.local_rank}')


    #### update global values
    resume = options.get_firstvalue(args.resume, option['resume'], None)
    # add checkpoint folder with label
    if args.name :
        option['name'] = args.name
    exp_label = option['name']
    format_time = time.strftime("%Y%m%d-%H-%M-%S_", time.localtime())
    exp_label = format_time + exp_label
    exp_folder = os.path.join(option['exp_dir'], exp_label)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)
    # copy config to exp_dir
    if args.opt and rank <= 0:
        name = os.path.split(args.opt)[1]
        shutil.copy(args.opt, os.path.join(exp_folder, name))
    # save option to exp_dir
    # if rank <= 0:
    #     name = os.path.split(args.opt)[1]
    #     options.save_options( os.path.join(exp_folder, name), option)



    ######## Begin process
    ## 1 Initialize logger
    logger = get_logger(logger_name=option['name'], log_file=os.path.join(exp_folder, "running_log.txt"), rank=rank)
    tb_logger = get_tb_logger(comment=option['name'])  ## in 'runs' folder.
    showGPU = True # display gpu usage after the first train epoch
    logging.info(f'Using device {device}') 

    ## 2. Create dataset
    database = StockDatabase(option["dir_folder"], option['split_ratio'], additional_factors=True, minlength=option['time_length']+option['pred_day'])
    train_set = StockDatasetAll(database=database, time_length=option['time_length'], pred_day=option['pred_day'], timestep=2, mode="train")
    val_set = StockDatasetAll(database=database, time_length=option['time_length'], pred_day=option['pred_day'], timestep=2, mode="valid")
    n_train = len(train_set)
    n_val = len(val_set)

    # Split into train / validation partitions
    # else:
    #     val_percent = 0.1
    #     n_val = int(len(dataset) * val_percent)
    #     n_train = len(dataset) - n_val
    #     train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    # loader_args = dict(batch_size=option['batch_size'], num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=option['batch_size'], num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=option['batch_size'], num_workers=4, pin_memory=True)

    ## 3 build model
    # Change here to adapt to your data
    net = build_net(option['netname'], option['net_opt'] ).to(device=device)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    amp = False
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=option['lr'], betas=(0.9, 0.999))
    # optimizer = optim.RMSprop(net.parameters(), lr=option['lr'], momentum=0.9)
    scheduler = CosineAnnealingLR_WarmUpRestart(optimizer, T_total=option['total_epoch'], each_Tmax=option['each_Tmax'], restart_step=option['restart_step'], warmup_step=option['warmup_step'], eta_min=option['eta_min'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)  # goal: maximize Dice score

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    def params_L1loss(model, lamb):
        regularization_loss = 0
        for params in model.parameters():
            regularization_loss += torch.sum(torch.abs(params))
        return regularization_loss * lamb
    


    total_epoch = option['total_epoch']
    start_epoch = 0

    ## 5 resume train
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        # net.load_state_dict(torch.load(args.load, map_location=device))
        logger.info(f'typel loaded trained model from {resume}')

    ## after resuming, move the model to device
    if rank < 0:
        # net = net.to(device=device)
        pass
    else:
        net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.local_rank])


    # 5. Begin training
    logging.info(f'''Starting training:
        Start Epoch:     {start_epoch}
        End Epoch:       {total_epoch}
        Batch size:      {option['batch_size']}
        Learning rate:   {option['lr']}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Load from :      {resume}
    ''')

    net.train()
    global_step = 0
    for epoch in range(start_epoch, total_epoch):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{total_epoch}', unit='img') as pbar:
            for batch in train_loader:
                data_in, data_gt = batch
                data_gt = data_gt.reshape(-1,1)

                # assert data_in.shape[1] == net.in_ch, \
                #     f'Network has been defined with {net.in_ch} input channels, ' \
                #     f'but loaded data have {data_in.shape[1]} channels. Please check that ' \
                #     'the data are loaded correctly.'

                # images = images.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                data_in = data_in.to(device=device, dtype=torch.float32)
                data_gt = data_gt.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = net(data_in)
                    # params_loss = params_L1loss(net, regularization_lambda)
                    loss = criterion(pred, data_gt) 
                    # + params_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(data_in.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # update log info after each iter step
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                tb_logger.add_scalar('train/loss_step', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})


        ## Evaluation round - per epoch
                # show GPU status after the first epoch
        if showGPU:
            print("First iteration step GPU status:")
            os.system("nvidia-smi")
            # a = os.popen("nvidia-smi")
            # logger.info("First iteration step GPU status:\n" + a.read())
            # del(a)
            showGPU = False


            # histograms = {}
            # for tag, value in net.named_parameters():
            #     tag = tag.replace('/', '.')
            #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        ## validation
        valid_results = evaluate(net, val_loader, device)
        val_score = valid_results['MSE_Loss']
        val_corr = valid_results['CORRavg']
        val_rmse = valid_results['RMSEavg']
        val_updownacc = valid_results['UpDownAcc']

        ## save train and valid log
        if tb_logger is not None:
            logger.info('Validation MSE: {}'.format(val_score))
            # logger.info('Validation loss(L1): {}'.format(val_loss))

            # Add values to tensorboard
            tb_logger.add_scalar('train/loss_epoch', loss.item(), epoch)
            tb_logger.add_scalar('valid/learning_rate', optimizer.param_groups[0]['lr'], epoch )
            tb_logger.add_scalar('valid/loss', val_score, epoch)
            tb_logger.add_scalar('valid/Corr', val_corr, epoch)
            tb_logger.add_scalar('valid/RMSE', val_rmse, epoch)
            tb_logger.add_scalar('valid/UpDownAcc', val_updownacc, epoch)

            if rank < 0:
                testnet = net
            else:
                testnet = net.module
            for tag, value in testnet.named_parameters():
                tag = tag.replace('/', '.')
                tb_logger.add_histogram("Weights/"+tag, value.data.cpu(), global_step=epoch)
                tb_logger.add_histogram("Gradients/"+tag, value.grad.data.cpu(), global_step=epoch)


        # scheduler.step(val_score)
        scheduler.step()

        ## Save checkpoint
        # Path(dir_exp).mkdir(parents=True, exist_ok=True)
        # torch.save(net.state_dict(), str(dir_exp / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        if rank < 0:
            torch.save({
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },  os.path.join(exp_folder, 'train_state_epoch{}.tar'.format(epoch+1)) )
        elif rank == 0:
            torch.save({
                # distributed model has attribute module
                "net": net.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },  os.path.join(exp_folder, 'train_state_epoch{}.tar'.format(epoch+1)) )
        else:
            time.sleep(1)
        logger.info(f'Checkpoint {epoch + 1} saved!')


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    score = 0
    Corrs = []
    RMSEs = []
    updownaccs = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        data_in, data_gt = batch
        # move images and labels to correct device and type
        data_in = data_in.to(device=device, dtype=torch.float32)
        data_gt = data_gt.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(data_in)

            # compute the score
            score += F.mse_loss(pred, data_gt)
        pred_array = pred.detach().cpu().numpy().flatten()
        gt_array = data_gt.detach().cpu().numpy().flatten()
        test_results = evaluate_single(pred_array, gt_array, updownthreshold=0.005)
        Corrs.append(test_results['Corr'])
        RMSEs.append(test_results['RMSE'])
        updownaccs.append(test_results['UpDownAcc'])


    net.train()
    results = {
        "MSE_Loss": score / num_val_batches,
        "CORRavg": np.mean(Corrs),
        "RMSEavg": np.mean(RMSEs),
        "UpDownAcc": np.mean(updownaccs),
    }
    return results




#### --------- main process
if __name__ == '__main__':
    train_model()

