import argparse
import logging
import sys
import time
import os
import shutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from utils.stock_dataset import StockDatasetAll, StockDatabase
from net import build_net
from utils import options
from utils.stock_dataset import StockDatabase
from utils.log import get_logger, get_tb_logger
from utils.metrics import evaluate_predicts

"""
220724
预测多因子模型的Alphanetv1 with dropout
wukong

"""

DEFAULTS = dict(
    # name
    name = 'Alphanetv3-multiple-20ep',

    ## dataset 
    dir_folder = r'D:\syx-working\quant\stockdata\wukong',
    # dir_folder = r'D:\syx-working\quant\stockdata\sample',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/wukong',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/test',
    # dir_folder = '/home/csjunxu-3090/yx/jupyter-detect/proj-quant/stockdata/sample',
    # dir_folder = './data-sample/test/'
    time_length = 30,
    pred_day = 10,

    ## network 
    # in_features = 9 ,
    netname = "AlphaNetv3GRU",
    net_opt = dict(
        in_features = 32,
        out_features = 1,
        time_length = 30,
    ),

	## test
	checkpoint_path = r"" ,
	out_dir = "test_results",  # defaults
)

remove_suspend = True

def get_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--label', type=str, default=None, help='Experiment label')
    parser.add_argument('-opt', type=str, default=None, help='Read options from file.')
    parser.add_argument('-resume', type=str, default=None, help='Resume training state from file.')
    parser.add_argument('-checkpoint', type=str, default=None, help='Checkpoint file.')
    parser.add_argument('-name', type=str, default=None, help='Test Name.')
    parser.add_argument('-dir_folder', type=str, default=None, help='Dir folder.')

    # DDP arguments
    parser.add_argument('--local_rank',type=int, default=-1)
    parser.add_argument("--local_world_size", type=int, default=1)

    args = parser.parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def load_model_checkpoint(checkpoint_file: str):
    "Load checkpoint model state_dict"
    checkpoint = torch.load(checkpoint_file)
    if checkpoint_file.endswith(".tar"):
        return checkpoint['net']
    else:
        return checkpoint


def test_model():
	# Get options
    args = get_args()
    if args.opt:
        option = DEFAULTS
        option_new = options.read_fromyaml(args.opt)
        option = options.copyvalues(option, option_new)
        del(option_new)
    else:
        option = DEFAULTS

    ## overwrite values
    if args.name :
        option['name'] = args.name
    if args.dir_folder:
        option['dir_folder'] = args.dir_folder

    ## init seed and dist
    seed = option.get("seed", 2022)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Use GPU to test model
    device = torch.device("cuda")
    
    net = build_net(option['netname'], option['net_opt'] )
    time_length = option['time_length']
    pred_day = option['pred_day']
    timestep = 1
    
    ## load checkpoint
    if args.checkpoint:
        option['checkpoint_path'] = args.checkpoint
    checkpoint = load_model_checkpoint(option['checkpoint_path'])
    net.load_state_dict(checkpoint)

    ## build dataset 
    data_folder = option['dir_folder']
    database = StockDatabase(data_folder, split_ratio=[0,0,1], additional_factors=True)
    datalist = database.get_datalist(mode="test")
    name_list = database.datanames
    # name_list = [name for name in os.listdir(data_folder) if name.endswith(".csv")]


    ## create output folder
    format_time = time.strftime("%Y%m%d-%H-%M-%S_", time.localtime())
    label = format_time + option['name']
    out_dir = os.path.join(option['out_dir'], label)
    visual_folder = os.path.join(out_dir, 'outputs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(visual_folder, exist_ok=True)

    ## move to device 
    net = net.cuda()
    net.eval()

    test_results = {
        "stock_names": [],
        "results": [],
    }

    ## begin test 
    for i, name in enumerate(name_list):
        # filename = os.path.join(data_folder, name)
        # df_data = pd.read_csv(filename)
        df_data = datalist[i]
        datainfo = database.full_datainfos[i]
        stock_name = name.replace(".csv", "")
        index_date = datainfo['trade_dt']


        data_len = df_data.shape[0]
        data_batch = []
        gt_batch = []
        datelist = []
        # gts = []
        print(f"Begin testing the {i+1} / {len(name_list)} sample...")
        ## build batch level 把每个测试集的数据合并起来构成一个batch维度
        for j in range(0, data_len - time_length - pred_day+1, timestep):
            time_index = j
            data_in = df_data.iloc[time_index:(time_index+time_length), :]
            ## 移除停牌信息
            # if remove_suspend and datalist[i]['suspend'].iloc[j:j+time_length+pred_day].sum() > pred_day :
            #     continue
            
            data_batch.append(torch.from_numpy(data_in.to_numpy() ))
            gt_return = df_data['s_dq_close'].iloc[time_index+time_length+pred_day-1] / df_data['s_dq_close'].iloc[time_index+time_length-1] - 1
            gt_batch.append(torch.tensor([gt_return]))
            datelist.append(index_date.iloc[time_index+time_length+pred_day-1])

        ## ignore the stock  that does not have enough data
        if len(data_batch) == 0:
            print(f"{name} does not contain enough data , only with length {data_len}")
            continue

        ## test each length
        with torch.no_grad():
            data_batch = torch.stack(data_batch).to(device=device, dtype=torch.float32)  # shape (B, T, C)
            ## 添加标准化
            # epsilon = 1e-8
            # data_batch = (data_batch - data_batch.mean(dim=1, keepdim=True)) / (data_batch.std(dim=0, keepdim=True) + epsilon)

            gt_batch = torch.stack(gt_batch).to(device=device, dtype=torch.float32)  # shape (B, 1)

            ## predict
            pred = net(data_batch) # (B, 1)

            ## unpack 单个股票的时长数据作为输入
            predicts = pred.cpu().numpy().flatten()
            gts = gt_batch.cpu().numpy().flatten()



        ## draw results 
        # dateseries = index_date.iloc[time_length+pred_day-1:].apply(str)
        dateseries = pd.Series(datelist).apply(str)

        plt.plot(dateseries, gts, color='orange')
        plt.plot(dateseries, predicts, color='blue')
        plt.legend(["gt_return10", "predict",])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))  # xaxis show 1,6 month
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
        plt.xticks(rotation=30, horizontalalignment='right')
        plt.xlabel("Datetime")
        plt.ylabel(f"{pred_day} days return rate")
        # plt.vlines([], 0, 0.5)
        # save fig
        png_file = os.path.join(visual_folder, stock_name+'.png')
        plt.savefig(png_file)
        plt.close()


        # Append to test_results
        test_results['stock_names'].append( stock_name )
        test_results['results'].append({
            "df_date": dateseries.to_numpy(),
            "predict": np.array(predicts),
            "gt_return10" : np.array(gts),
        })

    ## save test results
    print("Writing results to files...")
    result_file = os.path.join(out_dir, "test_results.npz")
    np.savez(result_file, **test_results)
    # with open(result_file, 'w') as f:
    #     json.dump(test_results, f)



    ##### Evaluate predictions
    import traceback
    try:
        print("Begin Evaluation...")
        metric_results, mean_results = evaluate_predicts(test_results)
        df_results = pd.DataFrame(data=metric_results['train'])
        df_results.to_csv(os.path.join(out_dir, "totalresults-train.csv"))
        df_results = pd.DataFrame(data=metric_results['test'])
        df_results.to_csv(os.path.join(out_dir, "totalresults-test.csv"))


        df_means = pd.DataFrame(data=mean_results)
        df_means.to_csv(os.path.join(out_dir, "average_metrics.csv"))
    except:
        traceback.print_exc()
    
    



    print("End of test process.")





if __name__ == "__main__":
	test_model()