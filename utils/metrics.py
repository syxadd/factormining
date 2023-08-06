import os
from typing import List
import pandas as pd
import numpy  as np

"""
Compute metrics.
"""
def compute_R2(pred: np.ndarray, target: np.ndarray):
    residualsq = (target - pred)**2
    totalsq = (target - np.mean(target))**2
    R2 = 1 - np.sum(residualsq) / np.sum(totalsq)
    return R2

def compute_corr(pred: np.ndarray, target: np.ndarray, epsilon=1e-7):
    # return np.cov(pred, target) / (np.std(pred) * np.std(target) + epsilon)
    return np.corrcoef(pred, target)[0, 1]

def compute_rmse(pred: np.ndarray, target: np.ndarray):
    residualsq = (target - pred)**2
    return np.sqrt(np.mean(residualsq))

def compute_mae(pred: np.ndarray, target: np.ndarray):
    return np.mean(np.abs(pred - target))

def compute_updown_acc(pred: np.ndarray, target: np.ndarray, threshold = 0.0):
    """
    计算涨跌预测率，可根据一个threshold确定涨跌情况。
    """
    up = (pred > threshold) & (target > threshold)
    down = (pred < -threshold) & (target < -threshold)
    total_length = len(pred)
    return (up + down).sum() / total_length

def get_ranks(x: np.ndarray):
    """
    Compute Ranks of the 1-d array.
    """
    index = x.argsort()
    ranks = np.empty_like(index)
    ranks[index] = np.arange(len(x))

    return ranks

def compute_rankIC(pred: np.ndarray, target: np.ndarray, epsilon=1e-7):
    """
    计算RankIC
    """
    # rankIC
    pred_rank = get_ranks(pred) + 1
    target_rank = get_ranks(target) + 1
    # return np.cov(pred_rank, target_rank) / (np.std(pred_rank) * np.std(target_rank) + epsilon)
    return np.corrcoef(pred_rank, target_rank)[0, 1]



#### filter bad results 
def remove_nonevalues(df_data: pd.DataFrame):
    length = df_data.shape[0]
    na_loc = np.zeros(length, dtype=bool)
    for column in df_data.columns:
        # 去除 nan 和 inf 值
        na_loc = na_loc | df_data[column].isna()
        na_loc = na_loc | np.isinf(df_data[column])

    loc = ~na_loc
    ## show the location
    visual_loc = np.arange(length)[na_loc]
    print("None values in : ", visual_loc)
    print("Total remove ", np.sum(na_loc))


    # TODO: 待验证是否能起到筛选的作用
    
    return df_data[loc]


######## ----------
# def split_timeseries()

def evaluate_single(pred: np.ndarray, gt: np.ndarray, updownthreshold):
    assert len(pred) == len(gt)
    
    ## compute corr
    corr = compute_corr(pred, gt)
    ## compue RankIC
    rankic = compute_rankIC(pred, gt)
    ## compute R2
    r2 = compute_R2(pred, gt)
    ## compute RMSE
    rmse = compute_rmse(pred, gt)
    ## MAE
    mae = compute_mae(pred, gt)
    ## updown accuracy
    updownacc = compute_updown_acc(pred, gt, updownthreshold)

    results = {
        "Corr": corr,
        "RankIC": rankic,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "UpDownAcc": updownacc,
    }
    return results

# def test_each_prediction(pred: np.ndarray, gt:np.ndarray):
#     return {
#         "Corr": compute_corr(pred, gt),
#         "R2": compute_R2(pred, gt),
#         "RMSE": compute_rmse(pred, gt),
#         "MAE": compute_mae(pred, gt),
#         "UpDownAcc": compute_updown_acc(pred, gt, 0.005),
#     }

def evaluate_prediction_withdate(predictions_all : dict, start_date = None, end_date = None, updownthreshold = 0.0):
    """
    评估预测结果，结果包含prediction, gt, stocknames.

    Structure of 'predictions_all':
    {
        "stock_names" : [str],
        "results" : {
            "df_date": [date],
            "predict": [float],
            "gt_return10": [float],
        }
    }
    
    """
    stock_names = predictions_all['stock_names']
    results = predictions_all['results']
    
    
    # corr_res_train = []
    # R2_res_train = []
    # rmse_res_train = []
    # mae_res_train = []
    # updown_acc_train = []
    corr_res_test = []  # Corr
    rankic_test = []
    R2_res_test = []
    rmse_res_test = []
    mae_res_test = []
    updown_acc_test = []
    ndays = []


    stock_indices = np.ones_like(stock_names, dtype=bool)


    # test each 
    for index in range(len(stock_names)):
        df_date = results[index]['df_date']
        # df_date = pd.to_datetime(pd.Series(df_date).apply(str))
        df_date = pd.to_datetime(df_date)
        predicts_test = results[index]['predict']
        gt_return10_test = results[index]['gt_return10']

        ## 筛选时间区间
        if start_date or end_date:
            loc = np.ones_like(df_date, dtype=bool)
            if start_date:
                loc = loc & (df_date >= start_date)
            if end_date:
                loc = loc & (df_date < end_date)
            ## 筛选
            predicts_test = predicts_test[loc]
            gt_return10_test = gt_return10_test[loc]
            df_date = df_date[loc]

            if len(predicts_test) == 0:
                print(f"Omit the {stock_names[index]} for it has no values." )
                stock_indices[index] = False
                continue

        # Corr - IC
        corr = compute_corr(predicts_test, gt_return10_test)
        corr_res_test.append(corr)

        # RankIC
        rankic = compute_rankIC(predicts_test, gt_return10_test)
        rankic_test.append(rankic)
        # R2
        r2 = compute_R2(predicts_test, gt_return10_test)
        R2_res_test.append(r2)

        # RMSE
        rmse = compute_rmse(predicts_test, gt_return10_test)
        rmse_res_test.append(rmse)

        # MAE
        mae = compute_mae(predicts_test, gt_return10_test)
        mae_res_test.append(mae)

        ## up down accuracy
        threshold = updownthreshold
        updownacc = compute_updown_acc(predicts_test, gt_return10_test, threshold)
        updown_acc_test.append(updownacc)
        ## ndays 
        ndays.append(len(predicts_test))
    
    stock_names = stock_names[stock_indices]
    results_data = {
        "stock_names": stock_names,
        "Corr": corr_res_test,
        "RankIC": rankic_test,
        "R2": R2_res_test,
        "RMSE": rmse_res_test,
        "MAE": mae_res_test,
        "UpDownAcc": updown_acc_test,
        "Days": ndays,
    }
    df_results = pd.DataFrame(data=results_data)
    return df_results





def evaluate_predicts(predict_results: dict, threshold=0.005):
    """
    评估预测结果，结果包含prediction, gt, stocknames

    Structure of 'predict_results':
    {
        "stock_names" : [str],
        "results" : {
            "df_date": [date],
            "predict": [float],
            "gt_return10": [float],
        }
    }

    旧的，对序列进行了划分， 按4：1划分了训练集和测试集。
    
    """
    stock_names = predict_results['stock_names']
    results = predict_results['results']

    split_ratio = [4, 1]
    train_split, test_split = split_ratio
    train_split = train_split / sum(split_ratio)
    test_split = test_split / sum(split_ratio)

    ## metrics
    corr_res_train = []
    corr_res_test = []  # Corr
    R2_res_train = []
    R2_res_test = []
    rmse_res_train = []
    rmse_res_test = []
    mae_res_train = []
    mae_res_test = []
    updown_acc_train = []
    updown_acc_test = []
    rankic_train = []
    rankic_test = []

    # test each 
    for index in range(len(stock_names)):
        df_date = results[index]['df_date']
        predicts = results[index]['predict']
        gt_return10 = results[index]['gt_return10']

        ## split train and test interval
        train_length = int(train_split*len(df_date))

        predicts_train = predicts[:train_length]
        gt_return10_train = gt_return10[:train_length]
        predicts_test = predicts[train_length:]
        gt_return10_test = gt_return10[train_length:]
        # Corr
        corr = compute_corr(predicts_train, gt_return10_train)
        corr_res_train.append(corr)
        corr = compute_corr(predicts_test, gt_return10_test)
        corr_res_test.append(corr)

        ## RankIC
        rankic = compute_rankIC(predicts_train, gt_return10_train)
        rankic_train.append(rankic)
        rankic = compute_rankIC(predicts_test, gt_return10_test)
        rankic_test.append(rankic)

        # R2
        r2 = compute_R2(predicts_train, gt_return10_train)
        R2_res_train.append(r2)
        r2 = compute_R2(predicts_test, gt_return10_test)
        R2_res_test.append(r2)

        # RMSE
        rmse = compute_rmse(predicts_train, gt_return10_train)
        rmse_res_train.append(rmse)
        rmse = compute_rmse(predicts_test, gt_return10_test)
        rmse_res_test.append(rmse)

        # MAE
        mae = compute_mae(predicts_train, gt_return10_train)
        mae_res_train.append(mae)
        mae = compute_mae(predicts_test, gt_return10_test)
        mae_res_test.append(mae)

        ## up down accuracy
        # threshold = 0.005
        updownacc = compute_updown_acc(predicts_train, gt_return10_train, threshold)
        updown_acc_train.append(updownacc)
        updownacc = compute_updown_acc(predicts_test, gt_return10_test, threshold)
        updown_acc_test.append(updownacc)

    ## show metric results 
    print("Evaluate with threshold {}".format(threshold))
    print("----Train Part----")
    print("Total sample: ", len(stock_names))
    print("Average Corr: ", np.mean(corr_res_train))
    print("Average RankIC: ", np.mean(rankic_train))
    print("Average R2: ", np.mean(R2_res_train))
    print("Average RMSE: ", np.mean(rmse_res_train))
    print("Average MAE: ", np.mean(mae_res_train))
    print("Average Up and Down prediction accuracy: ", np.mean(updown_acc_train))

    print("----Test Part----")
    print("Total sample: ", len(stock_names))
    print("Average Corr: ", np.mean(corr_res_test))
    print("Average RankIC", np.mean(rankic_test))
    print("Average R2: ", np.mean(R2_res_test))
    print("Average RMSE: ", np.mean(rmse_res_test))
    print("Average MAE: ", np.mean(mae_res_test))
    print("Average Up and Down prediction accuracy: ", np.mean(updown_acc_test))

    metric_results = {
        "train":{
            "stock_names": stock_names,
            "Corr": corr_res_train,
            "RankIC": rankic_train,
            "R2": R2_res_train,
            "RMSE": rmse_res_train,
            "MAE": mae_res_train,
            "UpDownAcc": updown_acc_train,
        },
        "test":{
            "stock_names": stock_names,
            "Corr": corr_res_test,
            "RankIC": rankic_test,
            "R2": R2_res_test,
            "RMSE": rmse_res_test,
            "MAE": mae_res_test,
            "UpDownAcc": updown_acc_test,
        },
    }

    mean_results = {
        "train": None,
        "test": None,
    }

    rownames = ["Avg-Corr", "Avg-RankIC", "Avg-R2", "Avg-RMSE", "Avg-MAE", "Avg-UpDownAcc"]
    metric_names = ["Corr", "RankIC", "R2", "RMSE", "MAE", "UpDownAcc"]
    for key in metric_results.keys():
        rowvalues = []
        for name in metric_names:
            rowvalues.append( np.mean(metric_results[key][name]) )
        mean_results[key] = rowvalues
    
    mean_results["rownames"] = rownames
    return metric_results, mean_results