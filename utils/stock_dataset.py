import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from net.ts_factor import *

"""
Stock Dataset.
Read data with shape (Batch, Time, Character)

"""
# class StockDatasetOld(Dataset):
#     def __init__(self, data_root: str, time_length: int = 30, random_select=True):
#         """
#         Stock dataset. Data are saved in csv files. Each file contains one single stock price info.
#         Read each file in each iteration.
#         """
#         super().__init__()
#         self.folder = data_root
#         self.time_length = time_length
#         # self.datanames = [name for name in os.listdir(data_root) if name != '.ipynb_checkpoints']
#         self.datanames = [name for name in os.listdir(data_root) if name.endswith('.csv')]
#         # self.random_select = random_select
#         self.colnames = ['s_dq_close', 's_dq_open', 's_dq_low', 's_dq_high', 's_dq_volume', 'dq_vwap', 'dq_return', 'dq_turnover', 'dq_free_turnover']
        
        

#     def __len__(self):
#         return len(self.datanames)

#     def __getitem__(self, idx):
#         data_file = os.path.join(self.folder, self.datanames[idx])
#         # df_data = pd.read_csv(data_file, encoding='utf8')
#         df_data = pd.read_csv(data_file, encoding='utf8')

#         columns = self.colnames
#         df_data = df_data[columns]
#         ## random select
#         length = df_data.shape[0]
#         if length - self.time_length <= 1:
#             raise ValueError("Time length is toot short!" + str(self.time_length) + "File is " + str(self.datanames[idx]))
#         time_start = np.random.randint(1, length-self.time_length, 1)[0]
#         time_end = time_start + self.time_length
#         sub_data = df_data.iloc[time_start:time_end, :]
#         # get input data and gt label
#         gt_price = df_data['s_dq_close'][time_start-1]
#         data = sub_data.to_numpy()
#         data = data.transpose([1,0])
#         data = torch.from_numpy(data)

#         return data, gt_price



# class StockDatasetSimple(Dataset):

#     def __init__(self, data_root: str, time_length: int = 30, pred_day: int = 1 , timestep: int = 1, minlength: int = 40):
#         """
#         Stock dataset. Data are saved in csv files. Each file contains one single stock price info.
#             Assume data are sorted in positive order.
#             Read all data together into memory once. And then read sequences in batch for loader.
#             After reading all data files, create an index list with format : (stock_index, time_index)  which is used for loading data.
#             Each Step return data with shape (, Time, Characts)

#         这部分为没有在数据里添加额外因子的数据集，读取的时候
            
#         """
#         super().__init__()
#         self.folder = data_root
#         self.time_length = time_length
#         self.pred_day = pred_day
#         assert timestep > 0
#         self.timestep = timestep
#         self.minlength = minlength
#         # self.datanames = [name for name in os.listdir(data_root) if name != '.ipynb_checkpoints']
#         self.datanames = [name for name in os.listdir(data_root) if name.endswith('.csv')]
#         self.colnames = ['s_dq_close', 's_dq_open', 's_dq_low', 's_dq_high', 's_dq_volume', 'dq_vwap', 'dq_return', 'dq_turnover', 'dq_free_turnover']

#         # read all data into memory so memory must be large enough
#         print("Read all data...")
#         self.data_list = []
#         self.index_list = []
#         for i, name in enumerate(self.datanames):
#             df_data = self.read_data_fromcsv(os.path.join(self.folder, name))

#             ## Fill NA values
#             df_data['s_dq_close'] = df_data['s_dq_close'].fillna(method='pad')
#             df_data['s_dq_open'] = df_data['s_dq_open'].fillna(method='pad')
#             df_data['s_dq_low'] = df_data['s_dq_low'].fillna(method='pad')
#             df_data['s_dq_high'] = df_data['s_dq_high'].fillna(method='pad')
#             df_data['s_dq_volume'] = df_data['s_dq_volume'].fillna(0)
#             df_data['dq_vwap'] = df_data['dq_vwap'].fillna(0)
#             df_data['dq_return'] = df_data['dq_return'].fillna(0)
#             df_data['dq_turnover'] = df_data['dq_turnover'].fillna(0)
#             df_data['dq_free_turnover'] = df_data['dq_free_turnover'].fillna(0)

#             #TODO: 取一半作为训练集，一半作为测试集

#             self.data_list.append( df_data )
#             # check with time step
#             total_length = df_data.shape[0]
#             for j in range(0, total_length - self.time_length - self.pred_day, timestep):
#                 self.index_list.append( (i, j) )

#         print(f"Finish reading data with {len(self.datanames)} examples.")
        

#     def __len__(self):
#         return len(self.index_list)

#     def __getitem__(self, idx):
#         stock_index, time_index = self.index_list[idx]
#         df_data = self.data_list[stock_index]
#         # input data
#         data_in = df_data.iloc[time_index:(time_index+self.time_length), :]
#         data_in = torch.from_numpy(data_in.to_numpy())
    
#         # gt label
#         gt_price = df_data['s_dq_close'][time_index+self.time_length+self.pred_day-1]
        
#         return data_in, gt_price

#     def read_data_fromcsv(self, filename: str):
#         df_data = pd.read_csv(filename, encoding='utf8')
#         columns = self.colnames
#         df_data = df_data[columns]

#         return df_data




class StockDatabase:
    """
    完整的股票数据库，包含了添加的因子等东西；一次性读取到内存中使用。

    220821之前配合export_data_tosubs.py 来处理。之后建立新的数据库直接读取数据。
    """
    # 保存处理好的DataFrame，可直接用
    full_datalist = []
    # 保存时间日期等信息
    full_datainfos = []

    # 基于以上的数据进行划分
    train_datalist = []
    train_datainfos = []
    valid_datalist = []
    valid_datainfos = []
    test_datalist = []
    test_datainfos = []

    def __init__(self, data_root: str, split_ratio: List[int or float], additional_factors: bool, minlength : int = 40, factormode: str = "customize") -> None:
        """
        Read all stock data at once, and save in a list.
        Args: 
            data_root : folder containing csv files;
            split_ratio:  train, valid, test split ratio.
        """
        self.folder = data_root
        self.datanames = [name for name in os.listdir(data_root) if name.endswith('.csv')]
        self.minlength = minlength
        self.additional_factors = additional_factors
        self.factormode = factormode.lower()
        # 先保留日期信息，留着构建完因子之后再剔除
        if factormode.lower() == "alphanetv1":
            ## only 9 factors
            self.colnames = ['trade_dt', 's_dq_close', 's_dq_open', 's_dq_low', 's_dq_high', 's_dq_volume', 'dq_vwap', 'dq_return1', 'dq_turnover', 'dq_free_turnover']

        else:
            ### "customize" mode 
            self.colnames = ['trade_dt', 's_dq_close', 's_dq_open', 's_dq_low', 's_dq_high', 's_dq_adjfactor', 's_dq_volume', 's_dq_amount', 'tot_shr_today', 'float_a_shr_today', 'free_shares_today',
            'dq_vwap', 'dq_return1', 'dq_turnover', 'dq_free_turnover',
            'national_debt_return_close', 'shangzheng_index_close', 'suspend']


        ## 分析分割ratio
        if len(split_ratio) == 0:
            # 完全是测试集
            split_ratio = [0, 0, 1]
        else:
            train_split, valid_split, test_split = split_ratio
            if (train_split + valid_split + test_split) > 1:
                train_split = train_split / (train_split + valid_split + test_split)
                valid_split = valid_split / (train_split + valid_split + test_split)
                test_split = test_split / (train_split + valid_split + test_split)
        assert valid_split > 1e-7 or test_split > 1e-7

        # read all data into memory so memory must be large enough
        print("Read all data...")

        for i, name in enumerate(self.datanames):
            df_data = self.read_data_fromcsv(os.path.join(self.folder, name))
            df_data['trade_dt'] = pd.to_datetime(df_data['trade_dt'].apply(str))
            # df_data = df_data.set_index('trade_dt', drop=False)
            length = df_data.shape[0]
            # 去掉不够划分训练集测试集中最小时间区间的股票数据
            if valid_split > 1e-7 and test_split > 1e-7:
                minsplit = min(valid_split, test_split)
            else:
                minsplit = valid_split if valid_split > 1e-7 else test_split
            if length*minsplit < self.minlength:
                print("Omit the stock ", name, " with length ", length, " since the minimum length %f is less than %f." % (length*minsplit, self.minlength))
                continue


            ## Fill NA values
            df_data['s_dq_close'] = df_data['s_dq_close'].fillna(method='pad')
            df_data['s_dq_open'] = df_data['s_dq_open'].fillna(method='pad')
            df_data['s_dq_low'] = df_data['s_dq_low'].fillna(method='pad')
            df_data['s_dq_high'] = df_data['s_dq_high'].fillna(method='pad')
            df_data['s_dq_volume'] = df_data['s_dq_volume'].fillna(0)
            ## 加权平均价在停牌时设定为当天价格
            na_loc = df_data['dq_vwap'].isna()
            # df_data['dq_vwap'][na_loc] = df_data['s_dq_close'][na_loc]
            df_data.loc[na_loc, 'dq_vwap'] = df_data['s_dq_close'][na_loc]
            df_data['dq_return1'] = df_data['dq_return1'].fillna(0)
            df_data['dq_turnover'] = df_data['dq_turnover'].fillna(0)
            df_data['dq_free_turnover'] = df_data['dq_free_turnover'].fillna(0)
            ### suspend dtype from bool
            df_data['suspend'] = df_data['suspend'] * 1
            if self.factormode == "customize":
                df_data['s_dq_adjfactor'] = df_data['s_dq_adjfactor'].fillna(method='pad')
                df_data['s_dq_amount'] = df_data['s_dq_amount'].fillna(0)
                df_data['tot_shr_today'] = df_data['tot_shr_today'].fillna(method='pad')
                df_data['float_a_shr_today'] = df_data['float_a_shr_today'].fillna(0)
                df_data['free_shares_today'] = df_data['free_shares_today'].fillna(0)
                df_data['national_debt_return_close'] = df_data['national_debt_return_close'].fillna(method='pad')
                df_data['shangzheng_index_close'] = df_data['shangzheng_index_close'].fillna(method='pad')

                

            if self.additional_factors:
                ## 创建新的因子，并增加技术指标
                df_data = self.add_factors(df_data=df_data)
                
                # 去除生成NA的时间段，也就是剔除最开始几天的时间段
                df_data = df_data.dropna()
            



            ## (IN THE FINAL STEP !!!)  backup date infos and remove columns
            datainfo = pd.DataFrame()
            datainfo['trade_dt'] = df_data['trade_dt']
            df_data.drop(columns=['trade_dt'], inplace=True)

            self.full_datalist.append(df_data)
            self.full_datainfos.append(datainfo)


            #根据分割比例，划分训练，验证和测试集（一般来说划分训练和验证，或者训练和测试即可）
            df_train = df_data.iloc[:int(train_split*length), :]
            train_info = datainfo.iloc[:int(train_split*length), :]
            df_valid = df_data.iloc[int(train_split*length):int((train_split+valid_split)*length), :]
            valid_info = datainfo.iloc[int(train_split*length):int((train_split+valid_split)*length), :]
            df_test = df_data.iloc[int((train_split+valid_split)*length):, :]
            test_info = datainfo.iloc[int((train_split+valid_split)*length):, :]

            self.train_datalist.append(df_train)
            self.train_datainfos.append(train_info)
            self.valid_datalist.append(df_valid)
            self.valid_datainfos.append(valid_info)
            self.test_datalist.append(df_test)
            self.test_datainfos.append(test_info)

            col_len = df_data.shape[1]

        print(f"Finish reading data with {len(self.datanames)} examples.")
        print(f"Has columns : {col_len}.")
        print(df_data.columns)

    def read_data_fromcsv(self, filename: str):
        df_data = pd.read_csv(filename, encoding='utf8')
        # remove unused columns 
        columns = self.colnames
        df_data = df_data[columns]

        return df_data

    # def export_databases_tocsv(self, out_folder: str):
    #     for i in range(len(self.datanames)):
    #         name = self.datanames[i]
    #         subdata = self.full_datalist[i]
    #         subinfo = self.full_datainfos[i]
    #         subdata['trade_dt'] = subinfo['trade_dt']

    @staticmethod
    def add_factors(df_data: pd.DataFrame, inplace=True):
        ## 增加了技术指标
        ndays = 14
        df_data['ts_AD'] = ts_AD(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], volume=df_data['s_dq_volume'])
        # ATR 参考设置为14天：https://baike.baidu.com/item/%E7%9C%9F%E5%AE%9E%E6%B3%A2%E5%B9%85/175870
        df_data['ts_ATR'] = ts_ATR(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], days=14)
        # ADTM 设置为23日
        df_data['ts_ADTM'] = ts_ADTM(high=df_data['s_dq_high'], low=df_data['s_dq_low'], open_price=df_data['s_dq_open'], days_n=23)
        # CCI 设置为14日
        df_data['ts_CCI'] = ts_CCI(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], days=14)
        # 59tian na
        # 心理线一般设置为12天
        df_data['ts_PSY'] = ts_PSY(open_price=df_data['s_dq_open'], close_price=df_data['s_dq_close'], days=12)
        # RSI默认14天
        df_data['ts_RSI'] = ts_RSI(close_price=df_data['s_dq_close'], days=14)
        # 乖离率，短线设置为6天，中线设置为10天或12天
        df_data['ts_BIAS'] = ts_BIAS(close_price=df_data['s_dq_close'], days=12, percent=False)
        # 威廉指标没说，可能选14天？
        df_data['ts_WandR'] = ts_WandR(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], days=ndays, percent=False)
        # quanshi NA
        # 短线用9天表示？
        df_data['ts_K'], df_data['ts_D'], df_data['ts_J'] = ts_KDJ(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], day_n=9)
        # 好像是用6天表示
        df_data['ts_ASI'] = ts_ASI(high=df_data['s_dq_high'], low=df_data['s_dq_low'], open_price=df_data['s_dq_open'], close_price=df_data['s_dq_close'], days=6)
        # OBV 
        df_data['ts_OBV'] = ts_OBV(volume=df_data['s_dq_volume'], close_price=df_data['s_dq_close'])
        # 人气意愿指标为，设置26天
        df_data['ts_AR'], df_data['ts_BR'] = ts_BRAR(high=df_data['s_dq_high'], low=df_data['s_dq_low'], open_price=df_data['s_dq_open'], close_price=df_data['s_dq_close'], days=26)
        df_data['ts_CR'] = ts_CR(high=df_data['s_dq_high'], low=df_data['s_dq_low'], open_price=df_data['s_dq_open'], close_price=df_data['s_dq_close'], days=26)
        # 59tian
        # 货币能量指数 14天
        df_data['ts_MFI'] = ts_MFI(high=df_data['s_dq_high'], low=df_data['s_dq_low'], close_price=df_data['s_dq_close'], volume=df_data['s_dq_volume'], days=ndays)
        # MA 短期移动平均线一般以5日或10日为计算期间，中期移动平均线大多以30日、60日为计算期间；长期移动平均线大多以100天和200天为计算期间。
        df_data['ts_MA'] = ts_MA( close_price=df_data['s_dq_close'], days=10)
        df_data['ts_MACD'] = ts_MACD(close_price=df_data['s_dq_close'], days_n=12, days_m=26)
        df_data['ts_EXPMA'] = ts_EXPMA(close_price=df_data['s_dq_close'], nday=10)
        # 布林带 中轨线是一条周期为20日的简单移动平均线 (SMA)
        df_data['ts_BOLL_UP'], df_data['ts_BOLL_MB'], df_data['ts_BOLL_DOWN'] = ts_BOLL(close_price=df_data['s_dq_close'], nday=20)

        ## clean NA generated NA values --> this step will be used outside
        # df_data = df_data.dropna()

        return df_data

    


    def get_datalist(self, mode="train"):
        if mode == "train":
            return self.train_datalist
        elif mode == "valid":
            return self.valid_datalist
        elif mode == "test":
            return self.test_datalist
        else:
            ## full mode
            return self.full_datalist


class StockDatasetAll(Dataset):
    """
    基于StockDatabase的Dataset，用于从数据库中进行抽样对模型训练；
    """
    def __init__(self, database: StockDatabase, time_length: int = 30, pred_day: int = 10 , timestep: int = 1, mode: str = "train"):
        """
        Stock dataset. Data are saved in csv files. Each file contains one single stock price info.
            Assume data are sorted in positive order.
            Read all data together into memory once. And then read sequences in batch for loader.
            After reading all data files, create an index list with format : (stock_index, time_index)  which is used for loading data.
            Each Step return data with shape (, Time, Characts)
        """
        super().__init__()
        # self.folder = data_root
        self.time_length = time_length
        self.pred_day = pred_day
        assert timestep > 0
        self.timestep = timestep

        # read all data into memory so memory must be large enough
        self.data_list = database.get_datalist(mode=mode)
        self.index_list = []
        for i, df_data in enumerate(self.data_list):

            # check with time step
            total_length = df_data.shape[0]
            for j in range(0, total_length - self.time_length - self.pred_day, timestep):
                ## 去除停牌时间超过10天的区间。
                # NOTE: Here is the problem
                # if self.data_list[i]['suspend'].iloc[j:j+self.time_length+self.pred_day].sum() > pred_day :
                #     continue
                self.index_list.append( (i, j) )

        # print(f"Finish reading data with {len(self.datanames)} examples.")
        

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        stock_index, time_index = self.index_list[idx]
        df_data = self.data_list[stock_index]
        # input data
        data_in = df_data.iloc[time_index:(time_index+self.time_length), :]
        data_in = torch.from_numpy(data_in.to_numpy())
    
        # gt label -- 以预测10天后对比最后一天的收益率为预测
        gt_return = df_data['s_dq_close'].iloc[time_index+self.time_length+self.pred_day-1] / df_data['s_dq_close'].iloc[time_index+self.time_length-1] - 1
        gt_return = torch.tensor([gt_return])
        # gt_price = df_data['s_dq_close'][time_index+self.time_length+self.pred_day-1]

        ## normalize input
        # epsilon = 1e-8
        # data_in = (data_in - data_in.mean(dim=0, keepdim=True)) / (data_in.std(dim=0, keepdim=True) + epsilon)
        
        return data_in, gt_return



class StockDatabaseSimple(StockDatabase):
    """
    Read all stock data at once, and save in a list.
    股票数据库，一次读取所有数据后提供给dataset使用；这个Simple没有在数据里添加额外技术性因子。

    """

    def __init__(self, data_root: str, split_ratio: List[int or float], minlength: int = 40) -> None:
        additional_factors = False
        super().__init__(data_root, split_ratio, minlength, additional_factors)




if __name__ == '__main__':
    dataset = StockDatasetAll('../data-sample/test/')
    data, gt_price = dataset[1]

    dataloader = DataLoader(dataset,batch_size=2)
    num_batch = len(dataloader)

    for i, batch in enumerate(dataloader):
        data_in, gt_price = batch
        print(data_in.shape)
        print(gt_price.shape)
        print(i)

    print("End of test.")