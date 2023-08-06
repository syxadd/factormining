import logging
import os
import pandas as pd
from tqdm import tqdm

# from utils.stock_dataset import StockDatabase
'''
Export all data to each stock data with descending order.
导出所有数据，到每个个股数据单独为一个文件夹，同时添加其他一些数据信息。

最后修改 220817

必须在主目录运行该脚本。


备注：在220821时保留该代码用于回顾往期处理的数据。之后用新的'prepare_trainData.py'来导出训练数据。
'''


data_file = '../quant-data/status_data_all.csv'
national_debt_file = r'../quant-data/中国一年期国债收益率历史数据.csv'
stock_index_file = r"../quant-data/上证指数历史数据.csv"

# out_folder = '../stockdata/wukong/'
out_folder = '../stockdata/wukong-220821/'



#### ----------------

#### 读取交易数据
## 全体股票数据
data_all = pd.read_csv(data_file, encoding='gbk')
stocknames  = data_all['s_info_windcode']
stocknames = stocknames.drop_duplicates()


#### 读取外部数据
## 国债数据处理
df_debt = pd.read_csv(national_debt_file)
df_debt = df_debt.sort_values(by="日期", ascending=True)
df_debt['日期'] = pd.to_datetime(df_debt['日期'])
# df_debt = df_debt.reset_index(drop=True)  # unessential
df_debt = df_debt.set_index('日期', drop=False)
## 指数数据的处理
df_stock_index = pd.read_csv(stock_index_file)
df_stock_index = df_stock_index.sort_values(by="日期", ascending=True)
timedate = pd.to_datetime(df_stock_index['日期'])
df_stock_index['日期'] = timedate
df_stock_index = df_stock_index.set_index('日期', drop=False)
df_stock_index



## 准备日志
print("Exporting data to sub files...")
logger = logging.getLogger('warn_log')
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler('warn_file.log')
file_handler.setLevel(level=logging.INFO)
logger.addHandler(file_handler)
# make out folder
if not os.path.exists(out_folder):
	os.makedirs(out_folder, exist_ok=True)




## 分别处理个股信息
for subname in tqdm(stocknames):
	## 读取子数据并处理日期
	subdata = data_all[data_all['s_info_windcode'] == subname]
	subdata = subdata.sort_values(by='trade_dt', ascending=True)
	# subdata = subdata.reset_index(drop=True)
	trade_dt = subdata['trade_dt']
	trade_dt = pd.to_datetime(trade_dt.apply(str))
	subdata['trade_dt'] = trade_dt
	subdata = subdata.set_index('trade_dt', drop=False)


	# PROCESS SOME values -- 添加一些信息
	# 换手率
	subdata['dq_turnover'] = subdata['s_dq_volume'] / subdata['float_a_shr_today']
	# 自由流通股换手率
	subdata['dq_free_turnover'] = subdata['s_dq_volume'] / subdata['free_shares_today']
	# 每日回报率 - computed on closed price
	subdata['s_dq_close_1'] = subdata['s_dq_close'].shift(-1) # shift index to increase with positive value, or shift in the opposite direction with negative value
	subdata['dq_return1'] = subdata['s_dq_close'] / subdata['s_dq_close_1'] - 1
	# add vwap value
	subdata['dq_vwap'] = subdata['s_dq_amount'] / subdata['s_dq_volume']
	## 添加停牌信息
	subdata['suspend'] = subdata['s_info_suspension'] > 0
	# Drop na
	# subdata.dropna()

	#### ---- 
	#### 添加各种技术指标


	## 导入外部信息数据
	# TODO: 添加外部数据
	# location = df_debt['日期'].isin(trade_dt)
	# sub_debt_close = df_debt['收盘'][location]
	# subdata['national_debt_return_close'] = sub_debt_close
	## 添加国债
	location = df_debt['日期'].isin(subdata['trade_dt'])
	sub_debt_close = df_debt['收盘'][location]
	subdata['national_debt_return_close'] = sub_debt_close / 100
	# print("Added debt infos.")
	## 添加指数数据
	location = df_stock_index['日期'].isin(subdata['trade_dt'])
	sub_index_close = df_stock_index['收盘'][location]
	subdata['shangzheng_index_close'] = sub_index_close / 1.
	# print("Added stock index infos.")
	



	## Ignore stocks that does not have enough data
	time_length = 90
	if subdata.shape[0] < time_length:
		logger.info(f'File: {subname} has only {subdata.shape[0]} lines, but require {time_length}. So remove it.')
		continue


	## write df data to files
	outfile_name = out_folder + '/' + subname + '.csv'
	# save with utf-8 encoding
	subdata.to_csv(outfile_name, index=False, encoding='utf8')

print("Finish the process.")