'''
Analyze the given stock data.
'''

import os
import numpy as np
import pandas as pd


#### **** Prapare ----
DATA_FILE = "../../status_data_all.csv"


#### ----

data = pd.read_csv(DATA_FILE, encoding='gbk')

#### analyze all stock names
#%%
# import numpy as np
import pandas as pd

stockname_file = './data/stock_name.csv'
stocknames = pd.read_csv(stockname_file)
# 查明股票代码在 'ts_code' 字段中，'list_date'提供了上市日期，
# 一共7列，第一列是顺序号，第二列是 ts_code；第三列是 symbol ；第四列是 name ；第五列是area；第六列是industry；第七列是list_date
tscode = stocknames.ts_code


#%%
data_file = 'data-sample/test/000001.SZ.csv'
data = pd.read_csv(data_file, encoding='utf-8')
timelength = len(data)
start_time = 0
time_step = 50
subdata = data.iloc[start_time:(start_time+time_step), :]

feature_columns = ['s_dq_close', 's_dq_open', 's_dq_low', 's_dq_high', 's_dq_volume', 'dq_vwap', 'dq_return', 'dq_turnover', 'dq_free_turnover']
subdata = data[feature_columns]

data['s_dq_turnover'] = data['s_dq_volume'] / data['float_a_shr_today']

test = data['s_dq_volume'] / data['free_shares_today']


## 说明
subdata['s_dq_close_1'] = subdata['s_dq_close'].shift(1)

subdata['trade_dt'].shift(-1)


