import os
import numpy as np
import pandas as pd
import tushare as ts

'''
Get stock data from tushare database. 
First get each year data;
Then split all data into each individual dataset. 
Read dataset is set to 
'''
## Initialize
pro = ts.pro_api('fd626d40f2ceae3b665087f766a8b4c5d1288d078e3d73b337f792b9')


#%%
## 查询当前所有正常上市交易的股票列表
# stock_data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# stock_data.to_csv("data/stock_name.csv", encoding='utf-8')





#%%
## Get all data in time step
##
# total near 4900 stocks
stockcodes = pd.read_csv('./data/stock_name.csv')['ts_code']

## get daily data
def get_daily(self, ts_code='', trade_date='', start_date='', end_date=''):
    import time
    retry = 5
    for _ in range(retry):
        try:
            if trade_date:
                df = self.pro.daily(ts_code=ts_code, trade_date=trade_date)
            else:
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except:
            time.sleep(1)
        else:
            # if try getting successfully, then return the data
            return df


start_date = '20160101'
end_date = '20180101'
