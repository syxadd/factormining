import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

#TODO: 
# 1 lstm 模块
# 2 数据读取需要增加一种能提供多个gt的
# 3 特征提取的维度也要变

class AlphaNetv2(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(AlphaNetv2, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        # First part
        combine_cov_ch = self.in_ch * (self.in_ch-1) // 2
        self.tscorr10 = nn.Sequential(
            TS_Corr(days=10, stride=10),
            TS_Batchnorm1d(combine_cov_ch),
        )
        self.ts_cov10 = nn.Sequential(
            TS_Cov(days=10, stride=10),
            TS_Batchnorm1d(combine_cov_ch)
        )
        self.ts_stddev10 = nn.Sequential(
            TS_Stddev(days=10, stride=10),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_zscore10 = nn.Sequential(
            TS_Zscore(days=10, stride=10),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_return10 = nn.Sequential(
            TS_Return(days=10, stride=10),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_decaylinear10 = nn.Sequential(
            TS_Decaylinear(days=10, stride=10),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_mean = nn.Sequential(
            TS_Mean(days=10, stride=10),
            TS_Batchnorm1d(self.in_ch)
        )
        # Second part
        self.ch2 = self.in_ch*5 + combine_cov_ch*2
        # self.ts2_mean = nn.Sequential(
        #     TS_Mean(days=3, stride=3),
        #     TS_Batchnorm1d(self.ch2)
        # )
        # self.ts2_max = nn.Sequential(
        #     TS_Max(days=3, stride=3),
        #     TS_Batchnorm1d(self.ch2)
        # )
        # self.ts2_min = nn.Sequential(
        #     TS_Min(days=3, stride=3),
        #     TS_Batchnorm1d(self.ch2)
        # )

        ## LSTM
        self.hidden_units = 30
        self.lstm = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)

        self.ch2_out = self.hidden_units

        ## MLP layer
        # self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        # self.inner_ch = 30
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
        #     nn.ReLU()
        # )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.ch2_out, out_features=1),
        )
        
        ## init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight)



    def forward(self, x):
        ## assuming x time series is reversed; with shape (batch, time value, features)
        x1 = self.tscorr10(x)
        x2 = self.ts_cov10(x)
        x3 = self.ts_stddev10(x)
        x4 = self.ts_zscore10(x)
        x5 = self.ts_return10(x)
        x6 = self.ts_decaylinear10(x)
        x7 = self.ts_mean(x)

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2) # argument dim and axis are both usable

        xout, hidden = self.lstm(x_cat)
        xout = self.bn2(xout.transpose(1, 2))  # out with (B, C, T)
        # xout = F.avg_pool1d(xout, kernel_size=xout.shape[2])


        out = self.linear2(xout[:,:,-1])

        return out



