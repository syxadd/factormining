import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

#TODO: 
# 1 GRU模块
# 2 两个维度的特征提取

class AlphaNetv3GRU(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(AlphaNetv3GRU, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        # First part
        ## day10 part
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
        # self.ts_mean = nn.Sequential(
        #     TS_Mean(days=10, stride=10),
        #     TS_Batchnorm1d(self.in_ch)
        # )

        ## day5 part
        self.tscorr5 = nn.Sequential(
            TS_Corr(days=5, stride=5),
            TS_Batchnorm1d(combine_cov_ch)
        )
        self.ts_cov5 = nn.Sequential(
            TS_Cov(days=5, stride=5),
            TS_Batchnorm1d(combine_cov_ch)
        )
        self.ts_stddev5 = nn.Sequential(
            TS_Stddev(days=5, stride=5),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_zscore5 = nn.Sequential(
            TS_Zscore(days=5, stride=5),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_return5 = nn.Sequential(
            TS_Return(days=5, stride=5),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_decaylinear5 = nn.Sequential(
            TS_Decaylinear(days=5, stride=5),
            TS_Batchnorm1d(self.in_ch)
        )
        

        # Second part
        self.ch2 = self.in_ch*4 + combine_cov_ch*2

        ## GRU
        self.hidden_units = 30
        # self.lstm = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)

        self.gru10 = nn.GRU(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        self.bn10 = nn.BatchNorm1d(self.hidden_units)
        
        self.gru5 = nn.GRU(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        self.bn5 = nn.BatchNorm1d(self.hidden_units)


        self.ch2_out = self.hidden_units

        ## MLP layer
        # self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        # self.inner_ch = 30
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
        #     nn.ReLU()
        # )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.ch2_out * 2, out_features=1),
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
        # x7 = self.ts_mean(x)

        y1 = self.tscorr5(x)
        y2 = self.ts_cov5(x)
        y3 = self.ts_stddev5(x)
        y4 = self.ts_zscore5(x)
        y5 = self.ts_return5(x)
        y6 = self.ts_decaylinear5(x)

        
        # x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2) # argument dim and axis are both usable
        x_cat = torch.cat([x1, x2, x3, x4, x5, x6], dim=2) # argument dim and axis are both usable ; # input timesteps=3
        y_cat = torch.cat([y1, y2, y3, y4, y5, y6], dim=2)  # input timesteps=6

        xout, hidden = self.gru10(x_cat)
        xout = self.bn10(xout.transpose(1, 2))  # return (B, C, T)

        yout, hidden = self.gru5(y_cat)
        yout = self.bn5(yout.transpose(1, 2))  # return (B, C, T)

        out = torch.cat([xout[:,:,-1], yout[:,:,-1]], dim=1)
        out = self.linear2(out)

        return out



