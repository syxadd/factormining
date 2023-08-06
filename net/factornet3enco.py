import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

#### ----
## Factor Extraction module
class FactorExtraction(nn.Module):
    def __init__(self, in_features, days=10, time_length=30,):
        super().__init__()
        self.in_ch = in_features
        self.time_length = time_length

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

        ## additional technical indicators

        self.out_features = combine_cov_ch*2 + self.in_ch * 5

    def forward(self, x):
        # input with shape (B, T, C)
        x1 = self.tscorr10(x)
        x2 = self.ts_cov10(x)
        x3 = self.ts_stddev10(x)
        x4 = self.ts_zscore10(x)
        x5 = self.ts_return10(x)
        x6 = self.ts_decaylinear10(x)
        x7 = self.ts_mean(x)

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)
        return x_cat


class FactorNetLSTM_3encoder(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(FactorNetLSTM_3encoder, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        # First part
        self.extraction = FactorExtraction(self.in_ch, time_length=self.time_length)

        self.ch1 = self.extraction.out_features

        ## autoencoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1, out_channels=self.ch1//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.ch1//2),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1//2, out_channels=self.ch1 // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.ch1 // 4),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1//4, out_channels=self.ch1 // 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.ch1 // 8),
            nn.ReLU()
        )
        # self.encoder4 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ch1 // 8, out_channels=self.ch1 // 16, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(num_features=self.ch1 // 16),
        #     nn.ReLU()
        # )

        self.ch2 = self.ch1 // 8

        ## LSTM
        self.hidden_units = 30
        self.lstm = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)

        self.ch2_out = self.hidden_units * 3

        ## MLP layer
        # self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        # self.inner_ch = 30
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
        #     nn.ReLU()
        # )

        # output predict
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.ch2_out, out_features=1),
        )
        
        ## init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight)



    def forward(self, x):
        ## assuming x time series is reversed; with shape (batch, time value, features)
        # x1 = self.tscorr10(x)
        # x2 = self.ts_cov10(x)
        # x3 = self.ts_stddev10(x)
        # x4 = self.ts_zscore10(x)
        # x5 = self.ts_return10(x)
        # x6 = self.ts_decaylinear10(x)
        # x7 = self.ts_mean(x)
        xfeat = self.extraction(x) 
        xfeat = xfeat.transpose(1, 2)  # (B, C, T)

        xout = self.encoder1(xfeat)
        xout = self.encoder2(xout)
        xout = self.encoder3(xout)
        # xout = self.encoder4(xout)

        xout = xout.transpose(1, 2) # to (B, T, C)
        xout, hidden = self.lstm(xout)

        xout = self.bn2(xout.transpose(1, 2))  # out with (B, C, T)
        xout = xout.flatten(start_dim=1)

        out = self.linear2(xout)

        return out



