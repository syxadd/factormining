from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import pdb

#### ----
## Factor Extraction module
class FactorExtraction(nn.Module):
    def __init__(self, in_features, days=10, time_length=30, stride=0, padding=0):
        super().__init__()
        self.in_ch = in_features
        self.time_length = time_length
        self.days = days
        self.stride = days if stride <= 0 else stride
        self.padding = (0, 0, padding, padding) if padding > 0 else None

        combine_cov_ch = self.in_ch * (self.in_ch-1) // 2
        self.tscorr10 = nn.Sequential(
            TS_Corr(days=days, stride=self.stride),
            TS_Batchnorm1d(combine_cov_ch),
        )
        self.ts_cov10 = nn.Sequential(
            TS_Cov(days=days, stride=self.stride),
            TS_Batchnorm1d(combine_cov_ch)
        )
        self.ts_stddev10 = nn.Sequential(
            TS_Stddev(days=days, stride=self.stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_zscore10 = nn.Sequential(
            TS_Zscore(days=days, stride=self.stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_return10 = nn.Sequential(
            TS_Return(days=days, stride=self.stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_decaylinear10 = nn.Sequential(
            TS_Decaylinear(days=days, stride=self.stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_mean = nn.Sequential(
            TS_Mean(days=days, stride=self.stride),
            TS_Batchnorm1d(self.in_ch)
        )

        ## additional technical indicators

        self.out_features = combine_cov_ch*2 + self.in_ch * 5

    def forward(self, x):
        # input with shape (B, T, C)

        if self.padding:
            x = F.pad(x, self.padding, "constant", 0)
        
        x1 = self.tscorr10(x)
        x2 = self.ts_cov10(x)
        x3 = self.ts_stddev10(x)
        x4 = self.ts_zscore10(x)
        x5 = self.ts_return10(x)
        x6 = self.ts_decaylinear10(x)
        x7 = self.ts_mean(x)

        # pdb.set_trace()

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)
        return x_cat


class MultiEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch // 16

        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.in_ch//2),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch//2, out_channels=self.in_ch // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.in_ch // 4),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch//4, out_channels=self.in_ch // 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.in_ch // 8),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch // 8, out_channels=self.in_ch // 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=self.in_ch // 16),
            nn.ReLU()
        )

    def forward(self, x):
        # input with dim (B, C, T)
        xout = self.encoder1(x)
        xout = self.encoder2(xout)
        xout = self.encoder3(xout)
        xout = self.encoder4(xout)
        
        return xout


class FactorNetID16_LargerLevelLearnable220911(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(FactorNetID16_LargerLevelLearnable220911, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        self.layernorm = nn.LayerNorm(normalized_shape=in_features)

        # First part
        self.extraction10 = FactorExtraction(self.in_ch, days=10, time_length=time_length, stride=10)
        self.extraction15 = FactorExtraction(self.in_ch, days=15, time_length=time_length, stride=15)
        self.extraction30 = FactorExtraction(self.in_ch, days=30, time_length=time_length, stride=30)

        self.params_extraction10 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=10, stride=10, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )
        self.params_extraction15 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=15, stride=15, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )
        self.params_extraction30 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=30, stride=30, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )


        self.feat_ch = self.extraction10.out_features

        self.ch2 = self.feat_ch + self.in_ch

        self.encoder10 = MultiEncoder(self.ch2)
        self.encoder15 = MultiEncoder(self.ch2)
        self.encoder30 = MultiEncoder(self.ch2)


        # self.ch2 = self.ch1 // 16
        

        ## LSTM
        # self.hidden_units = 30
        # self.lstm = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)

        # self.ch2_out = self.encoder10.out_ch * (1+2+3)
        self.ch2_out = self.encoder10.out_ch * (time_length // 10 + time_length // 15 + time_length // 30)

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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)



    def forward(self, x):
        ## assuming x time series is reversed; with shape (batch, time value, features)

        xfeat10 = self.extraction10(x).transpose(1, 2)
        xfeat15 = self.extraction15(x).transpose(1, 2)
        xfeat30 = self.extraction30(x).transpose(1, 2)

        xnorm = self.layernorm(x).transpose(1, 2) # to (B, C, T)

        xfeatlearn10 = self.params_extraction10(xnorm)
        xfeatlearn15 = self.params_extraction15(xnorm)
        xfeatlearn30 = self.params_extraction30(xnorm)

        xfeat10 = torch.cat([xfeat10, xfeatlearn10], dim=1)
        xfeat15 = torch.cat([xfeat15, xfeatlearn15], dim=1)
        xfeat30 = torch.cat([xfeat30, xfeatlearn30], dim=1)

        ## to (B, C, T)

        ## enoder
        xfeat10 = self.encoder10(xfeat10)
        xfeat15 = self.encoder15(xfeat15)
        xfeat30 = self.encoder30(xfeat30)

        ## ch2



        # xout = xout.transpose(1, 2) # to (B, T, C)
        # xout, hidden = self.lstm(xout)

        # xout = self.bn2(xout.transpose(1, 2))  # out with (B, C, T)
        # xout = xout.flatten(start_dim=1)

        xout = torch.cat([xfeat10.flatten(start_dim=1), xfeat15.flatten(start_dim=1), xfeat30.flatten(start_dim=1)], dim=1)

        out = self.linear2(xout)

        return out



