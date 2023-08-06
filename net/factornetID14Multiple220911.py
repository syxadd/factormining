from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import pdb

"""
220911 
"""
#### ----
## Factor Extraction module
# 无参数化的特征因子选取
class FactorExtractionWindow(nn.Module):
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

## 
# class FactorExtractModule(nn.Module):
#     """
#     特征因子选取，包含三个部分，原始输入，无参的特征因子选取，含参的特征因子选取。

#     """
#     def __init__(self, in_features, windows_size, time_length):
#         super().__init__()


class FactorNetID14_Multiple3Level220911(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(FactorNetID14_Multiple3Level220911, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        self.input_norm = nn.LayerNorm(normalized_shape=[time_length, self.in_ch])

        # First part
        
        self.extraction_k9 = FactorExtractionWindow(self.in_ch, days=9, time_length=time_length, stride=1, padding=4)
        self.extraction_k5 = FactorExtractionWindow(self.in_ch, days=5, time_length=time_length, stride=1, padding=2)
        self.extraction_k3 = FactorExtractionWindow(self.in_ch, days=3, time_length=time_length, stride=1, padding=1)


        ## Group Conv
        self.param_extraction_k9 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=9, stride=1, padding=4, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )
        self.param_extraction_k5 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=5, stride=1, padding=2, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )
        self.param_extraction_k3 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=3, stride=1, padding=1, groups=self.in_ch),
            nn.BatchNorm1d(self.in_ch),
            nn.ReLU(),
        )

        ## kernel selection

        self.ch1_params = self.in_ch
        self.ch1 = self.extraction_k9.out_features * 3 + self.in_ch * 3 + self.in_ch

        self.conv11_k9 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1_params, out_channels=self.ch1_params, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.ch1_params),
            nn.ReLU(),
        )
        self.conv11_k5 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1_params, out_channels=self.ch1_params, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.ch1_params),
            nn.ReLU(),
        )
        self.conv11_k3 = nn.Sequential(
            nn.Conv1d(in_channels=self.ch1_params, out_channels=self.ch1_params, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.ch1_params),
            nn.ReLU(),
        )

        # self.conv_input = nn.Sequential(
        #     nn.Conv1d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(self.in_ch),
        #     nn.ReLU(),
        # )

        ## Cross 1x1 conv over all channels
        # self.encoder1 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ch1, out_channels=self.ch1//2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(num_features=self.ch1//2),
        #     nn.ReLU()
        # )
        # self.encoder2 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ch1//2, out_channels=self.ch1 // 4, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(num_features=self.ch1 // 4),
        #     nn.ReLU()
        # )
        # self.encoder3 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ch1//4, out_channels=self.ch1 // 8, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(num_features=self.ch1 // 8),
        #     nn.ReLU()
        # )
        # self.encoder4 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.ch1 // 8, out_channels=self.ch1 // 16, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(num_features=self.ch1 // 16),
        #     nn.ReLU()
        # )

        self.ch2 = self.ch1

        ## LSTM
        # self.hidden_units = 64
        # self.lstm_k10 = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.lstm_k5 = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.lstm_k3 = nn.LSTM(input_size=self.ch2, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.bn2_k10 = nn.BatchNorm1d(num_features=self.hidden_units)
        # self.bn2_k5 = nn.BatchNorm1d(num_features=self.hidden_units)
        # self.bn2_k3 = nn.BatchNorm1d(num_features=self.hidden_units)

        # self.lstm_input = nn.LSTM(input_size=self.in_ch, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        # self.bn2_input = nn.BatchNorm1d(num_features=self.hidden_units)

        # self.ch2_out = self.hidden_units * (3 + 6 + 10 + 30)

        ## MLP layer
        # self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        # self.inner_ch = 30
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
        #     nn.ReLU()
        # )

        self.ch2_out = self.ch2 * 30

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
        # (B, T, C)
        ## layernormalization
        x = self.input_norm(x)

        xfixfactor9 = self.extraction_k9(x).transpose(1, 2)
        xfixfactor5 = self.extraction_k5(x).transpose(1, 2)
        xfixfactor3 = self.extraction_k3(x).transpose(1, 2)

        x1 = x.transpose(1, 2)

        xfeat9 = self.param_extraction_k9(x1)
        xfeat9 = self.conv11_k9(xfeat9)
        xfeat5 = self.param_extraction_k5(x1)
        xfeat5 = self.conv11_k5(xfeat5)
        xfeat3 = self.param_extraction_k3(x1)
        xfeat3 = self.conv11_k3(xfeat3)


        # xfeat10 = self.conv11_k9(self.extraction_k10(x).transpose(1, 2))
        # xfeat5 = self.conv11_k5(self.extraction_k5(x).transpose(1, 2))
        # xfeat3 = self.conv11_k3(self.extraction_k3(x).transpose(1, 2))

        # xfeat_in = self.conv_input(self.input_bn(x.transpose(1, 2)))

        # xout = self.encoder1(xfeat)
        # xout = self.encoder2(xout)
        # xout = self.encoder3(xout)
        # xout = self.encoder4(xout)

        # xout_10, _ = self.lstm_k10(xfeat10.transpose(1, 2)) # -> (b, t, c)
        # xout_10 = self.bn2_k10(xout_10.transpose(1, 2)) # -> (b, c, t)
        # xout_5, _ = self.lstm_k5(xfeat5.transpose(1, 2))
        # xout_5 = self.bn2_k5(xout_5.transpose(1, 2))
        # xout_3, _ = self.lstm_k3(xfeat3.transpose(1, 2))
        # xout_3 = self.bn2_k3(xout_3.transpose(1, 2))

        # xout_in, _ = self.lstm_input(xfeat_in.transpose(1, 2))
        # xout_in = self.bn2_input(xout_in.transpose(1, 2))

        # xout = torch.cat([xout_10.flatten(start_dim=1), xout_5.flatten(start_dim=1), xout_3.flatten(start_dim=1), xout_in.flatten(start_dim=1)], dim=1)


        # xout = xout.transpose(1, 2) # to (B, T, C)
        # xout, hidden = self.lstm(xout)

        # xout = self.bn2(xout.transpose(1, 2))  # out with (B, C, T)
        # xout = xout.flatten(start_dim=1)
        pdb.set_trace()

        xout = torch.cat([x1, xfixfactor9, xfixfactor5, xfixfactor3, xfeat9, xfeat5, xfeat3], dim=1)
        xout = xout.flatten(start_dim=1)

        out = self.linear2(xout)

        return out



