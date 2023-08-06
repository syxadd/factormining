import torch
import torch.nn as nn
from .layers import *

class AlphaNetv1(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(AlphaNetv1, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        # First part
        combine_cov_ch = self.in_ch * (self.in_ch-1) // 2
        self.tscorr10 = nn.Sequential(
            TS_Corr(days=10, stride=10),
            TS_Batchnorm1d(combine_cov_ch)
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
        self.ts2_mean = nn.Sequential(
            TS_Mean(days=3, stride=3),
            TS_Batchnorm1d(self.ch2)
        )
        self.ts2_max = nn.Sequential(
            TS_Max(days=3, stride=3),
            TS_Batchnorm1d(self.ch2)
        )
        self.ts2_min = nn.Sequential(
            TS_Min(days=3, stride=3),
            TS_Batchnorm1d(self.ch2)
        )

        self.ch2_out = self.ch2 * 3

        ## MLP layer
        self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        self.inner_ch = 30
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.inner_ch, out_features=1),
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

        # print("shape x1", x1.shape)
        # print("shape x2", x2.shape)
        # print("shape x3", x3.shape)
        # print("shape x4", x4.shape)
        # print("shape x5", x5.shape)
        # print("shape x6", x6.shape)
        # print("shape x7", x7.shape)

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2) # argument dim and axis are both usable
        # print("first layer output features ", x_cat.shape)

        x2_1 = self.ts2_mean(x_cat)
        x2_2 = self.ts2_max(x_cat)
        x2_3 = self.ts2_min(x_cat)

        # print("shape xpool1", x2_1.shape)
        # print("shape xpool2", x2_2.shape)
        # print("shape xpool3", x2_3.shape)
        #
        x_cat2 = torch.cat([x2_1, x2_2, x2_3], dim=2)
        # print("shape x_cat2:", x_cat2.shape)

        x_cat1pool = torch.flatten(x_cat, start_dim=1)
        x_cat2pool = torch.flatten(x_cat2, start_dim=1)
        # print("shape x1flatten:", x_cat1pool.shape)
        # print("shape x2flatten:", x_cat2pool.shape)

        xpool = torch.cat([x_cat1pool, x_cat2pool], dim=1)
        out = self.linear2(self.linear1(xpool))

        return out






