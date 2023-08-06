import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

"""
220821

先增加多尺度的方式处理数据。
先用不重合的多尺度方式处理。

增加多尺度的处理方式
"""
#### ----
## Factor Extraction module
class FactorExtraction(nn.Module):
    def __init__(self, in_features, days=10, time_length=30, stride=10):
        super().__init__()
        self.in_ch = in_features
        self.time_length = time_length
        self.days = days

        combine_cov_ch = self.in_ch * (self.in_ch-1) // 2
        self.tscorr10 = nn.Sequential(
            TS_Corr(days=days, stride=stride),
            TS_Batchnorm1d(combine_cov_ch),
        )
        self.ts_cov10 = nn.Sequential(
            TS_Cov(days=days, stride=stride),
            TS_Batchnorm1d(combine_cov_ch)
        )
        self.ts_stddev10 = nn.Sequential(
            TS_Stddev(days=days, stride=stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_zscore10 = nn.Sequential(
            TS_Zscore(days=days, stride=stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_return10 = nn.Sequential(
            TS_Return(days=days, stride=stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_decaylinear10 = nn.Sequential(
            TS_Decaylinear(days=days, stride=stride),
            TS_Batchnorm1d(self.in_ch)
        )
        self.ts_mean = nn.Sequential(
            TS_Mean(days=days, stride=stride),
            TS_Batchnorm1d(self.in_ch)
        )
        # self.ts_identity = nn.Sequential(
        #     nn.Identity(),
        #     TS_Batchnorm1d(self.in_ch)
        # )

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
        x8 = self.ts_identity(x)

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=2)
        return x_cat


class FeatureModuleUNet30(nn.Module):
    """
    多尺度类Unet处理一维序列。
    我先不加因子的处理方法，直接拿UNet来处理真实数据我看看效果如何。
    """
    def __init__(self, in_features, out_features, time_length=30, dropout = 0.0):
        super().__init__()
        self.in_ch = in_features
        self.inner_features = out_features
        self.out_ch = out_features

        self.size1 = time_length  # 30
        self.size2 = time_length // 2  # 15
        self.size3 = time_length // 4 + 1  # 8
        self.size4 = self.size3 // 2  # 4
        
        # t=30, c
        # self.feat_extract = nn.Sequential(

        # )
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=self.inner_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features),
            nn.ReLU(),
        )
        # t=30 -> t=15
        self.down1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features, out_channels=self.inner_features*2, kernel_size=3, stride=2, padding=1),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*2, out_channels=self.inner_features*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features*2),
            nn.ReLU(),
        )
        # t=15 -> t=8
        self.down2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*2, out_channels=self.inner_features*4, kernel_size=3, stride=2, padding=1),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*4, out_channels=self.inner_features*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features*4),
            nn.ReLU(),
        )
        # t=8 -> t=4
        self.down3 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*4, out_channels=self.inner_features*8, kernel_size=3, stride=2, padding=1),
        )
        # t=4
        self.inner_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*8, out_channels=self.inner_features*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features*8),
            nn.ReLU(),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(self.size3, mode='linear'),
        )
        self.reduce3 = nn.Sequential(
            nn.Conv1d(self.inner_features*4*3, self.inner_features*4, kernel_size=1, stride=1),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*4, out_channels=self.inner_features*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features*4),
            nn.ReLU(),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(size=self.size2, mode='linear'),
        )
        self.reduce2 = nn.Sequential(
            nn.Conv1d(self.inner_features*2*3, self.inner_features*2, kernel_size=1, stride=1),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features*2, out_channels=self.inner_features*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features*2),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(size=self.size1, mode='linear'),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv1d(self.inner_features*3, self.inner_features, kernel_size=1, stride=1),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inner_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.inner_features),
            nn.ReLU(),
        )

    
    def forward(self, x):
        # x with shape (B, C, T)
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.inner_conv(self.down3(x3))

        x33 = torch.cat([x3, self.up3(x4)], dim=1)
        x33 = self.decoder3(self.reduce3(x33))
        x22 = torch.cat([x2, self.up2(x33)], dim=1)
        x22 = self.decoder2(self.reduce2(x22))
        x11 = torch.cat([x1, self.up1(x22)], dim=1)
        x11 = self.decoder1(self.reduce1(x11))

        return x11





class FactorNetLSTM_UNet(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 time_length=30,
                 dropout = 0.0,
                 ):
        super(FactorNetLSTM_UNet, self).__init__()
        self.in_ch = in_features
        self.out_ch = out_features
        self.time_length = time_length

        # First part
        # self.extraction = FactorExtraction(self.in_ch, time_length=self.time_length)
        self.inner_ch1 = 64
        self.extraction = FeatureModuleUNet30(in_features, self.inner_ch1, time_length=time_length)



        ## autoencoder
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
        # self.decoder3 = nn.Sequential(
        #     nn.Conv1d(in_channels=)
        # )

        ## LSTM
        self.inner_ch2 = 64
        self.hidden_units = self.inner_ch2
        self.lstm = nn.LSTM(input_size=self.inner_ch1, hidden_size=self.hidden_units, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)

        self.ch2_out = self.hidden_units * time_length  # * 30 days

        ## MLP layer
        # self.ch3 = self.ch2 * (self.time_length// 10) + self.ch2_out
        # self.inner_ch = 30
        # self.linear1 = nn.Sequential(
        #     nn.Linear(in_features=self.ch3, out_features=self.inner_ch),
        #     nn.ReLU()
        # )

        # output predict
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.ch2_out, out_features=out_features),
        )
        
        ## init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)



    def forward(self, x):
        ## assuming x time series is reversed; with shape (batch, time value, features)
        # x1 = self.tscorr10(x)
        # x2 = self.ts_cov10(x)
        # x3 = self.ts_stddev10(x)
        # x4 = self.ts_zscore10(x)
        # x5 = self.ts_return10(x)
        # x6 = self.ts_decaylinear10(x)
        # x7 = self.ts_mean(x)

        ## x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        xfeat = self.extraction(x)
        xfeat = xfeat.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        # xout = self.encoder1(xfeat)
        # xout = self.encoder2(xout)
        # xout = self.encoder3(xout)
        # xout = self.encoder4(xout)
        xout, _ = self.lstm(xfeat)

        xout = self.bn2(xout.transpose(1, 2))  # out with (B, C, T)
        xout = xout.flatten(start_dim=1) # -> (B, C*T)

        out = self.linear2(xout)

        return out



if __name__ == "__main__":
    net = FactorNetLSTM_UNet(39, 1, 30)
    x = torch.randn(2, 30, 39)
    print(x.shape)
    y = net(x)
    print(y.shape)