import torch
import torch.nn as nn


__all__ = ['TS_Mean', 'TS_Stddev', 'TS_Cov', 'TS_Corr', 'TS_Zscore', 'TS_Min', 'TS_Max', 'TS_Sum', 'TS_Batchnorm1d', 'TS_Return','TS_Decaylinear']



# def unfold1d(x: torch.Tensor, kernel_size: int, step: int = 1, padding: int = 0):
#     ## input x with shape (B, T, C)
#     # padding is padding size, with value 0
#     if padding > 0:
#         x = ?



class TS_Mean(nn.Module):
    def __init__(self, days, stride):
        assert days > 1
        super(TS_Mean, self).__init__()
        self.days = days
        self.stride = stride
        # self.padding = padding

    def forward(self, x):
        assert len(x.shape) == 3
        b, t, cin = x.shape
        newx = x.unfold(1, self.days, self.stride)
        out = torch.mean(newx, dim=3)
        return out

class TS_Stddev(nn.Module):
    def __init__(self, days, stride):
        assert days > 1
        super(TS_Stddev, self).__init__()
        self.days = days
        self.stride=stride

    def forward(self, x):
        assert len(x.shape) == 3
        b, t, cin = x.shape
        # sum_kernel = torch.ones(t, device=x.device)
        newx = x.unfold(1, self.days, self.stride) # with shape (b, t_out, cin, kernel_size)
        out = torch.std(newx, dim=3)
        # out = torch.mul(newx, sum_kernel/self.days).sum(dim=3) # out with shape (b, cin, tout)
        return out

def _time_inter_conv(input_x, kernel, kernel_size=3):
    # input_x has shape: (b, t_out, 2, kernel_size)
    b, t, c, k = input_x.shape
    assert c == 2
    x = input_x[:,:,0,...] # will squeeze the second dimension
    y = input_x[:,:,1,...]
    # x shape: (b, tout, kernel_size)
    out = x * y
    out = (out * kernel).sum(dim=2) # will squeeze the last dimension
    # return shape is (b, t_out, 1)
    return out.unsqueeze(2)


class TS_Cov(nn.Module):
    def __init__(self, days, stride):
        assert days > 1
        super(TS_Cov, self).__init__()
        self.days = days
        self.stride = stride

    def _gen_pairs(self, length, pairs=2):
        results = []
        # select index from 0 to length-1
        temp_series = list(range(pairs))

        def gen_series(idx_start, idx_end, pairs_start):
            if pairs_start == pairs:
                results.append(temp_series[:])
                return None
            for i in range(idx_start, idx_end):
                temp_series[pairs_start] = i
                gen_series(i + 1, idx_end, pairs_start + 1)

        gen_series(0, length, 0)

        return results

    def forward(self, x):
        assert len(x.shape) == 3
        b, t, cin = x.shape
        index_pairs = self._gen_pairs(cin)
        # kernel with shape ( days, )
        cov_kernel = torch.ones(self.days, device=x.device) / (self.days - 1)

        newx = x.unfold(1, self.days, self.stride) # get with shape (b, t_out, cin, kernel_size)
        out = newx - newx.mean(dim=3, keepdim=True)
        out = torch.cat([_time_inter_conv(out[:,:, [i,j], ...], kernel=cov_kernel) for i,j in index_pairs], dim=2)

        return out


class TS_Corr(nn.Module):
    def __init__(self, days, stride, epsilon=1e-6):
        assert days > 1 and stride > 0
        super(TS_Corr, self).__init__()
        self.days = days
        self.stride = stride
        self.epsilon = epsilon

        self.std_conv = TS_Stddev(days=self.days, stride=self.stride)

    def _gen_pairs(self, length, pairs=2):

        results = []
        # select index from 0 to length-1
        temp_series = list(range(pairs))

        def gen_series(idx_start, idx_end, pairs_start):
            if pairs_start == pairs:
                results.append(temp_series[:])
                return None
            for i in range(idx_start, idx_end):
                temp_series[pairs_start] = i
                gen_series(i + 1, idx_end, pairs_start + 1)

        gen_series(0, length, 0)

        return results

    def forward(self, x):
        # x with shape (b, t, c)
        assert len(x.shape) == 3
        b, t, cin = x.shape
        index_pairs = self._gen_pairs(cin)
        # kernel with shape ( days, )
        cov_kernel = torch.ones(self.days, device=x.device) / (self.days - 1)

        newx = x.unfold(1, self.days, self.stride) # with shape (b, cin, t_out, kernel_size)
        out = newx - newx.mean(dim=3, keepdim=True) # with shape (b, cin, t_out, kernel_size)
        newx_std = torch.std(newx, dim=3) # with shape (b, cin, t_out)
        out = torch.cat([_time_inter_conv(out[:,:, [i,j], ...], kernel=cov_kernel)/(newx_std[:,:, [i],...]*newx_std[:,:, [j],...] + self.epsilon) for i,j in index_pairs], dim=2)

        return out


class TS_Zscore(nn.Module):
    def __init__(self, days, stride, epsilon=1e-6):
        assert days > 1
        super(TS_Zscore, self).__init__()
        self.days = days
        self.stride = stride
        self.mean_conv = TS_Mean(self.days, self.stride)
        self.std_conv = TS_Stddev(self.days, self.stride)
        self.epsilon = epsilon

    def forward(self, x):
        # x with shape (b, t, c)
        out = self.mean_conv(x) / (self.std_conv(x) + self.epsilon)
        return out

# ts_decaylinear
class TS_WeightAverage(nn.Module):
    def __init__(self, days, stride):
        # default assuming that x time series is in positive order (old to now time)
        assert days > 0
        super(TS_WeightAverage, self).__init__()
        self.days = days
        self.stride = stride
        self.weights = torch.arange(start=1, end=self.days+1, step=1)
        self.weights = self.weights / self.weights.sum()


    def forward(self, x):
        # x with shape (b, t, c)
        b, t, cin = x.shape
        newx = x.unfold(1, self.days, self.stride) # with shape (b, t_out, cin, kernel_size)

        weights = self.weights.to(newx.device)
        out = torch.mul(newx, weights).sum(dim=3)
        return out

class TS_Min(nn.Module):
    def __init__(self, days, stride):
        assert days > 0
        super(TS_Min, self).__init__()
        self.days = days
        self.stride = stride

    def forward(self, x):
        # x with shape (b, t, c)
        b, t, cin = x.shape
        newx = x.unfold(1, self.days, self.stride)
        newx = torch.min(newx, dim=3)[0]
        return newx


class TS_Max(nn.Module):
    def __init__(self, days, stride):
        assert days > 0
        super(TS_Max, self).__init__()
        self.days = days
        self.stride = stride

    def forward(self, x):
        # x with shape (b, t, c)
        b, t, cin = x.shape
        newx = x.unfold(1, self.days, self.stride)
        newx = torch.max(newx, dim=3)[0]
        return newx

class TS_Sum(nn.Module):
    def __init__(self, days, stride):
        assert days > 0
        super(TS_Sum, self).__init__()
        self.days = days
        self.stride = stride

    def forward(self, x):
        # x with shape (b, t, c)
        b, t, cin = x.shape
        newx = x.unfold(1, self.days, self.stride)
        newx = torch.sum(newx, dim=3)
        return newx


class TS_Return(nn.Module):
    def __init__(self, days, stride, epsilon=1e-6):
        ## The input is assumed to be in order (timeline : old to now )
        #NOTE: the return rate is computed the first day and the last day in the time step. 
        assert days > 1
        super(TS_Return, self).__init__()
        self.days = days
        self.stride = stride
        self.epsilon = epsilon

    def forward(self, x):
        # x with shape (b, t, c)
        newx = x.unfold(1, self.days, self.stride)
        out_return = (newx[:,:,:,-1] - newx[:,:,:,0]) / (newx[:,:,:,0] + self.epsilon )
        return out_return


class TS_Decaylinear(TS_WeightAverage):
    pass




class TS_Batchnorm1d(nn.Module):
    def __init__(self, num_features):
        super(TS_Batchnorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # x with shape (b, t, c)
        b, t, c = x.shape
        x = x.transpose(1, 2)
        return self.bn(x).transpose(1, 2)

class TS_LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, ):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        # input x with shape (b, t, c)
        # b, t, c = x.shape
        x = x.transpose(1, 2)
        return self.ln(x).transpose(1, 2)
        






if __name__ == "__main__":
    import torch
    x = torch.randn(2,10,3)
    y = x.unfold(1, 5, 1)
    
