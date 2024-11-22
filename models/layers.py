import torch
import torch.nn as nn


def receptive_field(op_params):
    _, _, erfield, estride = op_params[0]
    for i in range(1, len(op_params)):
        _, _, kernel, stride = op_params[i]
        one_side = erfield // 2
        erfield = (kernel - 1) * estride + 1 + 2 * one_side
        estride = estride * stride
        if erfield % 2 == 0:
            print("EVEN", erfield)
        print(erfield, estride)
    return erfield, estride


class ResidualEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 activation=torch.nn.ELU(), dropout=0.1, last=False):
        super(ResidualEncoder, self).__init__()
        self.last = last

        self.conv_op = torch.nn.Conv1d(
            in_channels=in_channels, # 1
            out_channels=2 * out_channels, # 200
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1, groups=1, bias=True
        )

        self.nin_op = torch.nn.Conv1d(
            in_channels=2 * out_channels, # 200
            out_channels=out_channels, # 100
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )
        
        self.res_op = torch.nn.Conv1d(
            in_channels=2 * out_channels, # 200
            out_channels=out_channels, # 100
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)


    def forward(self, x):
        z_ = self.bn(self.conv_op(x))
        # print('after conv1: ', z_.shape)
        z = self.dropout(self.activation(z_))
        y_ = self.nin_op(z)
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_


class ResidualDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 activation=torch.nn.ELU(), dropout=0.5, last=False):
        ''' 
        in_channels: 100
        out_channels: 1
        kernel_size: 2049
        stride: 2048
        '''
        super(ResidualDecoder, self).__init__()
        self.last = last
        self.conv_op = torch.nn.ConvTranspose1d(
            in_channels=in_channels, # 100
            out_channels=out_channels * 2, # 1*2
            kernel_size=kernel_size, # 2049
            stride=stride, # 2048
            padding=padding, 
            dilation=1, groups=1, bias=True
        )
        self.nonlin = torch.nn.Conv1d(
            in_channels=out_channels * 2, # 1*2
            out_channels=out_channels, # 1
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )
        self.res_op = torch.nn.Conv1d(
            in_channels=out_channels * 2, # 
            out_channels=out_channels, # 
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)


    def forward(self, x):
        
        z_ = self.bn(self.conv_op(x))
        # print('after convTranspose1: ', z_.shape)
        z = self.dropout(self.activation(z_))
        y_ = self.nonlin(z)
        # print(y_.size(), z.size())
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_