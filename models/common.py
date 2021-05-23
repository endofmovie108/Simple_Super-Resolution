import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv_1x1_9(in_channels, out_channels, kernel_size=1, bias=True):
    kernel_size = 1
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class conv_1x1_9_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(conv_1x1_9_layer, self).__init__()
        self.one_div_nine = 1.0/9.0
        kernel_size=1
        # # self.conv_1x1_l1 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l2 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l3 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l4 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l5 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l6 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l7 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l8 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # # self.conv_1x1_l9 = default_conv_1x1_9(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f1 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f2 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f3 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f4 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f5 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f6 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f7 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f8 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv_1x1_f9 = F.conv2d(in_channels, out_channels, kernel_size=1)
        # initialize weights and biases\
        self.weight0 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight1 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight2 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight3 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight4 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight5 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight6 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight7 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        self.weight8 = Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))

        self.bias0 = Parameter(torch.Tensor(out_channels))
        self.bias1 = Parameter(torch.Tensor(out_channels))
        self.bias2 = Parameter(torch.Tensor(out_channels))
        self.bias3 = Parameter(torch.Tensor(out_channels))
        self.bias4 = Parameter(torch.Tensor(out_channels))
        self.bias5 = Parameter(torch.Tensor(out_channels))
        self.bias6 = Parameter(torch.Tensor(out_channels))
        self.bias7 = Parameter(torch.Tensor(out_channels))
        self.bias8 = Parameter(torch.Tensor(out_channels))

        # self.weights = [Parameter(torch.Tensor(out_channels, in_channels // 1, kernel_size, kernel_size))
        #                 for _ in range(9)]
        # self.biases = [Parameter(torch.Tensor(out_channels))
        #                for _ in range(9)]
        # self.reset_parameters(self.weights, self.biases)

        self.reset_parameter(self.weight0, self.bias0)
        self.reset_parameter(self.weight1, self.bias1)
        self.reset_parameter(self.weight2, self.bias2)
        self.reset_parameter(self.weight3, self.bias3)
        self.reset_parameter(self.weight4, self.bias4)
        self.reset_parameter(self.weight5, self.bias5)
        self.reset_parameter(self.weight6, self.bias6)
        self.reset_parameter(self.weight7, self.bias7)
        self.reset_parameter(self.weight8, self.bias8)

    def reset_parameter(self, weight, bias) -> None:
        def _init_kaiming_uniform_(w, b):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            if b is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(b, -bound, bound)

        _init_kaiming_uniform_(weight, bias)

    def reset_parameters(self, weights, biases) -> None:
        def _init_kaiming_uniform_(w, b):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            if b is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(b, -bound, bound)

        for idx in range(len(self.weights)):
            _init_kaiming_uniform_(self.weights[idx], self.biases[idx])

    def forward(self, x):
        y = F.conv2d(x, self.weight0, self.bias0, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight2, self.bias2, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight4, self.bias4, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight5, self.bias5, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight6, self.bias6, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight7, self.bias7, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        y += F.conv2d(x, self.weight8, self.bias8, stride=1, padding=0, dilation=1, groups=1) * self.one_div_nine
        return y
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
