import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.vgg_D as vgg_D
import models.vgg_F as vgg_F
from models import common

def make_model(args, parent=False):
    models = {}
    net_G = RRDBNet(args, in_nc=3, out_nc=3,
                    nf=args.n_feats, nb=args.n_rir, gc=args.n_feats2)
    net_D = vgg_D.Discriminator_VGG_128(in_nc=3, nf=64)
    net_F = vgg_F.VGGFeatureExtractor(args)
    models['net_G'] = net_G
    models['net_D'] = net_D
    models['net_F'] = net_F

    # if args.load_trained != '.':
    #     models['net_G'].load_state_dict(
    #         torch.load(args.load_trained),
    #         strict=False
    #     )

    return models

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def make_layer_PRV(block, block_1x1, n_layers):
    layers = []
    # 3x3 conv layers, secure receptive field modules
    for _ in range(8):
        layers.append(block())
    # 1x1 conv layers, secure depth modules
    for _ in range(8, n_layers):
        layers.append(block_1x1())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C_1x1(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_1x1, self).__init__()
        conv = nn.Conv2d
        conv_1x1 = common.conv_1x1_9_layer
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_1x1(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = conv_1x1(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = conv_1x1(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = conv_1x1(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = conv_1x1(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDB_1x1(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_1x1(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_1x1(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_1x1(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, args, in_nc, out_nc, nf, nb, gc=32): # nb:23
        super(RRDBNet, self).__init__()
        RRDB_block_f1 = functools.partial(RRDB, nf=nf, gc=gc)
        RRDB_block_f2 = functools.partial(RRDB_1x1, nf=nf, gc=gc)
        self.args = args
        self.ref_model = args.RRDB_ref

        self.sub_mean = common.MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer_PRV(RRDB_block_f1, RRDB_block_f2, nb)
        self.trunk_conv = common.default_conv_1x1_9(nf, nf)
        #self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.add_mean = common.MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def ori_forward(self, x):
        if self.ref_model is False:
            x = self.sub_mean(x)
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        if self.ref_model is False:
            out = self.add_mean(out)
        return out

    def forward_monkey(*inputs):
        x = inputs[0].ori_forward(inputs[1])
        return x

    def forward(self, x):
        if not self.args.forward_ver == 1:
            x=self.forward_monkey(self, x)
        else:
            x=self.forward_monkey(x)
        return x
