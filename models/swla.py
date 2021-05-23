from models import common
import os
import torch
import torch.nn as nn

def make_model(args, parent=False):
    models = {}
    net_G = SWLA(args)
    models['net_G'] = net_G

    return models

## Local Attention Block
class LAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias):
        super(LAB, self).__init__()
        self.gap3d = nn.AdaptiveAvgPool3d(1)
        self.LA = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.Sigmoid()
        )
        self.conv_relu = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.conv_relu(x)
        y2 = self.conv_relu(torch.rot90(x, 1, [2, 3]))
        y3 = self.conv_relu(torch.rot90(x, 2, [2, 3]))
        y4 = self.conv_relu(torch.rot90(x, 3, [2, 3]))

        w = torch.cat((self.gap3d(y1),
                       self.gap3d(y2),
                       self.gap3d(y3),
                       self.gap3d(y4)))
        w = self.LA(w)

        y1 = self.conv_relu(w[0]*y1+w[1]*y2+w[2]*y3+w[3]*y4)
        return x + y1


## Residual Local Attention Block
class RLAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RLAB, self).__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            LAB(conv, n_feat, kernel_size, bias)
        ]
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        return (x + res).mul(self.res_scale)


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RLAB(
                conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Shallow and Wide Local Attention network
class SWLA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SWLA, self).__init__()
        self.args = args
        self.scale = args.scale
        self.lr_p_size = args.patch_size // self.scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, self.args.scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def _forward_patch(self, x):
        sr = torch.zeros(1,
                         self.args.n_colors,
                         self.lr_r_size*self.scale,
                         self.lr_c_size*self.scale).to(self.device)
        lr_crop = torch.zeros(self.args.batch_size,
                              self.args.n_colors,
                              self.lr_p_size,
                              self.lr_p_size).to(self.device)
        b_cnt = 0
        lr_rc_bag = []
        for r_idx in range(0, self.lr_r_size, self.lr_r_size//2):
            for c_idx in range(0, self.lr_c_size, self.lr_c_size//2):
                lr_rc_bag.append((r_idx, c_idx))
                lr_crop[b_cnt, :, :, :] = \
                    x[b_cnt, :, r_idx:r_idx+self.lr_p_size, c_idx:c_idx+self.lr_p_size]
                b_cnt += 1

                if b_cnt > 16 == 0:
                    b_cnt = 0
                    # SWLA forward
                    lr_crop = self.sub_mean(lr_crop)
                    lr_crop = self.head(lr_crop)

                    res = self.body(lr_crop)
                    res += lr_crop

                    lr_crop = self.tail(res)
                    lr_crop = self.add_mean(lr_crop)

                    sr[]


        x = self.sub_mean(x)
        x = self.head(x)

        return x

    def forward(self, x):
        self.lr_r_size = x.shape[2]
        self.lr_c_size = x.shape[3]

        if (x.shape[2] is not self.lr_p_size) and (x.shape[3] is not self.lr_p_size):
            x = self._forward_patch(x)
        else:

            # SWLA forward
            x = self.sub_mean(x)
            x = self.head(x)

            res = self.body(x)
            res += x

            x = self.tail(res)
            x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))