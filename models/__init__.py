import torch
import models.edsr as edsr
import models.rcan as rcan
import models.rrdb as rrdb
import models.san as san



class SR_model_selector:
    def __init__(self, args):
        self.phase = args.phase
        if args.SR_model.find('RCAN') > -1:
            self.SR_model = rcan.make_model(args)
        elif args.SR_model.find('EDSR') > -1:
            self.SR_model = edsr.make_model(args)
        elif args.SR_model.find('RRDB') > -1:
            self.SR_model = rrdb.make_model(args)
        elif args.SR_model.find('SAN') > -1:
            self.SR_model = san.make_model(args)
        else:
            print('model error!')

        # forward function monkey patch for test phase
        if args.phase == 'test':
            if args.forward_ver == 2:
                self.SR_model['net_G'].forward_monkey = pb_forward
            elif args.forward_ver == 3:
                self.SR_model['net_G'].forward_monkey = div4_forward
            elif args.forward_ver == 4:
                self.SR_model['net_G'].forward_monkey = dpb_forward

        for model_type in list(self.SR_model.keys()):
            if not args.pre_train == '.':
                self.SR_model[model_type].load_state_dict(
                    torch.load(args.pre_train),
                    strict=False
                )
            self.SR_model[model_type].to(args.device)

    def model_return(self):
        return self.SR_model

# For monkey patch of forward function
def dpb_forward(self, x):
    def _dbat_to_sr(y, bat, sc_bag, scale, sr_p):
        sr_p_h = sr_p // 2
        c_st = sr_p_h - scale // 2
        c_ed = c_st + scale
        for idx in range(bat.shape[0]):
            sp, sh, sw = bat[idx, :, :, :].unsqueeze(0), sc_bag[idx][0], sc_bag[idx][1]
            h_st = sh * scale
            w_st = sw * scale

            y[:, :, h_st + c_st:h_st + c_ed, w_st + c_st:w_st + c_ed] = \
                sp[:, :, c_st:c_ed, c_st:c_ed]

        return y

    lr_p=self.args.patch_size//self.args.scale
    scale = self.args.scale
    bsize = self.args.batch_size
    sr_p=self.args.patch_size
    b, c, h, w = x.size()
    y = x.new(b, c, h*scale, w*scale).zero_()
    w_num, h_num = w-lr_p, h-lr_p

    p_cnt = 0
    lp_bag = []
    sp_sc_bag = []
    for h_idx in range(h_num):
        for w_idx in range(w_num):
            print('DPB forward......%s [%.1f]%%' % (self.args.te_name, 100*(h_idx*w_num + w_idx)/(h_num*w_num)))
            xp = x[:, :, h_idx:h_idx + lr_p, w_idx:w_idx + lr_p]
            lp_bag.append(xp)
            sp_sc_bag.append((h_idx, w_idx))
            p_cnt += 1

            if p_cnt == bsize:
                lp_bat = torch.cat(lp_bag, dim=0)
                sp_bat = self.ori_forward(lp_bat)

                y = _dbat_to_sr(y, sp_bat, sp_sc_bag, scale, sr_p)
                sp_sc_bag = []
                lp_bag = []
                p_cnt = 0

    if p_cnt != 0:
        lp_bat = torch.cat(lp_bag, dim=0)
        sp_bat = self.ori_forward(lp_bat)

        y = _dbat_to_sr(y, sp_bat, sp_sc_bag, scale, sr_p)

    return y

def pb_forward(self, x):
    def _bat_to_sr(y, bat, sc_bag, sr_p, sr_p_h):
        c_hook = sr_p // 4
        for idx in range(bat.shape[0]):
            sp, sh, sw = bat[idx, :, :, :].unsqueeze(0), sc_bag[idx][0], sc_bag[idx][1]
            h_st = sh * sr_p_h
            h_ed = h_st + sr_p
            w_st = sw * sr_p_h
            w_ed = w_st + sr_p

            y[:, :, h_st + c_hook:h_ed - c_hook, w_st + c_hook:w_ed - c_hook] = \
                sp[:, :, c_hook:-c_hook, c_hook:-c_hook]
        return y

    lr_p=self.args.patch_size//self.args.scale
    lr_p_h=lr_p//2
    scale = self.args.scale
    bsize = self.args.batch_size
    sr_p=self.args.patch_size
    sr_p_h=sr_p//2
    b, c, h, w = x.size()
    y = x.new(b, c, h*scale, w*scale).zero_()
    w_num, h_num = (w-lr_p)//lr_p_h, (h-lr_p)//lr_p_h

    p_cnt = 0
    lp_bag = []
    sp_sc_bag = []
    for h_idx in range(h_num):
        for w_idx in range(w_num):
            h_st = h_idx*lr_p_h
            h_ed = h_st + lr_p
            w_st = w_idx*lr_p_h
            w_ed = w_st + lr_p

            xp = x[:, :, h_st:h_ed, w_st:w_ed]
            lp_bag.append(xp)
            sp_sc_bag.append((h_idx, w_idx))
            p_cnt += 1

            if p_cnt == bsize:
                lp_bat = torch.cat(lp_bag, dim=0)
                sp_bat = self.ori_forward(lp_bat)

                y = _bat_to_sr(y, sp_bat, sp_sc_bag, sr_p, sr_p_h)
                sp_sc_bag = []
                lp_bag = []
                p_cnt = 0

    if p_cnt != 0:
        lp_bat = torch.cat(lp_bag, dim=0)
        sp_bat = self.ori_forward(lp_bat)

        y = _bat_to_sr(y, sp_bat, sp_sc_bag, sr_p, sr_p_h)

    return y

def div4_forward(self, x):
    b, c, h, w = x.size()

    shave = 10

    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave

    x1 = x[:, :, 0:h_size, 0:w_size]  # top left
    x2 = x[:, :, 0:h_size, (w - w_size):w]  # top right
    x3 = x[:, :, (h - h_size):h, 0:w_size]  # bottom left
    x4 = x[:, :, (h - h_size):h, (w - w_size):w]  # bottom right

    y1 = self.ori_forward(x1)
    y2 = self.ori_forward(x2)
    y3 = self.ori_forward(x3)
    y4 = self.ori_forward(x4)

    scale = self.args.scale
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    y = x.new(b, c, h, w)
    y[:, :, 0:h_half, 0:w_half] = y1[:, :, 0:h_half, 0:w_half]
    y[:, :, 0:h_half, w_half:w] = y2[:, :, 0:h_half, (w_size - w + w_half):w_size]
    y[:, :, h_half:h, 0:w_half] = y3[:, :, (h_size - h + h_half):h_size, 0:w_half]
    y[:, :, h_half:h, w_half:w] = y4[:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return y
