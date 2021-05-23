import os
import math
import imageio
import datetime

from option import args
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import numpy as np
import matplotlib.pyplot as plt
import random

myrandom = random
myrandom.seed(args.rnd_seed)

def print_remained_time(timer, epoch, max_epoch):
    pasted_time_avg_per_epoch = timer.tocvalue()
    pasted_time_avg_per_epoch /= (epoch+1)
    remained_sec = (max_epoch-1-epoch) * pasted_time_avg_per_epoch
    remained_days = remained_sec // 60 // 60 // 24
    remained_hour = (remained_sec - (remained_days * 60 * 60 * 24)) // 60 // 60
    remained_min = (remained_sec - (remained_days * 60 * 60 * 24) - (remained_hour * 60 * 60)) // 60
    print('training might be ended in [%03d days %02d h %02d m] from now...'
          % (remained_days, remained_hour, remained_min))

def opt_zerograd(obj):
    for model_type in list(obj.keys()):
        obj[model_type].zero_grad()

def sch_opt_step(obj):
    for model_type in list(obj.keys()):
        obj[model_type].step()

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def hr_crop_for_pb_forward(hr, args):
    b, c, h, w = hr.size()
    hr_p = args.patch_size
    c_hook = hr_p // 4
    hr_p_h=hr_p//2
    w_num, h_num = ((w-hr_p)//hr_p_h) - 1, ((h-hr_p)//hr_p_h) - 1
    h_st = c_hook
    h_ed = h_num*hr_p_h+hr_p-c_hook
    w_st = c_hook
    w_ed = w_num*hr_p_h+hr_p-c_hook
    hr_pb = hr.new(b, c, h, w).zero_()
    hr_pb[:,:,h_st:h_ed, w_st:w_ed] = hr[:,:,h_st:h_ed, w_st:w_ed]
    return hr_pb

def gen_gauss_kernel(ker_size):
    half_ker_size = ker_size//2
    kernel = np.zeros((ker_size, ker_size))

    for d in range(1, half_ker_size+1):
        value = 1.0 / (2.0*float(d)+1.0)**2
        for i in range(-d, d+1):
            for j in range(-d, d+1):
                kernel[half_ker_size-1-i][half_ker_size-1-j] += value

    kernel = kernel / np.sum(kernel)
    return kernel

def save_tensor_to_image(args, img_tensor, name):
    normalized = img_tensor[0].data.mul(255 / 255)
    img_np = normalized.byte().permute(1, 2, 0).cpu()
    imageio.imwrite(name+'.png', img_np)

def save_model(args, model, epoch):
    for model_type in list(model.keys()):
        if (model_type == 'net_G') or (model_type == 'net_D'):
            torch.save(
                model[model_type].state_dict(),
                args.save_trained + '_' + args.today_time + '/models/' +
                args.SR_model + '_' + model_type + '_%03d' % (epoch) + '.pt'
            )

def adjust_learning_rate(args, curr_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    next_lr = args.lr * (args.lr_decay_factor ** (curr_epoch // args.lr_decay_step))
    return next_lr

def calc_psnr_dpb_forward(args, sr, hr, rgb_range=255):
    b, c, h, w = hr.size()
    hr_p = args.patch_size
    scale = args.scale
    shave = scale
    hr_p_h = hr_p//2
    c_hook = hr_p_h - scale // 2

    diff = (sr[:,:,c_hook+shave:h-c_hook-shave,c_hook+shave:w-c_hook-shave] -
            hr[:,:,c_hook+shave:h-c_hook-shave,c_hook+shave:w-c_hook-shave]).data.div(rgb_range)

    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def calc_psnr_pb_forward(args, sr, hr, scale, rgb_range=255):
    b, c, h, w = hr.size()
    hr_p = args.patch_size
    c_hook = hr_p // 4
    hr_p_h=hr_p//2
    w_num, h_num = ((w-hr_p)//hr_p_h) - 1, ((h-hr_p)//hr_p_h) - 1
    h_st = c_hook
    h_ed = h_num*hr_p_h+hr_p-c_hook
    w_st = c_hook
    w_ed = w_num*hr_p_h+hr_p-c_hook
    shave = scale

    diff = (sr[:,:,h_st+shave:h_ed-shave,w_st+shave:w_ed-shave] -
            hr[:,:,h_st+shave:h_ed-shave,w_st+shave:w_ed-shave]).data.div(rgb_range)

    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def calc_psnr(sr, hr, scale, rgb_range=255, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def tensor_prepare(l, args, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda')

    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def add_noise(x, noise='.'):
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def get_patch(lr, hr, scale=4, patch_size=192):
    lr_row, lr_col = lr.shape[:2]
    hr_patch_size = int(patch_size)
    lr_patch_size = int(patch_size/scale)

    lr_rnd_row = myrandom.randrange(0, lr_row - lr_patch_size + 1)
    lr_rnd_col = myrandom.randrange(0, lr_col - lr_patch_size + 1)

    hr_rnd_row = lr_rnd_row * scale
    hr_rnd_col = lr_rnd_col * scale

    lr = lr[lr_rnd_row:lr_rnd_row+lr_patch_size,
         lr_rnd_col:lr_rnd_col+lr_patch_size, :]
    hr = hr[hr_rnd_row:hr_rnd_row+hr_patch_size,
         hr_rnd_col:hr_rnd_col+hr_patch_size, :]

    return lr, hr


def augment(l, hflip=True, rot=True):
    hflip = hflip and myrandom.random() < 0.5
    vflip = rot and myrandom.random() < 0.5
    rot90 = rot and myrandom.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]

def get_patch_tot(phase, args, lr, hr):
    if phase == 'train':
        lr, hr = get_patch(lr, hr, args.scale, args.patch_size)
        lr, hr = augment([lr, hr])
        #lr = add_noise(lr, '.')
    else:
        ih, iw = lr.shape[0:2]
        hr = hr[0:ih * args.scale, 0:iw * args.scale]

    return lr, hr

def np_to_tensor(img_np, rgb_range):
    def _np_to_tensor(img_np):
        img_np = np.ascontiguousarray(img_np.transpose((2,0,1)))
        img_t = torch.from_numpy(img_np).float()
        img_t.mul_(rgb_range/255)

        return img_t
    return [_np_to_tensor(_img_np) for _img_np in img_np]


def make_optimizer(args, my_model):
    optim_dict = {}
    model_types = list(my_model.keys())
    for model_type in model_types:
        if (model_type == 'net_G') or (model_type == 'net_D'):
            trainable = filter(lambda x: x.requires_grad,
                               my_model[model_type].parameters())
            if args.optimizer == 'SGD':
                optimizer_function = optim.SGD
                kwargs = {'momentum': args.momentum}
            elif args.optimizer == 'ADAM':
                optimizer_function = optim.Adam
                kwargs = {
                    'betas': (args.beta1, args.beta2),
                    'eps': args.epsilon
                }
            elif args.optimizer == 'RMSprop':
                optimizer_function = optim.RMSprop
                kwargs = {'eps': args.epsilon}

            kwargs['lr'] = args.lr
            kwargs['weight_decay'] = args.weight_decay

            optim_dict[model_type] = optimizer_function(trainable, **kwargs)

    return optim_dict

def make_scheduler(args, my_optimizer):
    if args.lr_decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_factor
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.lr_decay_factor
        )

    return scheduler

def make_lrscheduler(args, SR_optimizer):
    SR_lrscheduler = {}
    for model_type in list(SR_optimizer.keys()):
        if (model_type == 'net_G'):
            SR_lrscheduler[model_type] = lrs.StepLR(
                SR_optimizer[model_type],
                step_size=args.lr_decay_step,  # decay epoch
                gamma=args.lr_decay_factor
            )
        elif (model_type == 'net_D'):
            SR_lrscheduler[model_type] = lrs.StepLR(
                SR_optimizer[model_type],
                step_size=args.lr_decay_step,  # decay epoch
                gamma=args.lr_decay_factor
            )
    return SR_lrscheduler


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format(filename, p), ndarr)
