import os
import glob
import cv2
import scipy.io
import argparse
import imageio
from torch.autograd import Variable
from models import edsr
from models import rcan
from models import rrdb
from models import san
import torch
import numpy as np

What_model = 'RRDB'
device = 'cuda'

# load SR model
parser = argparse.ArgumentParser()
args = parser.parse_args()




if What_model == 'EDSR':
    args.load_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/pretrained_models/EDSR/EDSR_x4.pt'
    #args.n_resgroups = 10
    args.n_resblocks = 32
    args.n_feats = 256
    #args.reduction = 16
    args.scale = 4
    args.res_scale = 0.1
    args.rgb_range = 255
    args.rgb_mean = (0.4488, 0.4371, 0.4040)
    args.rgb_std = (1.0, 1.0, 1.0)
    args.n_colors = 3
    SR_model = edsr.make_model(args)['net_G'].to(device)
    SR_model.eval()

if What_model == 'RCAN':
    args.load_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/pretrained_models/RCAN/RCAN_BIX4.pt'
    args.n_resgroups = 10
    args.n_resblocks = 20
    args.n_feats = 64
    args.reduction = 16
    args.scale = 4
    args.res_scale = 1
    args.rgb_range = 255
    args.rgb_mean = (0.4488, 0.4371, 0.4040)
    args.rgb_std = (1.0, 1.0, 1.0)
    args.n_colors = 3
    SR_model = rcan.make_model(args)['net_G'].to(device)
    SR_model.eval()

if What_model == 'RRDB':
    args.load_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/pretrained_models/RRDB/RRDB_PSNR_x4.pth'
    args.n_rir = 23
    args.RRDB_ref_model = True
    args.cpu = False
    args.n_resblocks = 20
    args.n_feats = 64
    args.n_feats2 = 32
    args.scale = 4
    args.rgb_range = 255
    args.rgb_mean = (0.4488, 0.4371, 0.4040)
    args.rgb_std = (1.0, 1.0, 1.0)
    args.n_colors = 3
    SR_model = rrdb.make_model(args)['net_G'].to(device)
    SR_model.eval()

if What_model == 'SAN':
    args.load_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/pretrained_models/SAN/SAN_BI4X.pt'
    args.n_resgroups = 10
    args.n_resblocks = 20
    args.n_feats = 64
    args.reduction = 16
    args.scale = 4
    #args.n_feats = 64
    args.res_scale = 1
    args.rgb_range = 255
    args.rgb_mean = (0.4488, 0.4371, 0.4040)
    args.rgb_std = (1.0, 1.0, 1.0)
    args.n_colors = 3
    SR_model = san.make_model(args)['net_G'].to(device)
    SR_model.eval()

# load images

i_dir = '/media/hdsp/12TB1/JY/00_Datasets/DIV2K/Crop1600/DIV2K_crop1600/LR_bicubic/*'

if What_model != 'RRDB':
    for i_name in sorted(glob.glob(i_dir)):
        print(i_name)
        only_name = os.path.splitext(os.path.split(i_name)[-1])[0]
        img = imageio.imread(i_name)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(np_transpose).float()
        img.mul_(args.rgb_range / 255)
        #img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = Variable(img_LR.to(device), requires_grad = True)
        Wmask = torch.FloatTensor(img_LR.shape).zero_()
        Wmask = Wmask.to(device)

        img_SR = SR_model(img_LR)
        SR_size = img_SR.shape
        for dim_idx in range(SR_size[1]):
            img_SR[0, dim_idx, int(SR_size[2]/2), int(SR_size[3]/2)].backward(retain_graph=True)

            Wmask += img_LR.grad * img_LR.grad

            SR_model.zero_grad()
            img_LR.grad.zero_()

        Wmask /= SR_size[1]
        Wmask = Wmask.cpu().detach().numpy()
        dat = {'Wmask': Wmask}
        scipy.io.savemat('../../test_results/effective_receptive_field/'+What_model+'/'+only_name+'_Wmask.mat',
                        dat)

        # quatize
        pixel_range = 255 / args.rgb_range
        img_SR.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

        # save tensor to image
        img_SR = img_SR[0].data.mul(args.rgb_range / 255)
        img_SR = img_SR.byte().permute(1, 2, 0).cpu()
        imageio.imwrite('../../test_results/effective_receptive_field/'+What_model+'/'+only_name+'.png',
                        img_SR)

else:
    for i_name in sorted(glob.glob(i_dir)):
        print(i_name)
        only_name = os.path.splitext(os.path.split(i_name)[-1])[0]


        # RRDB reference version
        img = cv2.imread(i_name, cv2.IMREAD_COLOR)
        img = img * 1.0/255
        img = torch.from_numpy(np.transpose(img[:,:,[2,1,0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = Variable(img_LR.to(device), requires_grad=True)


        # img = imageio.imread(i_name)
        # np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        # img = torch.from_numpy(np_transpose).float()
        # img.mul_(args.rgb_range / 255)
        # # img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # img_LR = img.unsqueeze(0)
        # img_LR = Variable(img_LR.to(device), requires_grad=True)
        Wmask = torch.FloatTensor(img_LR.shape).zero_()
        Wmask = Wmask.to(device)

        img_SR = SR_model(img_LR)
        SR_size = img_SR.shape
        for dim_idx in range(SR_size[1]):
            img_SR[0, dim_idx, int(SR_size[2] / 2), int(SR_size[3] / 2)].backward(retain_graph=True)

            Wmask += img_LR.grad * img_LR.grad

            SR_model.zero_grad()
            img_LR.grad.zero_()

        Wmask /= SR_size[1]
        Wmask = Wmask.cpu().detach().numpy()
        dat = {'Wmask': Wmask}
        scipy.io.savemat('../../test_results/effective_receptive_field/' + What_model + '/' + only_name + '_Wmask.mat',
                         dat)


        img_SR = img_SR.data.squeeze().float().cpu().clamp_(0,1).numpy()
        img_SR = np.transpose(img_SR[[2,1,0], :, :], (1, 2, 0))
        img_SR = (img_SR*255.0).round()
        cv2.imwrite('../../test_results/effective_receptive_field/' + What_model + '/' + only_name + '.png',img_SR)

        # # quatize
        # pixel_range = 255 / args.rgb_range
        # img_SR.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
        #
        # # save tensor to image
        # img_SR = img_SR[0].data.mul(args.rgb_range / 255)
        # img_SR = img_SR.byte().permute(1, 2, 0).cpu()
        # imageio.imwrite('../../test_results/effective_receptive_field/' + What_model + '/' + only_name + '.png',
        #                 img_SR)



