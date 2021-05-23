import os
import glob
import cv2
import scipy.io
import argparse
import imageio
import time
from importlib import import_module
from torch.autograd import Variable
from models import edsr
from models import rcan
from models import rrdb
from models import san
import torch
import numpy as np

What_models = ['RCAN', 'RRDB', 'SAN']#'RCAN'
device = 'cuda'

def calculate_forward_time(import_model, args, img):
    # load model
   # in_img = torch.randn(img).cuda()
    in_img = img
    with torch.no_grad():
        SR_model = import_model.make_model(args)['net_G']
        SR_model.to('cuda')
        SR_model.eval()
        torch.cuda.synchronize()
        t0 = time.time()
        out = SR_model(in_img)
        torch.cuda.synchronize()
        t1 = time.time()
        inf_time = t1 - t0
        del SR_model, out

    return inf_time

# load SR model
parser = argparse.ArgumentParser()
args = parser.parse_args()

for What_model in What_models:

    if What_model == 'EDSR':
        args.load_trained = '.'
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

        model_opts = [
            (32, 256),
            (28, 271),
            (24, 289),
            (20, 312),
            (17, 333),
            (13, 369),
            (9, 420),
            (5, 501),
            (3, 564),
            (1, 659)
        ]




    if What_model == 'RCAN':
        args.load_trained = '.'
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

        model_opts = [
            (10, 64),
            (9, 67),
            (8, 71),
            (7, 76),
            (6, 82),
            (5, 89),
            (4, 99),
            (3, 114),
            (2, 137),
            (1, 185)
        ]


    if What_model == 'RRDB':
        args.load_trained = '.'
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

        model_opts = [
            (23, 64, 32),
            (21, 67, 33),
            (19, 70, 35),
            (17, 74, 37),
            (15, 79, 39),
            (13, 85, 42),
            (11, 92, 46),
            (8, 107, 53),
            (5, 135, 67),
            (1, 281, 140)
        ]


    if What_model == 'SAN':
        args.load_trained = '.'
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

        model_opts = [
            (10, 64),
            (9, 67),
            (8, 71),
            (7, 76),
            (6, 82),
            (5, 89),
            (4, 99),
            (3, 114),
            (2, 137),
            (1, 185)
        ]


    # load images

    i_dir = '/media/hdsp/12TB1/JY/00_Datasets/DIV2K/Crop1600/DIV2K_crop1600/LR_bicubic/*'
    mod_cnt = 0
    for model_opt in model_opts:
        mod_cnt += 1
        if What_model == 'EDSR':
            args.n_resblocks=model_opt[0]
            args.n_feats=model_opt[1]
            f = open('../../test_results/inference_time_res/' + What_model +
                     '/EDSR_RB%02d_F%03d_DIV2K_crop_1600.txt' % (args.n_resblocks, args.n_feats), 'wt')
        if What_model == 'RCAN' or What_model == 'SAN':
            args.n_resgroups=model_opt[0]
            args.n_feats=model_opt[1]
            f = open('../../test_results/inference_time_res/' + What_model +
                     '/'+What_model+'_RG%02d_F%03d_DIV2K_crop_1600.txt' % (args.n_resgroups, args.n_feats), 'wt')
        if What_model == 'RRDB':
            args.n_rir=model_opt[0]
            args.n_feats=model_opt[1]
            args.n_feats2=model_opt[2]
            f = open('../../test_results/inference_time_res/' + What_model +
                     '/'+What_model+'_RiR%02d_F%03d_F2%03d_DIV2K_crop_1600.txt' % (args.n_rir, args.n_feats, args.n_feats2), 'wt')



        idx = 0
        pasted_time_avg = 0.0
        if What_model != 'RRDB':
            for i_name in sorted(glob.glob(i_dir)):
                idx += 1

                only_name = os.path.splitext(os.path.split(i_name)[-1])[0]
                img = imageio.imread(i_name)
                np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
                img = torch.from_numpy(np_transpose).float()
                img.mul_(args.rgb_range / 255)
                #img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                img_LR = img.unsqueeze(0)
                img_LR = Variable(img_LR.to(device), requires_grad = True)
                if idx == 1:
                    pasted_time = calculate_forward_time(import_module('models.' + What_model.lower()), args, img_LR)
                    continue
                pasted_time = calculate_forward_time(import_module('models.'+What_model.lower()), args, img_LR)
                pasted_time_avg+= pasted_time
                print('%s, %d, %d %f sec' % (What_model, mod_cnt, idx, pasted_time))
                f.write('%d\t%f\tsec\n' % (idx, pasted_time))


        else:
            for i_name in sorted(glob.glob(i_dir)):
                idx += 1
                print(i_name)
                only_name = os.path.splitext(os.path.split(i_name)[-1])[0]
                # RRDB reference version
                img = cv2.imread(i_name, cv2.IMREAD_COLOR)
                img = img * 1.0/255
                img = torch.from_numpy(np.transpose(img[:,:,[2,1,0]], (2, 0, 1))).float()
                img_LR = img.unsqueeze(0)
                img_LR = Variable(img_LR.to(device), requires_grad=True)
                if idx == 1:
                    pasted_time = calculate_forward_time(import_module('models.' + What_model.lower()), args, img_LR)
                    continue

                pasted_time = calculate_forward_time(import_module('models.'+What_model.lower()), args, img_LR)
                pasted_time_avg += pasted_time
                print('%s, %d, %d, %f sec' % (What_model, mod_cnt, idx, pasted_time))
                f.write('%d\t%f\tsec\n' % (idx, pasted_time))

        f.write('pasted time average: %f\n'%(pasted_time_avg/(idx-1)))
        f.close()


