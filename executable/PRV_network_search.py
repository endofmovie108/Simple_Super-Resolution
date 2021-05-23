import torch
import argparse
import time
from torchsummary import summary
from external_lib.pytorch_modelsize import SizeEstimator

# original versions
from models import rcan
from models import edsr
from models import rrdb
from models import san

# PRV veresions
from models import rcan_PRV
from models import edsr_PRV
from models import rrdb_PRV
from models import san_PRV

What_models = ['RCAN'] #  'RCAN', 'RRDB','SAN'EDSR_baseline'RCAN','RRDB',

def calculate_forward_time(import_model):
    # load model
    in_img = torch.randn((1, 3, 48, 48)).cuda()
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

for What_model in What_models:
    f_txt = open(What_model+'_PRV.txt', 'wt')

    # load SR model
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cpu = False
    args.rgb_range = 255
    args.rgb_mean = (0.4488, 0.4371, 0.4040)
    args.rgb_std = (1.0, 1.0, 1.0)
    args.n_colors = 3
    args.forward_ver = 1
    args.act_ver = 1

    if What_model == 'EDSR':
        # original model's params
        args.load_trained = '.'
        args.n_resblocks = 32
        args.n_feats = 256
        args.scale = 4
        args.res_scale = 0.1

        # load original model
        SR_model = edsr.make_model(args)['net_G']
        SR_model.to('cuda')
        SR_model.train()

        # count params
        original_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

        # calculate memory
        ori_total_size_MB, tot_params, ori_tot_trble_params = summary(SR_model, input_size=(3, 48, 48))
        f_txt.write('RB_num: %d\tF: %d\tP: %d\tMB: %f\ttime: %f (original)\n' %
                    (args.n_resblocks, args.n_feats, ori_tot_trble_params, ori_total_size_MB, 0))

        # cuda sync & count time
        #inf_time1 = calculate_forward_time(edsr)
        del SR_model

        for rg_idx in range(60, 600):

            args.n_resblocks = rg_idx
            args.n_feats = 192

            # load PRV model
            SR_model = edsr_PRV.make_model(args)['net_G']
            SR_model.to('cuda')
            SR_model.train()

            # count params
            trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

            # calculate memory
            total_size_MB, tot_params, tot_trble_params=summary(SR_model.to('cuda'), input_size=(3,48,48))

            # cuda sync & count time
            #inf_time2 = calculate_forward_time(edsr)
            del SR_model

            if trainable_params > original_trainable_params:
                f_txt.write('RB_num: %d\tF: %d\tP: %d\tMB: %f\ttime: %f (PRV)\n'%
                            (max_rg_idx, args.n_feats, max_tprams, max_MB, max_inf_time))
                break
            max_rg_idx = rg_idx
            max_tprams = trainable_params
            max_MB = total_size_MB
            max_Pv2 = tot_params
            max_TPv2 = tot_trble_params
            max_inf_time = 0
            print('%s %d\tparams: %d ori_params: %d\tmem: %d, ori_mem: %d' %
                  (What_model, rg_idx, tot_trble_params, ori_tot_trble_params, total_size_MB, ori_total_size_MB))

    if What_model == 'RCAN':
        args.load_trained = '.'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.reduction = 16
        args.scale = 4
        args.res_scale = 1

        # load original model
        SR_model = rcan.make_model(args)['net_G']
        SR_model.to('cuda')
        SR_model.train()

        # count params
        original_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)
        # calculate memory
        ori_total_size_MB, tot_params, ori_tot_trble_params = summary(SR_model, input_size=(3, 48, 48))

        # cuda sync & count time
        #inf_time1 = calculate_forward_time(rcan)
        del SR_model

        for rg_idx in range(3, 300):
            args.n_resgroups = rg_idx
            args.n_feats = 64

            # load PRV model
            SR_model = rcan_PRV.make_model(args)['net_G']
            SR_model.to('cuda')
            SR_model.train()

            # count params
            trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

            # calculate memory
            total_size_MB, tot_params, tot_trble_params = summary(SR_model.to('cuda'), input_size=(3, 48, 48))

            # cuda sync & count time
            #inf_time2 = calculate_forward_time(rcan)
            del SR_model

            if trainable_params > original_trainable_params:
                f_txt.write('RG: %d\tF: %d\tP: %d\tMB: %f\ttime: %f\n'%
                            (max_rg_idx, args.n_feats, max_tprams, max_MB, max_inf_time))
                break
            max_rg_idx = rg_idx
            max_tprams = trainable_params
            max_MB = total_size_MB
            max_Pv2 = tot_params
            max_TPv2 = tot_trble_params
            max_inf_time = 0
            print('%s %d\tparams: %d ori_params: %d\tmem: %d, ori_mem: %d' %
                  (What_model, rg_idx, trainable_params, original_trainable_params, total_size_MB, ori_total_size_MB))



    if What_model == 'RRDB':
        args.load_trained = '.'
        args.n_rir = 23
        args.n_feats = 64
        args.n_feats2 = 32

        args.scale = 4
        args.res_scale = 1
        args.rgb_range = 255
        args.rgb_mean = (0.4488, 0.4371, 0.4040)
        args.rgb_std = (1.0, 1.0, 1.0)
        args.n_colors = 3

        # load model
        SR_model = rrdb.make_model(args)['net_G']
        SR_model.to('cuda')
        SR_model.train()

        # count params
        original_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

        # cuda sync & count time
        #inf_time1 = calculate_forward_time(rrdb)
        del SR_model

        # nfeats, nfeats2??
        for rir_idx in range(args.n_rir, 0, -1):
            for ft_idx in range(64, 2000):
                ratio = ft_idx/64
                ft2_idx = int(32 * ratio)

                args.n_rir = rir_idx
                args.n_feats = ft_idx
                args.n_feats2 = ft2_idx

                # load model
                SR_model = rrdb.make_model(args)['net_G']
                SR_model.to('cuda')
                SR_model.train()

                # count params
                trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

                # calculate memory
                total_size_MB, tot_params, tot_trble_params = summary(SR_model.to('cuda'), input_size=(3, 48, 48))

                # cuda sync & count time
                #inf_time2 = calculate_forward_time(rrdb)
                del SR_model

                if trainable_params > original_trainable_params:
                    f_txt.write('Rir: %d\tF: %d\tF2: %d\tP: %d\tMB: %f\ttime: %f\n'%
                                (rir_idx, max_ft_idx, max_ft2_idx, max_tprams, max_MB, max_inf_time))
                    break
                max_tprams = trainable_params
                max_rir = rir_idx
                max_ft_idx = ft_idx
                max_ft2_idx = ft2_idx
                max_MB = total_size_MB
                max_Pv2 = tot_params
                max_TPv2 = tot_trble_params
                max_inf_time = 0



    if What_model == 'SAN':
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

        # load model
        SR_model = san.make_model(args)['net_G']
        SR_model.to('cuda')
        SR_model.eval()

        # count params
        original_trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

        # cuda sync & count time
        #inf_time1 = calculate_forward_time(san)
        del SR_model

        for rg_idx in range(args.n_resgroups, 0, -1):
            for ft_idx in range(64, 2000):
                args.n_resgroups = rg_idx
                args.n_feats = ft_idx

                # load model
                SR_model = san.make_model(args)['net_G']
                SR_model.to('cuda')
                SR_model.eval()

                # count params
                trainable_params = sum(p.numel() for p in SR_model.parameters() if p.requires_grad)

                # calculate memory
                total_size_MB, tot_params, tot_trble_params = summary(SR_model, input_size=(3, 48, 48))

                # cuda sync & count time
                #inf_time2 = calculate_forward_time(san)
                del SR_model

                if trainable_params > original_trainable_params:
                    f_txt.write('RG: %d\tF: %d\tP: %d\tMB: %f\ttime: %f\n'%
                                (rg_idx, max_ft_idx, max_tprams,max_MB, max_inf_time))
                    break
                max_tprams = trainable_params
                max_ft_idx = ft_idx
                max_MB = total_size_MB
                max_Pv2 = tot_params
                max_TPv2 = tot_trble_params
                max_inf_time = 0

    f_txt.close()
