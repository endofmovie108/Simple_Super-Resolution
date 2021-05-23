#!/bin/bash/
# For release

#python ../main.py --SR_model RCAN --phase test --forward_ver 3 --PSNR_ver 1 --pre_train ../../../05_pytorch-JYSR/pretrained_models/RCAN/RCAN_BIX4.pt --te_dset_name DIV2K --n_resgroups 10 --n_resblocks 20 --n_feats 64 --save_test ../../test_results/RCAN_ORIGINAL_DIV2K_div4

#python ../main.py --SR_model RCAN --phase test --forward_ver 3 --PSNR_ver 1 --pre_train ../../pretrained_models_JY/RCAN_BIX4_G1R20F185P48_1000ep_v2/model/model_latest.pt --te_dset_name DIV2K --n_resgroups 1 --n_resblocks 20 --n_feats 185 --save_test ../../test_results/RCAN_SWSR_v2_DIV2K_latest_div4


# test set : BSDS100, Set5, Set14, Urban100, Manga109, DIV2K

# 300epoch model_list:
# RCAN_RG10_RB20_F064_ReLU_2020-08-11_01h_14m
# RCAN_RG03_RB20_F114_ReLU_2020-08-14_00h_44m
# RCAN_RG03_RB20_F114_LReLU_2020-08-15_18h_34m
python ../main.py --SR_model RCAN --phase test --forward_ver 1 --PSNR_ver 1 --pre_train ../../trained_results/RCAN/RCAN_RG03_RB20_F114_LReLU_2020-08-15_18h_34m/models/RCAN_net_G_299.pt --te_dset_name DIV2K --n_resgroups 3 --n_resblocks 20 --n_feats 114 --save_test ../../test_results/SWSR/RCAN_RG03_RB20_F114_LReLU_300ep_latest_DIV2K





