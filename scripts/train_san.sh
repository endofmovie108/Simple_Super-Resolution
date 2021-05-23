#!/bin/bash/
# For release
python ../main.py --SR_model SAN_L1 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 1 --n_resblocks 20 --n_feats 185 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 9 --n_resblocks 20 --n_feats 67 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 8 --n_resblocks 20 --n_feats 71 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 7 --n_resblocks 20 --n_feats 76 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 6 --n_resblocks 20 --n_feats 82 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 5 --n_resblocks 20 --n_feats 89 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 4 --n_resblocks 20 --n_feats 99 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 3 --n_resblocks 20 --n_feats 114 --vl_dset_name Set14
python ../main.py --SR_model SAN_L1 --n_resgroups 2 --n_resblocks 20 --n_feats 137 --vl_dset_name Set14


