#!/bin/bash/
# For release
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 16 --n_feats 64
python ../main.py --SR_model RCAN_L1 --n_resgroups 1 --n_resblocks 20 --n_feats 185
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 15 --n_feats 65
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 13 --n_feats 69
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 11 --n_feats 73
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 10 --n_feats 76
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 8 --n_feats 81
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 7 --n_feats 85
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 5 --n_feats 93
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 3 --n_feats 105
python ../main.py --SR_model EDSR_baseline_L1 --n_resblocks 1 --n_feats 123

