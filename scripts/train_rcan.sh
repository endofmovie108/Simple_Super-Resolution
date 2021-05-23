#!/bin/bash/
# Legacy
# python ../main.py --SR_model RCAN_L1 --n_resgroups 10 --n_resblocks 20 --n_feats 64
# python ../main.py --SR_model RCAN_L1 --n_resgroups 9 --n_resblocks 20 --n_feats 67
# python ../main.py --SR_model RCAN_L1 --n_resgroups 8 --n_resblocks 20 --n_feats 71
# python ../main.py --SR_model RCAN_L1 --n_resgroups 7 --n_resblocks 20 --n_feats 76
# python ../main.py --SR_model RCAN_L1 --n_resgroups 6 --n_resblocks 20 --n_feats 82
# python ../main.py --SR_model RCAN_L1 --n_resgroups 5 --n_resblocks 20 --n_feats 89
# python ../main.py --SR_model RCAN_L1 --n_resgroups 4 --n_resblocks 20 --n_feats 99
# python ../main.py --SR_model RCAN_L1 --n_resgroups 3 --n_resblocks 20 --n_feats 114
# python ../main.py --SR_model RCAN_L1 --n_resgroups 2 --n_resblocks 20 --n_feats 137

# New
python ../main.py --template RCAN_train --n_resgroups 3 --n_resblocks 20 --n_feats 114 --act_ver 3
# python ../main.py --template RCAN_train --n_resgroups 1 --n_resblocks 20 --n_feats 185

