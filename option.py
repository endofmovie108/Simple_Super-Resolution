import argparse
import option_templates

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='JYSR')

# Use template?
parser.add_argument('--template', type=str, default='.')

# Train? Test?
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=192)

# Model options
parser.add_argument('--SR_model', type=str,
                    default='RCAN') # EDSR, RCAN, RRDB, SAN...
parser.add_argument('--shift_mean', default=True)

# act ver1: ReLU
# act ver2: Leaky-ReLU
# act ver3: Parametric-ReLU
parser.add_argument('--act_ver', type=int,
                    default=1)


# EDSR & RCAN & SAN
parser.add_argument('--n_resgroups', type=int,
                    default=10)
parser.add_argument('--n_resblocks', type=int,
                    default=20)
parser.add_argument('--n_feats', type=int,
                    default=64)
parser.add_argument('--res_scale', type=float,
                    default=1) # EDSR: 0.1, RCAN & SAN: 1.0
parser.add_argument('--reduction', type=int,
                    default=16) # RCAN option

# RRDB
parser.add_argument('--RRDB_ref', type=str2bool,
                    default=False) # add mean, sub mean problem
parser.add_argument('--n_rir', type=int,
                    default=23)
parser.add_argument('--n_feats2', type=int,
                    default=32)

# Training options
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--save_trained', type=str, default='.')
parser.add_argument('--pre_train', type=str, default='.')
parser.add_argument('--optimizer', type=str, default='ADAM')
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--val_iter_step', type=int, default=1000)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay_type', type=str, default='step')
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step', type=int, default=200)
parser.add_argument('--ad_beta1', type=float, default=0.9)
parser.add_argument('--ad_beta2', type=float, default=0.999)
parser.add_argument('--ad_epsilon', type=float, default=1e-8)
parser.add_argument('--save_test', type=str,
                    default='../test_results')

# CPU / GPU
parser.add_argument('--cpu', type=str2bool, default=False)
parser.add_argument('--precision', type=str, default='single')
parser.add_argument('--rnd_seed', type=int, default=1)

# Dataset options
parser.add_argument('--dir_data', type=str,
                    default='/media/hdsp/12TB1/JY/00_Datasets')
parser.add_argument('--tr_dset_name', type=str,
                    default='DIV2K')
parser.add_argument('--rgb_range', type=int,
                    default=255)
parser.add_argument('--rgb_mean', type=tuple,
                    default=(0.4488, 0.4371, 0.4040))
parser.add_argument('--rgb_std', type=tuple,
                    default=(1.0, 1.0, 1.0))
parser.add_argument('--n_colors', type=int,
                    default=3)
parser.add_argument('--vl_dset_name', type=str,
                    default='DIV2K')

# Test phase forward version lists
# 1: original forward,
# 2: patch-based forward,
# 3: div4-forward
# 4: dense patch-based forward
parser.add_argument('--forward_ver', type=int,
                    default=1)

# PSNR version lists
# 1: original PSNR (shave SR, HR image's boundary a little)
# 2: patch-based PSNR (shave SR, HR images based on patch-based forwarded area)
# 3: div4 PSNR
# 4: dense patch-based PSNR (shave SR, HR images based on dense patch-based forwarded area)
parser.add_argument('--PSNR_ver', type=int,
                    default=1)

parser.add_argument('--psnr_cond_pbf', type=str2bool,
                    default=False)
parser.add_argument('--te_dset_name', type=str,
                    default='DIV2K')
parser.add_argument('--repeat', type=int,
                    default=20)
parser.add_argument('--n_train', type=int,
                    default=800)
parser.add_argument('--n_test', type=int,
                    default=100)
parser.add_argument('--n_val', type=int,
                    default=10)
# parser.add_argument('--test_start_end', nargs=2, type=int,
#                     action='append')

args = parser.parse_args()

if args.template != '.':
    option_templates.set_template(args)