def set_template(args):
    # EDSR original training
    if args.act_ver == 1:
        act = 'ReLU'
    elif args.act_ver == 2:
        act = 'LReLU'
    elif args.act_ver == 3:
        act = 'PReLU'


    if args.template == 'EDSR_train':
        args.SR_model = 'EDSR'
        args.loss = 'L1'
        #args.pre_train = '../pretrained_models/EDSR/EDSR_BIX4.pt'

        # architecture
        # args.n_resblocks = 32
        # args.n_feats = 256

        args.save_trained = '../../trained_results/EDSR/EDSR_RB%02d_F%03d_%s' % \
                            (args.n_resblocks, args.n_feats, act)

        # training options
        args.vl_dset_name = 'Set14'
        args.num_epoch = 30
        args.lr = 1e-4
        args.lr_decay_step = 10
        args.lr_decay_factor = 0.5

    # RCAN original training
    elif args.template == 'RCAN_train':
        args.SR_model = 'RCAN'
        args.loss = 'L1'
        #args.pre_train = '../pretrained_models/RCAN/RCAN_BIX4.pt'

        # architecture
        # args.n_resgroups = 10
        # args.n_resblocks = 20
        # args.n_feats = 64

        args.save_trained = '../../trained_results/RCAN/RCAN_RG%02d_RB%02d_F%03d_%s' % \
                            (args.n_resgroups, args.n_resblocks, args.n_feats, act)

        # training options
        args.vl_dset_name = 'Set14'
        args.num_epoch = 300
        args.lr = 1e-4
        args.lr_decay_step = 100
        args.lr_decay_factor = 0.5

    # RRDB original training
    elif args.template == 'RRDB_train':
        args.SR_model = 'RRDB'
        args.RRDB_ref = True
        args.loss = 'L1'
        #args.pre_train = '../pretrained_models/RRDB/RRDB_BIX4.pt'

        # architecture
        # args.n_rir = 23
        # args.n_feats = 64
        # args.n_feats2 = 32

        args.save_trained = '../../trained_results/RRDB/RRDB_RIR%02d_RDB%02d_F%03d_FF%03d_%s' % \
                            (args.n_rir, 3, args.n_feats, args.n_feats2, act)

        # training options
        args.vl_dset_name = 'Set14'
        args.num_epoch = 300
        args.lr = 1e-4
        args.lr_decay_step = 100
        args.lr_decay_factor = 0.5

    # SAN original training
    elif args.template == 'SAN_train':
        args.SR_model = 'SAN'
        args.loss = 'L1'
        #args.pre_train = '../pretrained_models/SAN/SAN_BIX4.pt'

        # architecture
        # args.n_resgroups = 10
        # args.n_resblocks = 20
        # args.n_feats = 64

        args.save_trained = '../../trained_results/SAN/SAN_RG%02d_RB%02d_F%03d_%s' % \
                            (args.n_resgroups, args.n_resblocks, args.n_feats, act)

        # training options
        args.vl_dset_name = 'Set14'
        args.num_epoch = 300
        args.lr = 1e-4
        args.lr_decay_step = 100
        args.lr_decay_factor = 0.5

    #
    #
    # if args.SR_model == 'RCAN':
    #
    #     args.num_epoch = 200
    #     args.pre_train = '../pretrained_models/RCAN/RCAN_BIX4'
    #     args.load_trained = '../trained_results/RCAN_x4'
    #     args.save_trained = '../trained_results/RCAN_x4'
    #
    # if args.SR_model == 'RCAN_L1':
    #     # DIV2K
    #     args.rgb_range = 255
    #     args.rgb_mean = (0.4488, 0.4371, 0.4040)
    #     args.rgb_std = (1.0, 1.0, 1.0)
    #     args.n_colors = 3
    #
    #     # train commands
    #     args.phase = 'train'
    #     args.num_epoch = 30
    #     args.lr = 1e-4
    #     # 1e-4 (RCAN default)
    #     args.lr_decay_step = 10 # decay epoch
    #     # 200 (lr_decay_step)
    #     args.loss_function = 'L1' # gaussian similarity
    #     args.gae_gamma = 0.49
    #     args.sgae_gamma = 1.25
    #     # 'sGAE', 'GAE', 'L1' (RCAN default),'MSE'
    #     # args.pre_train = '../pretrained_models/RCAN/RCAN_BIX4'
    #     # args.load_trained = '../pretrained_models/RCAN/RCAN_BIX4'
    #
    #     # train commands
    #     # args.save_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/trained_results_depth/RCAN/RCAN_L1_x4_RG%d_RB%d_F%d' % \
    #     #                     (args.n_resgroups, args.n_resblocks, args.n_feats)
    #
    #     # test commands
    #     args.phase = 'test'
    #     # args.cpu = True
    #     args.te_dset_name = 'DIV2K'
    #     args.load_trained = 'D:\\JY\\00_Research\\JYSR\\00_Trained_SR_models\\200805_RCAN_RG1_F185\\model_latest'
    #     args.save_test = 'D:\\JY\\00_Research\\JYSR\\1_JYSR\\200805_JYSR_codes\\test_results\\200805_RCAN_RG1_F185_model_latest'
    #     args.n_resgroups = 1
    #     args.n_feats = 185
    #
    # if args.SR_model == 'RRDB_L1':
    #     # DIV2K
    #     args.rgb_range = 255
    #     args.rgb_mean = (0.4488, 0.4371, 0.4040)
    #     args.rgb_std = (1.0, 1.0, 1.0)
    #     args.n_colors = 3
    #
    #     # train commands
    #     args.phase = 'train'
    #     #args.patch_size = 128 # lr=32
    #     args.num_epoch = 30
    #     args.lr = 1e-4
    #     # 1e-4 (RCAN default)
    #     args.lr_decay_step = 10  # decay epoch
    #     # 200 (lr_decay_step)
    #     args.loss_function = 'L1'  # gaussian similarity
    #     args.save_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/trained_results_depth/RRDB/RRDB_L1_x4_RIR%d_RDB%d_NF%d_GC%d' % \
    #                         (args.n_rir, 3, args.n_feats, args.n_feats2)
    #
    #     # test commands
    #     # args.phase = 'test'
    #     # args.cpu = True
    #     # args.load_trained = '../trained_results/RCAN_GAE_x4_2020-07-24_14h_20m/models/RCAN_GAE_net_G_019'
    #     # args.save_test = '../test_results/RCAN_GAE_x4_2020-07-24_14h_20m/RCAN_GAE_net_G_019'
    #
    # if args.SR_model == 'EDSR_baseline_L1':
    #     # DIV2K
    #     args.rgb_range = 255
    #     args.rgb_mean = (0.4488, 0.4371, 0.4040)
    #     args.rgb_std = (1.0, 1.0, 1.0)
    #     args.n_colors = 3
    #     args.phase = 'train'
    #     args.num_epoch = 30
    #     # 1000 (RCAN default)
    #     args.lr = 1e-4
    #     # 1e-4 (RCAN default)
    #     args.lr_decay_step = 10  # decay epoch
    #     # 200 (lr_decay_step)
    #     args.loss_function = 'L1'  # gaussian similarity
    #     # model structure
    #     args.res_scale = 1.0 # baseline-> 1.0, EDSR -> 0.1
    #     #args.load_trained = '../trained_results/EDSR_x4.pt'
    #     args.save_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/trained_results_depth/EDSR_baseline/EDSR_baseline_L1_x4_RB%d_NF%d' % \
    #                         (args.n_resblocks, args.n_feats)
    #
    # if args.SR_model == 'SAN_L1':
    #     # DIV2K
    #     args.rgb_range = 255
    #     args.rgb_mean = (0.4488, 0.4371, 0.4040)
    #     args.rgb_std = (1.0, 1.0, 1.0)
    #     args.n_colors = 3
    #
    #     # train commands
    #     args.phase = 'train'
    #     args.num_epoch = 30
    #     args.lr = 1e-4
    #     # 1e-4 (RCAN default)
    #     args.lr_decay_step = 10 # decay epoch
    #     # 200 (lr_decay_step)
    #     args.loss_function = 'L1' # gaussian similarity
    #     # 'sGAE', 'GAE', 'L1' (RCAN default),'MSE'
    #     # args.pre_train = '../pretrained_models/RCAN/RCAN_BIX4'
    #     # args.load_trained = '../pretrained_models/RCAN/RCAN_BIX4'
    #
    #     # train commands
    #     args.save_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/trained_results_depth/SAN/SAN_L1_x4_RG%d_RB%d_F%d' % \
    #                         (args.n_resgroups, args.n_resblocks, args.n_feats)
    #
    #     # test commands
    #     # args.phase = 'test'
    #     # # args.cpu = True
    #     # args.te_dset_name = 'DIV2K'
    #     # args.load_trained = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/pretrained_models_JY/200804 RCAN_RG1_F185/model_best'
    #     # args.save_test = '/media/hdsp/12TB1/JY/05_pytorch-JYSR/test_results/RCAN/200804 RCAN_RG1_F185/200804 RCAN_RG1_F185_model_best'
    #     # args.n_resgroups = 1
    #     # args.n_feats = 185