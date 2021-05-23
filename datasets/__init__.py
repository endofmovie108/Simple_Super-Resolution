from importlib import import_module
from torch.utils.data import DataLoader

class Set_Dataset():
    def __init__(self, args):
        type_dset_dict = {}
        for type_dset in ['tr_dset_name', 'vl_dset_name', 'te_dset_name']:
            dset_name = getattr(args, type_dset)
            type_dset_dict[type_dset] = dset_name.lower()

        tr_dset_module = import_module('datasets.' +
                                       type_dset_dict['tr_dset_name'])
        vl_dset_module = import_module('datasets.' +
                                       type_dset_dict['vl_dset_name'])
        te_dset_module = import_module('datasets.' +
                                       type_dset_dict['te_dset_name'])
        tr_dataset = getattr(tr_dset_module, 'SR_Dataset')(args, phase='train')
        vl_dataset = getattr(vl_dset_module, 'SR_Dataset')(args, phase='val')
        te_dataset = getattr(te_dset_module, 'SR_Dataset')(args, phase='test')
        self.tr_dataloader = DataLoader(tr_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)
        self.vl_dataloader = DataLoader(vl_dataset,
                                        batch_size=1,
                                        shuffle=False)
        self.te_dataloader = DataLoader(te_dataset,
                                        batch_size=1,
                                        shuffle=False)