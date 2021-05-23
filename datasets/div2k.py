import os
import utils
import imageio
import numpy as np
from torch.utils.data import Dataset

class SR_Dataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase

        if phase == 'train':
            self.HR_path = os.path.join(args.dir_data, 'DIV2K', 'Non-cropped', 'DIV2K_train_HR')
            self.LR_path = os.path.join(args.dir_data, 'DIV2K', 'Non-cropped', 'DIV2K_train_LR_bicubic', 'X%d'%(args.scale))

        elif phase == 'val' or phase == 'test':
            self.HR_path = os.path.join(args.dir_data, 'DIV2K', 'Non-cropped', 'DIV2K_valid_HR')
            self.LR_path = os.path.join(args.dir_data, 'DIV2K', 'Non-cropped', 'DIV2K_valid_LR_bicubic', 'X%d'%(args.scale))

        HR_list = os.listdir(self.HR_path)
        LR_list = os.listdir(self.LR_path)

        HR_list = [file for file in HR_list if file.endswith(".png")]
        LR_list = [file for file in LR_list if file.endswith(".png")]

        HR_list.sort()
        LR_list.sort()

        if phase == 'train':
            self.HR_list, self.LR_list = HR_list[:args.n_train], \
                                         LR_list[:args.n_train]
            # npy file check
            for idx in range(len(self.HR_list)):
                HR_png_name = self.HR_list[idx]
                HR_npy_name = HR_png_name.replace('.png', '.npy')
                if not os.path.isfile(os.path.join(self.HR_path, HR_npy_name)):
                    HR = imageio.imread(os.path.join(self.HR_path, HR_png_name))
                    np.save(os.path.join(self.HR_path, HR_npy_name), HR)
                    self.HR_list[idx] = HR_npy_name
                else:
                    self.HR_list[idx] = HR_npy_name

            for idx in range(len(self.LR_list)):
                LR_png_name = self.LR_list[idx]
                LR_npy_name = LR_png_name.replace('.png', '.npy')
                if not os.path.isfile(os.path.join(self.LR_path, LR_npy_name)):
                    LR = imageio.imread(os.path.join(self.LR_path, LR_png_name))
                    np.save(os.path.join(self.LR_path, LR_npy_name), LR)
                    self.LR_list[idx] = LR_npy_name
                else:
                    self.LR_list[idx] = LR_npy_name

            # rcan 16000 : 1 epoch
            # 800*1000/50 = 16000
            # self.args.iter_per_epoch = (len(self.HR_list) * self.args.val_iter_step
            #                             // self.args.patch_per_img)
        elif phase == 'val':
            self.HR_list, self.LR_list = HR_list[:args.n_val], \
                                         LR_list[:args.n_val]
        elif phase == 'test':
            # if hasattr(args, 'test_start_end'):
            #     st = args.test_start_end[0][0]-1
            #     ed = args.test_start_end[0][1]
            #     self.HR_list, self.LR_list = HR_list[st:ed], LR_list[st:ed]
            # else:
            self.HR_list, self.LR_list = HR_list, LR_list

    def __len__(self):
        if self.phase == 'train':
            return (len(self.HR_list)) * self.args.repeat
            # 16000
        elif (self.phase == 'val') or (self.phase == 'test'):
            return (len(self.HR_list))

    def _get_index(self, idx):
        if self.phase == 'train':
            return idx % len(self.HR_list)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_name = self.LR_list[idx]
        hr_name = self.HR_list[idx]
        filename = hr_name
        if lr_name.find('.npy') > 0:
            lr = np.load(os.path.join(self.LR_path, lr_name))
        else:
            lr = imageio.imread(os.path.join(self.LR_path, lr_name))

        if hr_name.find('.npy') > 0:
            hr = np.load(os.path.join(self.HR_path, hr_name))
        else:
            hr = imageio.imread(os.path.join(self.HR_path, hr_name))

        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        return lr, hr, filename

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = utils.get_patch_tot(self.phase, self.args, lr, hr)
        lr, hr = utils.set_channel([lr, hr], n_channel=self.args.n_colors)
        lr_t, hr_t = utils.np_to_tensor([lr, hr], rgb_range=self.args.rgb_range)
        return lr_t, hr_t, filename
