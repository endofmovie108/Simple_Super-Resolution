import os
import copy
import utils
import torch

from datetime import datetime
from tqdm import tqdm

class phases():
    def __init__(self, args):
        self.args = args
        self.loss_val = -1.0
        self.loss_val_prev = -1.0

        self.lr_G_val = -1.0

        if args.phase == 'train':
            if not os.path.exists(args.save_trained + '_' + args.today_time + '/models'):
                os.makedirs(args.save_trained + '_' + args.today_time + '/models')
            self.f_vl_fname = args.save_trained + '_' + args.today_time + '/' + \
                              args.SR_model + '_valid_' + args.today_time + '.txt'
            self.f_vl_rec = open(self.f_vl_fname, 'wt')
            #self.f_vl_rec.write('trainable params: %f\n' % (args.trainable_params))
            self.f_vl_rec.close()

            self.f_tr_fname = args.save_trained + '_' + args.today_time + '/' + \
                              args.SR_model + '_train_' + args.today_time + '.txt'
            self.f_tr_rec = open(self.f_tr_fname, 'wt')
            #self.f_tr_rec.write('trainable params: %f\n' % (args.trainable_params))
            self.f_tr_rec.close()

        elif args.phase == 'test':
            if args.forward_ver == 1: forward_method = '[Ori_FORWARD]'
            elif args.forward_ver == 2: forward_method = '[PB_FORWARD]'
            elif args.forward_ver == 3: forward_method = '[Div4_FORWARD]'
            elif args.forward_ver == 4: forward_method = '[DPB_FORWARD]'
            else: forward_method = '[Ori_FORWARD]'

            if args.PSNR_ver == 1: psnr_method = '[Ori_PSNR]'
            elif args.PSNR_ver == 2: psnr_method = '[PB_PSNR]'
            elif args.PSNR_ver == 3: psnr_method = '[Div4_PSNR]'
            elif args.PSNR_ver == 4: psnr_method = '[DPB_PSNR]'
            else: psnr_method = '[Ori_PSNR]'

            args.save_test = args.save_test + '_' + forward_method + '_' + psnr_method

            if not os.path.exists(args.save_test + '/images'):
                os.makedirs(args.save_test + '/images')

            self.f_te_fname = args.save_test + '/%s_%s.txt' % \
                              (forward_method, psnr_method)
            self.f_te_rec = open(self.f_te_fname, 'wt')
            self.f_te_rec.close()



    def do(self, phase, epoch,
           SR_model, loss, SR_optimizer,
           tr_dataloader, vl_dataloader, te_dataloader):

        if phase == 'train':
            # set model to training mode!
            for model_type in list(SR_model.keys()):
                if (model_type == 'net_G') or (model_type == 'net_D'):
                    SR_model[model_type].train()

            loss_sum = 0.0
            valid_iter_cnt = 0
            for iter, (lr, hr, _) in enumerate(tr_dataloader):
                lr, hr = utils.tensor_prepare([lr, hr], self.args)

                # forward/backward pass
                utils.opt_zerograd(SR_optimizer)
                sr = SR_model['net_G'](lr)
                loss_val = loss.SR_loss(sr, hr)
                self.loss_val = float(loss_val)
                self.lr_G_val = SR_optimizer['net_G'].param_groups[0]["lr"]
                loss_val.backward()

                # skip parameter update when loss is exploded
                if (epoch != 0 and iter != 0) and (loss_val > self.loss_val_prev*10):
                    print('loss_val: %f\tloss_val_prev: %f\tskip this batch!' %
                          (loss_val, self.loss_val_prev))
                    continue

                # update parameters
                utils.sch_opt_step(SR_optimizer)

                # save current loss to utilize next iteration
                self.loss_val_prev = loss_val
                valid_iter_cnt += 1
                loss_sum += loss_val

                if iter % self.args.print_freq == 0:
                    tr_res_txt = 'epoch: %d\tlr: %f\t%s loss: %05.2f\titer: %d/%d\t[%s]\n' % \
                                 (epoch, self.lr_G_val, self.args.loss, loss_sum/valid_iter_cnt,
                                  iter*self.args.batch_size, len(tr_dataloader.dataset),
                                  datetime.now())

                    self.f_tr_rec = open(self.f_tr_fname, 'at')
                    self.f_tr_rec.write(tr_res_txt)
                    self.f_tr_rec.close()
                    print(tr_res_txt[:len(tr_res_txt) - 1])
                # break # debug

        elif phase == 'valid':
            # set model to test mode!
            SR_model['net_G'].eval()
            val_psnr_avg = 0.0
            val_psnr_cnt = 0

            with torch.no_grad():
                for valiter, (val_lr, val_hr, _) in enumerate(vl_dataloader):
                    val_lr, val_hr = utils.tensor_prepare([val_lr, val_hr], self.args)
                    val_sr = SR_model['net_G'](val_lr)
                    val_sr = utils.quantize(val_sr)
                    val_psnr = utils.calc_psnr(val_sr, val_hr, self.args.scale)
                    val_psnr_avg += val_psnr
                    val_psnr_cnt += 1

                val_psnr_avg /= val_psnr_cnt
                val_res_text = 'epoch: %d\tlr: %f\t%s loss: %05.2f\ttrain %s valid %s PSNR avg: %f [%s]\n' % \
                               (epoch, self.lr_G_val, self.args.loss, self.loss_val,
                                self.args.tr_dset_name, self.args.vl_dset_name, float(val_psnr_avg), datetime.now())

                self.f_vl_rec = open(self.f_vl_fname, 'at')
                self.f_vl_rec.write(val_res_text)
                self.f_vl_rec.close()
                print(val_res_text[:len(val_res_text) - 1])

        elif phase == 'test':
            SR_model['net_G'].eval()
            te_psnr_avg = 0.0
            te_psnr_cnt = 0

            with torch.no_grad():
                for te_iter, (te_lr, te_hr, te_name) in tqdm(enumerate(te_dataloader)):
                    self.args.te_name = te_name[0]
                    te_lr, te_hr = utils.tensor_prepare([te_lr, te_hr], self.args)

                    if self.args.RRDB_ref:
                        te_lr = te_lr.mul_(1.0/255.0)

                    te_sr = SR_model['net_G'](te_lr)
                    if self.args.RRDB_ref:
                        te_lr = te_lr.mul_(255.0)
                        te_sr = te_sr.mul_(255.0)
                    te_sr = utils.quantize(te_sr)

                    if self.args.PSNR_ver == 1 or self.args.PSNR_ver == 3:
                        # original or div4 PSNR
                        te_psnr = utils.calc_psnr(te_sr, te_hr, self.args.scale, self.args.rgb_range)
                    elif self.args.PSNR_ver == 2:
                        # patch-based PSNR
                        #te_hr = utils.hr_crop_for_pb_forward(te_hr, self.args)
                        te_psnr = utils.calc_psnr_pb_forward(self.args, te_sr, te_hr, self.args.scale,
                                                             self.args.rgb_range)
                    elif self.args.PSNR_ver == 4:
                        te_psnr = utils.calc_psnr_dpb_forward(self.args, te_sr, te_hr)


                    lr_name = self.args.save_test + '/images/' + te_name[0] + '_LR'
                    sr_name = self.args.save_test + '/images/' + te_name[0] + '_SR'
                    hr_name = self.args.save_test + '/images/' + te_name[0] + '_HR'

                    utils.save_tensor_to_image(self.args, te_lr, lr_name)
                    utils.save_tensor_to_image(self.args, te_hr, hr_name)
                    utils.save_tensor_to_image(self.args, te_sr, sr_name)

                    psnr_txt = '%s\t%f\n' % (te_name[0], te_psnr)
                    self.f_te_rec = open(self.f_te_fname, 'at')
                    self.f_te_rec.write(psnr_txt)
                    self.f_te_rec.close()
                    print(psnr_txt[:len(psnr_txt) - 1])

                    te_psnr_avg += te_psnr
                    te_psnr_cnt += 1

            te_psnr_avg /= te_psnr_cnt

            print('%d of tests are completed, average PSNR: [%.2f]' % (te_iter+1, te_psnr_avg))
        else:
            print('phase error!')




#
#
# class test():
#     def __init__(self, args, SR_model,
#                  te_dataset):
#         self.args = args
#         self.model = SR_model
#         self.te_dataset = te_dataset
#
#     def do_test(self):
#         self.SR_model.eval()
#         te_psnr_avg = 0.0
#         te_psnr_cnt = 0
#         for te_iter, (te_lr, te_hr, te_name) in enumerate(self.te_dataloader):
#             te_lr, te_hr = utils.tensor_prepare([te_lr, te_hr], self.args)
#             te_sr = self.SR_model(te_lr)
#             sr_save_name = self.args.save_trained + '_' + self.args.today_time + '/images/' + te_name + '_SR'
#             utils.save_tensor_to_image(self.args, te_sr, te_name)
#             te_psnr = utils.calc_psnr(te_sr, te_hr, self.args.scale, self.args.rgb_range)
#             print('%s, PSNR: [%.2f]' % (te_name, te_psnr))
#             te_psnr_avg += te_psnr
#             te_psnr_cnt += 1
#
#
#         te_psnr_avg /= te_psnr_cnt
#
#         print('%d of test is completed, Avg PSNR: [%.2f]' % (te_iter, te_psnr_avg))

