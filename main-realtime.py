import utils
import models
import torch.cuda
from loss import Set_Loss
from option import args
from phases import phases
from pytictoc import TicToc
from datetime import datetime
from datasets import Set_Dataset

'''##########################################'''
'''#### 0. prepare                        ###'''
'''##########################################'''
# set timer & set random seeds, store today's date and time
timer = TicToc()
torch.manual_seed(args.rnd_seed)
torch.cuda.manual_seed_all(args.rnd_seed)
args.today_time = str(datetime.now())[:10]+'_%02dh_%02dm' % \
                  (int(datetime.now().hour), int(datetime.now().minute))
args.device = torch.device('cpu' if args.cpu else 'cuda')

'''###########################################'''
'''#### 1. define datasets                 ###'''
'''###########################################'''
dset = Set_Dataset(args)

'''###########################################'''
'''#### 2. define loss functions           ###'''
'''###########################################'''
loss = Set_Loss(args)

'''###########################################'''
'''#### 3. define SR model                 ###'''
'''###########################################'''
SR_model = models.SR_model_selector(args).model_return()

'''############################################'''
'''#### 4. define optimizer & scheduler     ###'''
'''############################################'''
SR_optimizer = utils.make_optimizer(args, SR_model)
SR_lrscheduler = utils.make_lrscheduler(args, SR_optimizer)

'''############################################'''
'''#### 5. do training or testing           ###'''
'''############################################'''
p = phases(args)
if args.phase == 'train':
    timer.tic()
    print('start training... %s' % (datetime.now()))

    for epoch in range(args.num_epoch):
        p.do('train',
             epoch, SR_model, loss, SR_optimizer,
             dset.tr_dataloader, dset.vl_dataloader, dset.te_dataloader)
        p.do('valid',
             epoch, SR_model, loss, SR_optimizer,
             dset.tr_dataloader, dset.vl_dataloader, dset.te_dataloader)

        utils.sch_opt_step(SR_lrscheduler)
        utils.save_model(args, SR_model, epoch)
        utils.print_remained_time(timer, epoch, args.num_epoch)

elif args.phase == 'test':
    print('start test... %s' % (datetime.now()))
    p.do('test',
         -1, SR_model, loss, SR_optimizer,
         dset.tr_dataloader, dset.vl_dataloader, dset.te_dataloader)