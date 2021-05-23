import custom_loss
import torch.nn as nn

class Set_Loss():
    def __init__(self, args):
        if args.loss == 'MSE':
            SR_loss = nn.MSELoss()
        elif args.loss == 'L1':
            SR_loss = nn.L1Loss()
        elif args.loss == 'GAE':
            SR_loss = custom_loss.GAE(args, args.device)
        elif args.loss == 'sGAE':
            SR_loss = custom_loss.sGAE(args, args.device)

        SR_loss.to(args.device)
        self.SR_loss = SR_loss