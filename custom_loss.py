import scipy
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#
# class GaussianLayer(nn.Module):
#     def __init__(self):
#         super(GaussianLayer, self).__init__()
#         self.seq = nn.Sequential(
#             nn.ReflectionPad2d(10),
#             nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None, groups=3)
#         )
#
#         self.weights_init()
#
#     def forward(self, x):
#         return self.seq(x)
#
#     def weights_init(self):
#         n= np.zeros((21,21))
#         n[10,10] = 1
#         k = scipy.ndimage.gaussian_filter(n,sigma=3)
#         for name, f in self.named_parameters():
#             f.data.copy_(torch.from_numpy(k))


def matlab_style_gauss2D(shape=(3,3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class sGAE(nn.modules.loss._Loss):
    def __init__(self, args, device, p_h=20):
        super(sGAE, self).__init__()
        self.batch_size = args.batch_size
        self.size_img = args.patch_size

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.gker = matlab_style_gauss2D((self.size_img, self.size_img),
                                         sigma=self.size_img/8)
        self.gker = torch.from_numpy(self.gker)
        self.gker = self.gker * self.gker
        self.gker = self.gker / torch.sum(self.gker)
        self.gker = self.gker * args.sgae_gamma
        self.gker = self.gker.to(device)

    def forward(self, input, target):
        diff = input-target
        gerr = torch.sum(self.gker * torch.abs(diff))/self.batch_size

        return gerr


class GAE(nn.modules.loss._Loss):
    def __init__(self, args, device, p_h=20):
        super(GAE, self).__init__()
        self.batch_size = args.batch_size
        self.size_img = args.patch_size
        self.p_h2 = p_h*p_h
        self.seq = nn.Sequential(
            #nn.ReflectionPad2d(10),
            nn.Conv2d(3, 3, self.size_img,
                      stride=1, padding=0, bias=None, groups=3)
        )
        # prevent update
        for param in self.seq.parameters():
            param.requires_grad = False

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.gker = matlab_style_gauss2D((self.size_img, self.size_img),
                                         sigma=self.size_img/8)
        self.gker = self.gker*args.gae_gamma
        self.gker = torch.from_numpy(self.gker)
        self.gker = self.gker.to(device)
        self.weights_init()

    def forward(self, input, target):
        # F.l1_loss(input, target, reduction='elementwise_mean')
        diff = input-target

        gerr = torch.sum(self.gker * torch.abs(diff))/self.batch_size

        #return -10000*torch.exp(-g_sim/self.p_h2)
        return gerr

    def weights_init(self):
        g_ker = utils.gen_gauss_kernel(self.size_img)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(g_ker))