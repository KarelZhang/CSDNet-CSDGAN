import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import torch.nn.functional as F
import random
from . import networks
# from . import networks
import sys
from .ssim import SSIM


class PairModel(BaseModel):
    def name(self):
        return 'PairModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        self.max = networks.max_operation()
        self.edge = networks.edge_operation()
        self.vgg_loss = networks.PerceptualLoss(opt)

        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./vgg_model", self.gpu_ids)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.net = networks.define_network(opt.network_model, self.gpu_ids)

        self.old_lr = opt.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        print('---------- Networks initialized -------------')
        networks.print_network(self.net)


        self.net.train()
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_A_gray_o = self.real_A_gray
        self.real_img = Variable(self.input_img)
        
        self.max_out = self.max(self.real_A_gray_o)
        self.edge_out = self.edge(self.real_A_gray_o)
        self.gadience = self.max_out + self.edge_out
        self.real_A_gray = torch.cat([self.gadience, self.real_A_gray], 1)
        
        self.fake_B, self.latent_real_A, self.gray= self.net.forward(self.real_img, self.real_A_gray)
        
    def backward(self, epoch):

        mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

        self.loss_MSE = mse_fn(self.fake_B, self.real_B)
        self.loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, self.real_B)
        self.smooth_loss = F.smooth_l1_loss(self.gray, self.real_A_gray_o)

        self.loss = self.loss_MSE + self.loss_vgg + self.smooth_loss
        self.loss.backward()

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer.zero_grad()
        self.backward(epoch)
        self.optimizer.step()
        # D_A

    def get_current_errors(self, epoch):
        loss = self.loss.data[0]
        # if self.opt.vgg > 0:
        #     vgg = self.loss_vgg_b.data[0]/self.opt.vgg if self.opt.vgg > 0 else 0
        return OrderedDict([ ('loss', loss)])#, ("vgg", vgg)])

    def save(self, label):
        self.save_network(self.net, 'net', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
