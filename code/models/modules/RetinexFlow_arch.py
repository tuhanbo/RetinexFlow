
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from models.modules.ConEncoder_Retinex import ConEncoder_Retinex, NoEncoder
from models.modules.loss import ColorLoss2
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast
from ipdb import set_trace as st

class RetinexFlow(nn.Module):
    def __init__(self, K=None, opt=None, step=None):
        super(RetinexFlow, self).__init__()
        self.crop_size = opt['datasets']['train']['GT_size']
        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        if opt['cond_encoder'] == 'ConEncoder_Retinex':
            self.RRDB = ConEncoder_Retinex()
        elif opt['cond_encoder'] ==  'NoEncoder':
            self.RRDB = NoEncoder()
        else:
            print('WARNING: Cannot find the conditional encoder %s, select RRDBNet by default.' % opt['cond_encoder'])


        self.ColorLoss = ColorLoss2()

        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((self.crop_size, self.crop_size, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
        self.i = 0
        if self.opt['to_yuv']:
            self.A_rgb2yuv = torch.nn.Parameter(torch.tensor([[0.299, -0.14714119, 0.61497538],
                                                              [0.587, -0.28886916, -0.51496512],
                                                              [0.114, 0.43601035, -0.10001026]]), requires_grad=False)
            self.A_yuv2rgb = torch.nn.Parameter(torch.tensor([[1., 1., 1.],
                                                              [0., -0.39465, 2.03211],
                                                              [1.13983, -0.58060, 0]]), requires_grad=False)

    def rgb2yuv(self, rgb):
        rgb_ = rgb.transpose(1, 3)  # input is 3*n*n   default
        yuv = torch.tensordot(rgb_, self.A_rgb2yuv, 1).transpose(1, 3)
        return yuv

    def yuv2rgb(self, yuv):
        yuv_ = yuv.transpose(1, 3)  # input is 3*n*n   default
        rgb = torch.tensordot(yuv_, self.A_yuv2rgb, 1).transpose(1, 3)
        return rgb

    @autocast()
    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None, align_condition_feature=False, get_color_map=False):
        if get_color_map:
            color_lr = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            return color_lr, color_gt
        if not reverse:
            if epses is not None and gt.device.index is not None:
                epses = epses[gt.device.index]
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label, align_condition_feature=align_condition_feature)
        else:
            # assert lr.shape[0] == 1

            assert lr.shape[1] == 3 or lr.shape[1] == 6
            # assert lr.shape[2] == 20
            # assert lr.shape[3] == 20
            # assert z.shape[0] == 1
            # assert z.shape[1] == 3 * 8 * 8
            # assert z.shape[2] == 20
            # assert z.shape[3] == 20
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise, gt=gt)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise, gt=gt)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None,
                    align_condition_feature=False):

        if self.opt['to_yuv']:
            gt = self.rgb2yuv(gt)
        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)
            gt_enc = self.rrdbPreprocessing(gt)


        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot)

        objective = logdet.clone()

        # if isinstance(epses, (list, tuple)):
        #     z = epses[-1]
        # else:
        #     z = epses
        z = epses

        if self.RRDB is not None:
            p = random.random()
            mean = squeeze2d(lr_enc['Reflection_map'], 8) if p > self.opt['train_with_gt_R'] else squeeze2d(gt_enc['Reflection_map'], 8)

        else:
            mean = squeeze2d(lr[: ,:3] ,8)

        objective = objective + flow.GaussianDiag.logp(mean, torch.tensor(0.).to(z.device), z)
        nll = (-objective) / float(np.log(2.) * pixels)

        R_low, I_low = lr_enc['Reflection_map'], lr_enc['Illuminance_map']
        R_high, I_high = gt_enc['Reflection_map'], gt_enc['Illuminance_map']
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)

        recon_loss_low = F.l1_loss(R_low * I_low_3, lr)
        recon_loss_high = F.l1_loss(R_high * I_high_3, gt)
        recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, lr)
        recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, gt)
        equal_R_loss = F.l1_loss(R_low, R_high.detach())

        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)
        color_loss = self.ColorLoss(R_low, R_high.detach())

        weight_dict = self.opt['train']['weight_Retinex']
        weight_list = []
        for key, value in weight_dict.items():
            weight_list.append(value)

        loss_Decom = weight_list[0] * recon_loss_low + \
                     weight_list[1] * recon_loss_high + \
                     weight_list[2] * recon_loss_mutal_low + \
                     weight_list[3] * recon_loss_mutal_high + \
                     weight_list[4] * Ismooth_loss_low + \
                     weight_list[5] * Ismooth_loss_high + \
                     weight_list[6] * equal_R_loss + \
                     weight_list[7] * color_loss

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, loss_Decom, logdet

    def rrdbPreprocessing(self, x):
        rrdbResults = self.RRDB(x, get_steps=True)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True, gt=None):

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)
            lr_reflectogram = lr_enc['Reflection_map']
            if gt is not None:
                gt_enc = self.rrdbPreprocessing(gt)
                gt_reflectogram = gt_enc['Reflection_map']
            else:
                gt_reflectogram = None

        if self.opt['cond_encoder'] == "NoEncoder":
            z = squeeze2d(lr[: ,:3] ,8)
        else:
            z = squeeze2d(lr_enc['Reflection_map'], 8)

        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)

        if self.opt['to_yuv']:
            x = self.yuv2rgb(x)
        return x, logdet, lr_reflectogram, gt_reflectogram

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)



