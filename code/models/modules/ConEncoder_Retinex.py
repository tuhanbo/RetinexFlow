import functools
import models.modules.module_util as mutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import opt_get
from ipdb import set_trace as st


##-------------------- Decomposition  -----------------------
class ConEncoder_Retinex(nn.Module):
    def __init__(self, bias=False):
        super(ConEncoder_Retinex, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)

        self.conv1 = RCBdown(n_feat=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = RCBdown(n_feat=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = RCBdown(n_feat=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = RCBdown(n_feat=64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = RCBdown(n_feat=64)

        self.upv6_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6_R = RCBup_R(n_feat=64)
        self.upv6_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6_L = RCBup_L(n_feat=64)

        self.upv7_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7_R = RCBup_R(n_feat=64)
        self.upv7_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7_L = RCBup_L(n_feat=64)

        self.upv8_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8_R = RCBup_R(n_feat=64)
        self.upv8_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8_L = RCBup_L(n_feat=64)

        self.upv9_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9_R = RCBup_R(n_feat=64)
        self.upv9_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9_L = RCBup_L(n_feat=64)

        self.conv10_R = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.conv10_L = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.conv_conditionInFea = nn.Conv2d(64, 32, kernel_size=1, stride=1)

    def forward(self, x, get_steps=False):

        input_max = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat((x, input_max), dim=1)

        conv1 = self.lrelu(self.conv_in(x))

        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        # R
        up_R = self.upv6_R(conv5)
        up_R = torch.cat([up_R, conv4], 1)
        up_R = self.conv6_R(up_R)
        # L
        up_L = self.upv6_L(conv5)
        up_L = self.conv6_L(up_L)
        # R
        up_R1 = self.upv7_R(up_R)
        up_R1 = torch.cat([up_R1, conv3], 1)
        up_R1 = self.conv7_R(up_R1)
        up_R11 = self.pool1(up_R1)
        # up_R11 = self.conv_conditionInFea(up_R11)


        # L
        up_L1 = self.upv7_L(up_L)
        up_L1 = self.conv7_L(up_L1)
        up_L11 = self.pool1(up_L1)
        # up_L11 = self.conv_conditionInFea(up_L11)

        # R
        up_R2 = self.upv8_R(up_R1)
        up_R2 = torch.cat([up_R2, conv2], 1)
        up_R2 = self.conv8_R(up_R2)
        up_R22 = self.pool2(up_R2)
        # up_R22 = self.conv_conditionInFea(up_R22)
        # L
        up_L2 = self.upv8_L(up_L1)
        up_L2 = self.conv8_L(up_L2)
        up_L22 = self.pool2(up_L2)
        # up_L22 = self.conv_conditionInFea(up_L22)

        # R
        up_R3 = self.upv9_R(up_R2)
        up_R3 = torch.cat([up_R3, conv1], 1)
        up_R3 = self.conv9_R(up_R3)
        up_R33 = self.pool3(up_R3)
        # up_R33 = self.conv_conditionInFea(up_R33)

        # L
        up_L3 = self.upv9_L(up_L2)
        up_L3 = self.conv9_L(up_L3)
        up_L33 = self.pool3(up_L3)
        # up_L33 = self.conv_conditionInFea(up_L33)

        R = torch.sigmoid(self.conv10_R(up_R3))
        L = torch.sigmoid(self.conv10_L(up_L3))

        fea_up2 = torch.cat((up_L11, up_R11), dim=1)
        fea_up4 = torch.cat((up_L22, up_R22), dim=1)
        fea_up8 = torch.cat((up_L33, up_R33), dim=1)

        results = {'fea_up0': fea_up2,

                   'fea_up1': fea_up4,

                   'fea_up2': fea_up8,

                   'fea_up4': fea_up8,

                   'last_lr_fea': fea_up4,

                   'Reflection_map': R,

                   'Illuminance_map': L
                   }

        if get_steps:
            return results
        else:
            return None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

class NoEncoder(nn.Module):
    def __init__(self, bias=False):
        super(NoEncoder, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)

        self.conv1 = RCBdown(n_feat=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = RCBdown(n_feat=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = RCBdown(n_feat=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = RCBdown(n_feat=64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = RCBdown(n_feat=64)

        self.upv6_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6_R = RCBup_R(n_feat=64)
        self.upv6_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6_L = RCBup_L(n_feat=64)

        self.upv7_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7_R = RCBup_R(n_feat=64)
        self.upv7_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7_L = RCBup_L(n_feat=64)

        self.upv8_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8_R = RCBup_R(n_feat=64)
        self.upv8_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8_L = RCBup_L(n_feat=64)

        self.upv9_R = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9_R = RCBup_R(n_feat=64)
        self.upv9_L = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9_L = RCBup_L(n_feat=64)

        self.conv10_R = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.conv10_L = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.conv_conditionInFea = nn.Conv2d(64, 32, kernel_size=1, stride=1)

    def forward(self, x, get_steps=False):

        input_max = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat((x, input_max), dim=1)

        conv1 = self.lrelu(self.conv_in(x))

        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        # R
        up_R = self.upv6_R(conv5)
        up_R = torch.cat([up_R, conv4], 1)
        up_R = self.conv6_R(up_R)
        # L
        up_L = self.upv6_L(conv5)
        up_L = self.conv6_L(up_L)
        # R
        up_R1 = self.upv7_R(up_R)
        up_R1 = torch.cat([up_R1, conv3], 1)
        up_R1 = self.conv7_R(up_R1)
        up_R11 = self.pool1(up_R1)
        # up_R11 = self.conv_conditionInFea(up_R11)


        # L
        up_L1 = self.upv7_L(up_L)
        up_L1 = self.conv7_L(up_L1)
        up_L11 = self.pool1(up_L1)
        # up_L11 = self.conv_conditionInFea(up_L11)

        # R
        up_R2 = self.upv8_R(up_R1)
        up_R2 = torch.cat([up_R2, conv2], 1)
        up_R2 = self.conv8_R(up_R2)
        up_R22 = self.pool2(up_R2)
        # up_R22 = self.conv_conditionInFea(up_R22)
        # L
        up_L2 = self.upv8_L(up_L1)
        up_L2 = self.conv8_L(up_L2)
        up_L22 = self.pool2(up_L2)
        # up_L22 = self.conv_conditionInFea(up_L22)

        # R
        up_R3 = self.upv9_R(up_R2)
        up_R3 = torch.cat([up_R3, conv1], 1)
        up_R3 = self.conv9_R(up_R3)
        up_R33 = self.pool3(up_R3)
        # up_R33 = self.conv_conditionInFea(up_R33)

        # L
        up_L3 = self.upv9_L(up_L2)
        up_L3 = self.conv9_L(up_L3)
        up_L33 = self.pool3(up_L3)
        # up_L33 = self.conv_conditionInFea(up_L33)

        R = torch.sigmoid(self.conv10_R(up_R3))
        L = torch.sigmoid(self.conv10_L(up_L3))

        fea_up2 = torch.cat((up_L11, up_R11), dim=1)
        fea_up4 = torch.cat((up_L22, up_R22), dim=1)
        fea_up8 = torch.cat((up_L33, up_R33), dim=1)


        results = {'fea_up0': fea_up2*0,

                   'fea_up1': fea_up4*0,

                   'fea_up2': fea_up8*0,

                   'fea_up4': fea_up8*0,

                   'last_lr_fea': fea_up4*0,

                   'Reflection_map': R,

                   'Illuminance_map': L
                   }

        if get_steps:
            return results
        else:
            return None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

class ContextBlock(nn.Module):
    def __init__(self, n_feat, opt=None, bias=False):
        self.opt = opt
        super(ContextBlock, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2)
        )

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, H, W]
        inp = x
        inp = self.head(inp)

        # [N, C, 1, 1]
        context = self.modeling(inp)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        inp = inp + channel_add_term
        x = x + self.act(inp)

        return x


class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = x
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res


class RCBup_L(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBup_L, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = x
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res


class RCBup_R(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBup_R, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body_head = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=1, stride=1, bias=bias, groups=groups),
            act
        )

        self.body = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = self.body_head(x)
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res


##########################################################################
##---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()
        # nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


