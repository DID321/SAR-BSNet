"""
@time: 2025/06/16
@file: model.py
@author: WD                     ___       __   ________            
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ / 
                                ____/|__/      /_____/  

SARBSNet model for Pytorch
"""
import torch.nn as nn
from .backbone import mit_b0_s6
from .backbone import Conv
from .roa import ROAConvBackbone
import torch
import torch.nn.functional as F



class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SARBSNetHead(nn.Module):

    def __init__(self, in_channels=[4, 8, 16, 32, 64, 128], embedding_dim=768, output_channels=1, dropout_ratio=0.1):
        super(SARBSNetHead, self).__init__()
        c0_in_channels, c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = in_channels
        self.fuse_conv5 = Conv(c1=c5_in_channels*2, c2=c5_in_channels, k=3, s=1)
        self.fuse_conv4 = Conv(c1=c4_in_channels * 2, c2=c4_in_channels, k=3, s=1)
        self.fuse_conv3 = Conv(c1=c3_in_channels * 2, c2=c3_in_channels, k=3, s=1)
        self.fuse_conv2 = Conv(c1=c2_in_channels * 2, c2=c2_in_channels, k=3, s=1)
        self.fuse_conv1 = Conv(c1=c1_in_channels * 2, c2=c1_in_channels, k=3, s=1)
        self.fuse_conv0 = Conv(c1=c0_in_channels * 2, c2=c0_in_channels, k=3, s=1)

        self.conv4 = Conv(c1=c5_in_channels+c4_in_channels, c2=embedding_dim, k=3, s=1)
        self.conv3 = Conv(c1=c3_in_channels+embedding_dim, c2=embedding_dim, k=3, s=1)
        self.conv2 = Conv(c1=c2_in_channels+embedding_dim, c2=embedding_dim, k=3, s=1)
        self.conv1 = Conv(c1=c1_in_channels+embedding_dim, c2=embedding_dim, k=3, s=1)
        self.conv0 = Conv(c1=c0_in_channels+embedding_dim, c2=embedding_dim, k=3, s=1)

        self.linear_pred = nn.Conv2d(embedding_dim, output_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, trans_inputs, conv_inputs):

        t_c0, t_c1, t_c2, t_c3, t_c4, t_c5 = trans_inputs
        c0, c1, c2, c3, c4, c5 = conv_inputs

        # Stage5 先融合
        c5 = self.fuse_conv5(torch.concat([t_c5, c5], dim=1))
        ############## Conv decoder on C1-C4 ###########
        # 先升采样 [16, 16, 128] -> [32, 32, 128]
        c5 = F.interpolate(c5, size=c4.size()[2:], mode='bilinear', align_corners=False)
        # Stage4 融合
        c4 = self.fuse_conv4(torch.concat([t_c4, c4], dim=1))
        # 与c4concat [32, 32, 128] + [32, 32, 64] -> [32, 32, 192] -> [32, 32, embedding_dim]
        c4 = self.conv4(torch.concat([c5, c4], dim=1))

        # c4 先升采样[32, 32, 64] -> [64, 64, 64]
        c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
        # Stage3 融合
        c3 = self.fuse_conv3(torch.concat([t_c3, c3], dim=1))
        # 与c3concat [64, 64, 64] + [64, 64, 32] -> [64, 64, 96] -> [64, 64, embedding_dim]
        c3 = self.conv3(torch.concat([c4, c3], dim=1))

        # c3 先升采样[64, 64, 64] -> [128, 128, 64]
        c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        # Stage2 融合
        c2 = self.fuse_conv2(torch.concat([t_c2, c2], dim=1))
        # 与c2concat [128, 128, 64] + [128, 128, 16] -> [128, 128, 80] -> [128, 128, embedding_dim]
        c2 = self.conv2(torch.concat([c3, c2], dim=1))

        # c2 先升采样[128, 128, 64] -> [256, 256, 64]
        c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # Stage1 融合
        c1 = self.fuse_conv1(torch.concat([t_c1, c1], dim=1))
        # 与c1concat [256, 256, 64] + [256, 256, 8] -> [256, 256, 72] -> [256, 256, embedding_dim]
        c1 = self.conv1(torch.concat([c2, c1], dim=1))

        # c1 先升采样[256, 256, 64] -> [512, 512, 64]
        c1 = F.interpolate(c1, size=c0.size()[2:], mode='bilinear', align_corners=False)
        # Stage0 融合
        c0 = self.fuse_conv0(torch.concat([t_c0, c0], dim=1))
        # 与c0concat [512, 512, 64] + [512, 512, 4] -> [512, 512, 68] -> [512, 512, embedding_dim]
        c0 = self.conv0(torch.concat([c1, c0], dim=1))

        c0 = self.linear_pred(c0)

        return c0

class SARBSNet(nn.Module):
    def __init__(self, phi='b0', output_channels=1, pretrained=False, derections = ['horizontal', 'vertical', 'diagonal1', 'diagonal2'], kernel_size = [5, 7, 9, 11]):
        super(SARBSNet, self).__init__()
        # stage 1-5
        self.in_channels = {
            # 16, 16, 32, 64, 160, 256
            'b0': [4, 8, 16, 32, 64, 128],
        }[phi]
        # 不同模型对应不同的backbone 和embedding_dim
        self.trans_backbone = {
            'b0': mit_b0_s6,
        }[phi](pretrained)

        self.conv_backbone = ROAConvBackbone(derections, 3, self.in_channels, kernel_size)
        # 输出embedding_dim
        self.embedding_dim = {
            'b0': 64,
        }[phi]
        self.output_channels = output_channels
        # decode head
        self.decode_head = SARBSNetHead(self.in_channels, self.embedding_dim, self.output_channels)

    def forward(self, inputs):
        # 正向传播
        # H, W = inputs.size(2), inputs.size(3)
        # Transformer backbone
        x1 = self.trans_backbone.forward(inputs)
        x2 = self.conv_backbone.forward(inputs)
        # MLP decoder
        x = self.decode_head.forward(x1, x2)
        # 上采样为原始输入的大小
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

