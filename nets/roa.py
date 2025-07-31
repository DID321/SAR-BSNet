"""
@time: 2025/06/18
@file: SAR_ROA.py
@author: WD                     ___       __   ________            
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ / 
                                ____/|__/      /_____/  

SAR ROA梯度算子
"""
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .backbone import C2f


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # print(c1, c2, k, s, autopad(k, p, d), g, d)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))


class DownConv(Conv):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__(c1, c2, k, s, d)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.act(self.bn(self.conv(x))))

class C2fModule(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        self.conv = Conv(c1=c1, c2=c2, k=k, s=s)
        self.c2f = C2f(c2, c2, shortcut=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2f(self.conv(x))

class ROAConv(nn.Module):
    def __init__(self, derection='horizontal', in_channel=3, out_channel=1, kernel_size=3, stride=1, padding=None,
                 dilation=1):
        super(ROAConv, self).__init__()
        self.kernel_size = kernel_size
        self.derection = derection
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = kernel_size // 2

        self.reflectionPad = nn.ReflectionPad2d(padding=(self.padding, self.padding, self.padding, self.padding))
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, stride, bias=False)
        self.conv2 = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, stride, bias=False)

        self.bn = nn.BatchNorm2d(self.out_channel)
        # 使用Sigmoid激活函数
        self.act = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the convolutional weights to the ROA kernel
        # 根据ROA梯度算子定义不同尺寸的卷积核参数
        if self.derection == 'horizontal':
            kernel1 = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            kernel2 = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            kernel1[:self.kernel_size // 2, :] = 1
            kernel2[self.kernel_size // 2 + 1:, :] = 1
        # 垂直方向的卷积核
        elif self.derection == 'vertical':
            kernel1 = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            kernel2 = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
            kernel1[:, :self.kernel_size // 2] = 1
            kernel2[:, self.kernel_size // 2 + 1:] = 1
        # 对角线方向的卷积核
        elif self.derection == 'diagonal1':
            kernel1 = np.tri(self.kernel_size, k=-1, dtype=np.float32)
            kernel2 = kernel1.T
        # 对角线方向的卷积核，旋转90度
        elif self.derection == 'diagonal2':
            kernel1 = np.tri(self.kernel_size, k=-1, dtype=np.float32)
            kernel2 = kernel1.T
            kernel1 = np.rot90(kernel1, 1)
            kernel2 = np.rot90(kernel2, 1)
        # 将numpy数组转换为连续的内存布局
        kernel1 = np.ascontiguousarray(kernel1, dtype=np.float32)
        kernel2 = np.ascontiguousarray(kernel2, dtype=np.float32)

        # 将numpy数组转换为torch张量并设置为卷积核权重
        kernel1 = torch.from_numpy(kernel1).unsqueeze(0).unsqueeze(0).repeat(self.out_channel, self.in_channel, 1, 1)
        kernel2 = torch.from_numpy(kernel2).unsqueeze(0).unsqueeze(0).repeat(self.out_channel, self.in_channel, 1, 1)
        # 将卷积核参数赋值给卷积层
        self.conv1.weight.data = kernel1
        self.conv2.weight.data = kernel2
        # 设置卷积层参数固定不更新
        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False

    def forward(self, x):
        x = self.reflectionPad(x)
        # x = torch.log(self.conv1(x) / (self.conv2(x)))  # 防止除0错误
        x = self.conv1(x) / (self.conv2(x) + 1e-7)
        # x = torch.log(x3)
        return self.bn(x)
        # return self.act(x)
        # return x


class ROAConvModule(nn.Module):
    def __init__(self, derections=['horizontal', 'vertical', 'diagonal1', 'diagonal2'], in_channel=3, mid_channel=1,
                 out_channel=4, kernel_size=[3, 5, 7, 9], stride=1, padding=None, dilation=1):
        super(ROAConvModule, self).__init__()
        self.kernel_size = kernel_size
        self.derections = derections
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel

        self.fuse_in_channel = None
        if len(self.derections) == 2:
            self.fuse_in_channel = (len(self.derections) + 1) * len(self.kernel_size) + self.in_channel
        elif len(self.derections) == 4:
            self.fuse_in_channel = (len(self.derections) + 2) * len(self.kernel_size) + self.in_channel
        else:
            print("❌ ERROR: The length of derection input must be 2 or 4, but current is %d" % (len(self.derections)))
            return
        # print('fuse_in_channel: ', self.fuse_in_channel)

        self.conv_blocks = nn.ModuleList()
        for direction in self.derections:
            for k in self.kernel_size:
                self.conv_blocks.append(
                    ROAConv(direction, self.in_channel, self.mid_channel, k, stride, padding, dilation))

        self.fuse_conv = Conv(c1=self.fuse_in_channel, c2=self.out_channel, k=1, s=1)

    def forward(self, x):
        outputs = []
        for conv in self.conv_blocks:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, dim=1)  # 将所有方向的输出拼接在一起
        Gm1, Gm2 = None, None  # 初始化Gm1和Gm2
        if 'horizontal' in self.derections:
            horizontal_output = outputs[:, :1 * len(self.kernel_size)]
            vertical_output = outputs[:, 1 * len(self.kernel_size):2 * len(self.kernel_size)]
            Gm1 = torch.sqrt(horizontal_output ** 2 + vertical_output ** 2)
        if 'diagonal1' in self.derections:
            diagonal1_output = outputs[:, 2 * len(self.kernel_size):3 * len(self.kernel_size)]
            diagonal2_output = outputs[:, 3 * len(self.kernel_size):]
            Gm2 = torch.sqrt(diagonal1_output ** 2 + diagonal2_output ** 2)

        if Gm1 is None:
            x = torch.cat([outputs, Gm2, x], dim=1)
        elif Gm2 is None:
            x = torch.cat([outputs, Gm1, x], dim=1)
        else:
            x = torch.cat([outputs, Gm1, Gm2, x], dim=1)
        return self.fuse_conv(x)


class ROAConvBackbone(nn.Module):
    def __init__(self, derections=['horizontal', 'vertical', 'diagonal1', 'diagonal2'], in_channel=3,
                 out_channels=[4, 8, 16, 32, 64, 128], kernel_size=[3, 5, 7, 9]):
        super(ROAConvBackbone, self).__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        # ROA Conv Stage0 [512, 512, 3] -> [512, 512, 4]
        self.roa_conv = ROAConvModule(derections=derections, in_channel=self.in_channel, mid_channel=1,
                                      out_channel=self.out_channels[0], kernel_size=kernel_size)
        # Stage1 -5
        self.conv_blocks = nn.ModuleList()
        for i in range(len(self.out_channels)-1):
            self.conv_blocks.append(DownConv(c1=self.out_channels[i], c2=self.out_channels[i + 1], k=3, s=1))

    def forward(self, x):
        outputs = []
        x = self.roa_conv(x)
        outputs.append(x)

        for conv in self.conv_blocks:
            x = conv(x)
            outputs.append(x)
        return outputs


class ROAC2fBackbone(nn.Module):
    def __init__(self, derections=['horizontal', 'vertical', 'diagonal1', 'diagonal2'], in_channel=3,
                 out_channels=[4, 8, 16, 32, 64, 128], kernel_size=[3, 5, 7, 9]):
        super(ROAC2fBackbone, self).__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        # ROA Conv Stage0 [512, 512, 3] -> [512, 512, 4]
        self.roa_conv = ROAConvModule(derections=derections, in_channel=self.in_channel, mid_channel=1,
                                      out_channel=self.out_channels[0], kernel_size=kernel_size)
        # Stage1 -5
        self.conv_blocks = nn.ModuleList()
        for i in range(len(self.out_channels)-1):
            self.conv_blocks.append(C2fModule(c1=self.out_channels[i], c2=self.out_channels[i + 1], k=3, s=2))

    def forward(self, x):
        outputs = []
        x = self.roa_conv(x)
        outputs.append(x)

        for conv in self.conv_blocks:
            x = conv(x)
            outputs.append(x)
        return outputs

# def main():
#
#     save_dir = './ROA_output'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     # 读取一张SAR图像
#     img = cv2.imread('E:/Pytorch_DL/data1_Looknum_1_019_028_X-70_70_Y-70_70__BP_rotate_crop_4.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (512, 512))
#     img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # 转换为NCHW格式
#     print(img.shape)
#
#     derections = ['horizontal', 'vertical']
#     kernel_size = [5, 9, 13, 17]
#     # 定义ROA梯度算子
#     roa_conv = ROAConvModule(derections=derections, in_channel=3, out_channel=1, kernel_size=kernel_size)
#
#     # 将图像输入到ROA梯度算子中
#     output = roa_conv(img)
#     print(output.shape)
#
#     # horizontal_output = output[:, :1 * len(kernel_size)]
#     # vertical_output = output[:, 1 * len(kernel_size):2 * len(kernel_size)]
#     # diagonal1_output = output[:, 2 * len(kernel_size):3 * len(kernel_size)]
#     # diagonal2_output = output[:, 3 * len(kernel_size):]
#     # print(horizontal_output.shape, vertical_output.shape, diagonal1_output.shape, diagonal2_output.shape)
#     #
#     # Gm1 = torch.sqrt(horizontal_output ** 2 + vertical_output ** 2)
#     # Gm2 = torch.sqrt(diagonal1_output ** 2 + diagonal2_output ** 2)
#     # # 将Gm1, Gm2, horizontal 输出结果保存图像
#     # for i, k in enumerate(kernel_size):
#     #     img1 = horizontal_output[:, i:i + 1, :, :].squeeze(0).squeeze(0).detach().numpy()
#     #     img2 = vertical_output[:, i:i + 1, :, :].squeeze(0).squeeze(0).detach().numpy()
#     #     img1 = (img1 - img1.min()) / (img1.max() - img1.min()) * 255
#     #     img2 = (img2 - img2.min()) / (img2.max() - img2.min()) * 255
#     #     img1 = img1.astype(np.uint8)
#     #     img2 = img2.astype(np.uint8)
#     #
#     #     cv2.imwrite(os.path.join(save_dir, f'ROA_horizontal_k{k}.png'), img1)
#     #     cv2.imwrite(os.path.join(save_dir, f'ROA_vertical_k{k}.png'), img2)
#     #
#     #     img1 = Gm1[:, i:i + 1, :, :].squeeze(0).squeeze(0).detach().numpy()
#     #     img2 = Gm2[:, i:i + 1, :, :].squeeze(0).squeeze(0).detach().numpy()
#     #     print(img1.max(), img1.min())
#     #     # 展示图像
#     #     plt.subplot(1, 2, 1)
#     #     plt.imshow(img1)
#     #     plt.title(f'ROA_horizontal_k{k}')
#     #     plt.subplot(1, 2, 2)
#     #     plt.imshow(img2)
#     #     plt.title(f'ROA_vertical_k{k}')
#     #     plt.show()
#     #     img1 = (img1 - img1.min()) / (img1.max() - img1.min()) * 255
#     #     img2 = (img2 - img2.min()) / (img2.max() - img2.min()) * 255
#     #
#     #     img1 = img1.astype(np.uint8)
#     #     img2 = img2.astype(np.uint8)
#     #
#     #     cv2.imwrite(os.path.join(save_dir, f'ROA_Gm1_k{k}.png'), img1)
#     #     cv2.imwrite(os.path.join(save_dir, f'ROA_Gm2_k{k}.png'), img2)
#     #
#     #
#     # # 将输出结果保存图像
#     # for derection in derections:
#     #     for k in kernel_size:
#     #         img = output.pop(0).squeeze(0).squeeze(0).detach().numpy()
#     #         img = (img - img.min()) / (img.max() - img.min()) * 255
#     #         img = img.astype(np.uint8)
#     #         cv2.imwrite(os.path.join(save_dir, f'ROA_{derection}_k{k}.png'), img)
#
#     # k = 3
#     # # --------------horizontal-----------------#
#     # # kernel1 = np.zeros((k, k), dtype=np.float32)
#     # # kernel2 = np.zeros((k, k), dtype=np.float32)
#     # # kernel1[:k//2, :] = 1
#     # # kernel2[k//2+1:, :] = 1
#     # # --------------vertical-----------------#
#     # # kernel1 = np.zeros((k, k), dtype=np.float32)
#     # # kernel2 = np.zeros((k, k), dtype=np.float32)
#     # # kernel1[:, :k // 2]     = 1
#     # # kernel2[:, k // 2 + 1:] = 1
#     # # --------------diagonal1-----------------#
#     # kernel1 = np.tri(k, k=-1, dtype=np.float32)
#     # kernel2 = kernel1.T
#     # # --------------diagonal2-----------------#
#     # # kernel1 = np.tri(k, k=-1, dtype=np.float32)
#     # # kernel2 = kernel1.T
#     # # kernel1 = np.rot90(kernel1)
#     # # kernel2 = np.rot90(kernel2)
#     # print(kernel1)
#     # print(kernel2)
#     # # kernel1 = torch.from_numpy(kernel1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
#     # # kernel2 = torch.from_numpy(kernel2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
#     # # print(kernel1)


# if __name__ == "__main__":
#     img = torch.rand((3, 3, 512, 512), dtype=torch.float32)
#     derections = ['horizontal', 'vertical', 'diagonal1', 'diagonal2']
#     kernel_size = [5, 9, 13]
#     # 定义ROA梯度算子
#     # roa_conv = ROAConvModule(derections=derections, in_channel=3, mid_channel=1, out_channel=4, kernel_size=kernel_size)
#     model = ROAConvBackbone(derections=derections, in_channel=3,out_channels=[4, 8, 16, 32, 64, 128], kernel_size=kernel_size)
#     # 将图像输入到ROA梯度算子中
#     outputs = model(img)
#     for i in outputs:
#         print(i.shape)
