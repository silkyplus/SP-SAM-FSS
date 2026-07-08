import math
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.
    """

    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depth-wise convolution module."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class MultiScaleExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # First stage
        self.dwconv3_1 = DWConv(channels,channels, 3)
        self.dwconv5_1 = DWConv(channels,channels, 5)
        # Second stage
        self.dwconv3_2 = DWConv(2 * channels, channels,3)
        self.dwconv5_2 = DWConv(2 * channels, channels,5)
        # Fusion
        self.fuse_conv = nn.Conv2d(2 * channels, channels, kernel_size=1,stride=1,padding=0)

        self.LN = nn.LayerNorm(channels)
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.LN(x)
        # back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        x3_1 = self.dwconv3_1(x)
        x5_1 = self.dwconv5_1(x)
        cat1 = torch.cat([x3_1, x5_1], dim=1)

        x3_2 = self.dwconv3_2(cat1)
        x5_2 = self.dwconv5_2(cat1)
        cat2 = torch.cat([x3_2, x5_2], dim=1)

        fused = self.fuse_conv(cat2)
        return fused

class MaskLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cg1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim)
        self.cg2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.cg3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        m1 = self.cg1(x)
        m2 = self.cg2(x)
        m3 = self.cg3(x)
        m = self.softmax(m1 + m2 + m3)
        return x * m
class SMMM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1x1 projection
        self.encoder_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.decoder_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Shared multi-scale extractor
        self.extractor = MultiScaleExtractor(channels)
        # Mask layers
        self.mask = MaskLayer(channels)
        # Final fusion
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(4, channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x_enc, x_dec):
        x1 = self.encoder_proj(x_enc)
        x2 = self.decoder_proj(x_dec)
        # Apply shared extractor to both branches
        f1 = self.extractor(x1)
        f2 = self.extractor(x2)
        # Apply  mask layers*3 to encoder branch
        merged = self.mask(f1) + self.mask(f2)
        # Final fusion
        out = self.dilated_conv(merged)
        out = self.norm(out)
        out = self.out_conv(out)
        return out
# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 SMMM 模块
    SMMM = SMMM(32)
    # 将输入图像传入SMMM 模块进行处理
    output = SMMM(input1,input2)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合怪即插即用模块永久更新-SMMM_input_size:', input1.size())
    print('Ai缝合怪即插即用模块永久更新-SMMM_output_size:', output.size())
