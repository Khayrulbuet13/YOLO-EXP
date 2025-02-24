import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import make_anchors


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           groups=conv.groups,
                           bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module):
    """Same as original."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(nn.Module):
    """Same as original (used inside CSP)."""
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = nn.Sequential(
            Conv(ch, ch, 3),
            Conv(ch, ch, 3)
        )

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(nn.Module):
    """Same as original (used in DarkFPN)."""
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(nn.Module):
    """Same as original (used in old DarkNet, might not be used now, 
       but we keep it if the FPN or Head references it)."""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


# -------------------------------------------------------------------------
#   BELOW: PicoDet (ESNet) Implementation
# -------------------------------------------------------------------------

acts = {
    "relu": nn.ReLU(inplace=True),
    "hard_swish": nn.Hardswish(inplace=True),
}

def make_divisible(v, divisor=16, min_value=None):
    """Utility from original PicoDet ESNet code."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNLayer(nn.Module):
    """Conv + BN + Activation used by ESNet blocks."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Identity() if act is None else acts[act]

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return self.act(y)


class SEModule(nn.Module):
    """Squeeze-and-Excitation block for ESNet."""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1, 1, 0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return inputs * outputs


def channel_shuffle(x, groups):
    """ShuffleNet channel shuffle, used by ESNet blocks."""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    """ESNet block that does NOT downsample."""
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super().__init__()
        # split channels in half, operate on x2
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)
        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.chunk(inputs, chunks=2, dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], axis=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], axis=1)
        out = channel_shuffle(out, 2)
        return out


class InvertedResidualDS(nn.Module):
    """ESNet block that DOES downsample."""
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super().__init__()
        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

        # extra conv after merging
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], axis=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)
        return out


class ESNet(nn.Module):
    """
    PicoDet's ESNet. 
    By default, we choose scale=0.5 so that the final c3,c4,c5 
    will be [64,128,256] channels at strides (8,16,32), 
    matching the original DarkNet p3,p4,p5 shape for YOLO-FPN.
    """
    def __init__(self,
                 scale=0.5,
                 act="hard_swish",
                 feature_maps=[4, 11, 14],
                 channel_ratio=None):
        super().__init__()
        if channel_ratio is None:
            # 13 total blocks for default stage_repeats=[3,7,3], 
            # each index can be a ratio
            channel_ratio = [1] * 13

        self.scale = scale
        self.feature_maps = feature_maps
        self._out_channels = []
        self._feature_idx = 0

        # stage blocks repeated times
        stage_repeats = [3, 7, 3]

        # base channels
        # Index: 1 => conv1 out, 2 => stage1 out, 3 => stage2 out, 4 => stage3 out
        # with scale=0.5 => they become [-1, 24, 64, 128, 256, 1024]
        stage_out_channels = [
            -1, 24,
            make_divisible(128 * scale),
            make_divisible(256 * scale),
            make_divisible(512 * scale),
            1024
        ]

        # conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # build the stages
        block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales), 8)
                if i == 0:
                    # downsampling block
                    block = InvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        act=act)
                else:
                    block = InvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        act=act)
                block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx)

        self._block_list = nn.ModuleList(block_list)

    def _update_out_channels(self, channel, feature_idx):
        if feature_idx in self.feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        # c2 stage
        y = self._conv1(inputs)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if (i + 2) in self.feature_maps:
                outs.append(y)
        # outs should be [c3, c4, c5] if feature_maps=[4,11,14]
        return outs


class PicoBackbone(nn.Module):
    """
    A thin wrapper around ESNet that simply returns
    three feature maps: c3, c4, c5 (like DarkNet did).
    """
    def __init__(self, scale=0.5):
        super().__init__()
        # Allow even smaller scale values like 0.25 or 0.375
        self.backbone = ESNet(
            scale=scale,
            act="hard_swish",
            feature_maps=[4, 11, 14],  # produce 3 outputs
            channel_ratio=[0.75]*13    # reduce channels within blocks
        )

    def forward(self, x):
        # returns [c3, c4, c5] each a Tensor
        c3, c4, c5 = self.backbone(x)
        return c3, c4, c5


# -------------------------------------------------------------------------
#   Unchanged FPN, DFL, Head, YOLO definitions, except
#   YOLO now uses PicoBackbone instead of DarkNet.
# -------------------------------------------------------------------------

class DarkFPN(nn.Module):
    """Same as your original FPN (unchanged)."""
    def __init__(self, width, depth):
        super().__init__()
        self.up = nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(nn.Module):
    """Same as original."""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False).to('cpu')
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(nn.Module):
    """Same as original."""
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = nn.ModuleList(nn.Sequential(Conv(x, c1, 3),
                                               Conv(c1, c1, 3),
                                               nn.Conv2d(c1, self.nc, 1))
                                 for x in filters)
        self.box = nn.ModuleList(nn.Sequential(Conv(x, c2, 3),
                                               Conv(c2, c2, 3),
                                               nn.Conv2d(c2, 4 * self.ch, 1))
                                 for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)

        if self.training:
            return x

        self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(nn.Module):
    """
    Modified YOLO wrapper:
      - Replaces DarkNet with PicoBackbone
      - Everything else remains identical
    """
    def __init__(self, width, depth, num_classes):
        super().__init__()
        # REPLACED: self.net = DarkNet(width, depth)
        # WITH:
        self.net = PicoBackbone(scale=0.5)  # scale=0.5 => c3=64, c4=128, c5=256

        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        # compute stride from a dummy forward
        self.head.stride = torch.tensor([
            256 / out.shape[-2] for out in self.forward(img_dummy)
        ])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        # net returns (c3, c4, c5)
        x = self.net(x)
        x = self.fpn(x)  # same FPN as before
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


# -------------------------------------------------------------------------
#  Model factory functions (unchanged except they 
#  now rely on the new YOLO that uses PicoBackbone).
# -------------------------------------------------------------------------

def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)


def yolo_v8_p(num_classes: int = 80):
    """
    Pico variant - extremely small model
    With scale=0.25: c3=32, c4=64, c5=128 channels
    """
    depth = [1, 1, 1]  # minimum depth
    
    # Create model instance and properly initialize nn.Module
    model = YOLO.__new__(YOLO)
    nn.Module.__init__(model)  # This is the key fix
    
    # Create backbone first to get channel dimensions
    backbone = PicoBackbone(scale=0.25)
    img_dummy = torch.zeros(1, 3, 256, 256)
    c3, c4, c5 = backbone(img_dummy)
    channels = [c3.shape[1], c4.shape[1], c5.shape[1]]
    
    # Create width array with actual backbone channels
    width = [3, 8, 16] + channels  # [in_ch, stage1, stage2, c3, c4, c5]
    
    # Assign all components
    model.net = backbone
    model.fpn = DarkFPN(width, depth)
    model.head = Head(num_classes, channels)
    
    # Compute strides
    outs = model.forward(img_dummy)
    model.head.stride = torch.tensor([
        256 / out.shape[-2] for out in outs
    ])
    model.stride = model.head.stride
    model.head.initialize_biases()
    
    return model
