import math
import torch
from utils.util import make_anchors

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
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

class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class DarkNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Exactly matching the YAML architecture
        self.backbone = torch.nn.ModuleList([
            Conv(3, 16, 3, 1),  # Conv1
            torch.nn.MaxPool2d(2, 2),  # MaxPool1
            Conv(16, 16, 3, 1),  # Conv2
            Conv(16, 16, 3, 1),  # Conv3
            torch.nn.MaxPool2d(2, 2),  # MaxPool2
            Conv(16, 32, 3, 1),  # Conv4
            Conv(32, 32, 3, 1),  # Conv5
            torch.nn.MaxPool2d(2, 2),  # MaxPool3
            Conv(32, 64, 3, 1),  # Conv6
            Conv(64, 64, 3, 1),  # Conv7
            torch.nn.MaxPool2d(2, 2),  # MaxPool4
            Conv(64, 64, 3, 1),  # Conv8
            Conv(64, 128, 3, 1),  # Conv9
            torch.nn.MaxPool2d(2, 2),  # MaxPool5
            Conv(128, 128, 3, 1)  # Conv10
        ])

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=20):
        super().__init__()
        self.nc = nc  # number of classes
        self.ch = 16  # DFL channels
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.nl = 1  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # As per YAML: Conv(24, 1, 1) followed by Detect
        self.conv = Conv(128, 24, 1, 1)
        self.dfl = DFL(self.ch)
        
        # Detection head with correct output channels
        self.cls = torch.nn.Conv2d(24, self.nc, 1)
        self.box = torch.nn.Conv2d(24, 4 * self.ch, 1)  # 4 * 16 = 64 channels for DFL

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            return [torch.cat((self.box(x), self.cls(x)), 1)]
            
        # For inference
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([x], self.stride, 0.5))
        
        x = torch.cat([self.box(x), self.cls(x)], 1)
        x = x.view(x.shape[0], self.no, -1)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        
        dfl_out = self.dfl(box)
        a, b = torch.split(dfl_out, 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        m = self
        a = self.box
        b = self.cls
        s = self.stride[0]
        a.bias.data[:] = 1.0
        b.bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = DarkNet()
        self.head = Head(num_classes)
        
        # Initialize strides
        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in [self.forward(img_dummy)[0]]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        return self.head(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8(num_classes: int = 20):
    return YOLO(num_classes)
