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
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


# class DarkNet(torch.nn.Module):
#     """
#     A 'backbone' that can be hard-coded (original YOLO code)
#     OR built from `widths` and `depths` if provided.
#     """
#     def __init__(self, widths=None, depths=None):
#         super().__init__()


#         layers = []
#         in_ch = widths[0]  # typically 3 for RGB
#         stage_count = min(len(depths), len(widths) - 1)

#         for i in range(stage_count):
#             out_ch = widths[i + 1]
#             nblocks = depths[i]
#             # repeat nblocks times: Conv(in_ch -> out_ch)
#             for _ in range(nblocks):
#                 layers.append(Conv(in_ch, out_ch, 3, 1))
#                 in_ch = out_ch
#             # Add a downsample
#             layers.append(torch.nn.MaxPool2d(2, 2))

#         self.backbone = torch.nn.ModuleList(layers)
#         self.out_channels = in_ch  # last conv out_ch

#     def forward(self, x):
#         for layer in self.backbone:
#             x = layer(x)
#         return x

# class DarkNet(torch.nn.Module):
#     """
#     A 'backbone' that can be hard-coded (original YOLO code)
#     OR built from `widths` and `depths` if provided.
#     """
#     def __init__(self, widths=None, depths=None):
#         super().__init__()

#         layers = []
#         in_ch = widths[0]  # typically 3 for RGB
#         stage_count = min(len(depths), len(widths) - 1)

#         for i in range(stage_count):
#             out_ch = widths[i + 1]
#             nblocks = depths[i]

#             # Apply nblocks-1 Conv layers with stride=1
#             for j in range(nblocks - 1):
#                 layers.append(Conv(in_ch, out_ch, 3, 1))  # Keep stride=1 for the blocks
#                 in_ch = out_ch  # Update in_ch for the next block

#             # After the last block, apply downsampling (stride=2)
#             if i < stage_count - 1:  # Only apply downsampling between stages
#                 layers.append(Conv(in_ch, out_ch, 3, 2))  # stride=2 for downsampling
#                 in_ch = out_ch  # Update in_ch after downsampling

#         self.backbone = torch.nn.ModuleList(layers)
#         self.out_channels = in_ch  # last conv out_ch

#     def forward(self, x):
#         for layer in self.backbone:
#             x = layer(x)
#         return x


class DarkNet(torch.nn.Module):
    """
    A 'backbone' that combines Conv and downsampling in the last layer of each stage.
    """
    def __init__(self, widths=None, depths=None):
        super().__init__()

        layers = []
        in_ch = widths[0]  # typically 3 for RGB
        stage_count = min(len(depths), len(widths) - 1)

        for i in range(stage_count):
            out_ch = widths[i + 1]
            nblocks = depths[i]

            # Add all nblocks Conv layers for the current stage
            for j in range(nblocks):
                # Use stride=2 only for the last Conv of the stage if it's not the final stage
                stride = 2 if (j == nblocks - 1 and i < stage_count - 1) else 1
                layers.append(Conv(in_ch, out_ch, 3, stride))
                in_ch = out_ch  # Update input channels for next layer

        self.backbone = torch.nn.ModuleList(layers)
        self.out_channels = in_ch  # last conv out_ch

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=20, ch_in=128):
        """
        nc: number of classes
        ch_in: number of input channels for the first Conv in the head
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.ch = 16  # DFL channels
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.nl = 1  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.conv = Conv(ch_in, 24, 1, 1)  # changed 128 -> ch_in
        self.dfl = DFL(self.ch)

        # Detection head with correct output channels
        self.cls = torch.nn.Conv2d(24, self.nc, 1)
        self.box = torch.nn.Conv2d(24, 4 * self.ch, 1)  # 4 * 16 = 64 channels for DFL

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            return [torch.cat((self.box(x), self.cls(x)), 1)]

        # For inference
        self.anchors, self.strides = (y.transpose(0, 1) for y in make_anchors([x], self.stride, 0.5))
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
    def __init__(self, widths=None, depths=None, num_classes=20):
        """
        If widths/depths are None, use the original architecture (DarkNet hard-coded).
        Otherwise, build custom backbone from widths/depths.
        """
        super().__init__()
        self.net = DarkNet(widths, depths)
        self.head = Head(num_classes, ch_in=self.net.out_channels)

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
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


# -----------------------------------------------------------------------------------
# OLD constructor renamed to "yolo_v8_xl" to reflect that it is your existing "XL" model
# -----------------------------------------------------------------------------------
def yolo_v8_xl(num_classes: int = 20):
    """
    This returns the original YOLO model exactly as before (the "XL" version).
    """
    widths =  [3, 16, 32, 64, 128]
    depths = [3, 2, 3, 2]
    return YOLO(widths, depths, num_classes)


# -----------------------------------------------------------------------------------
# Example custom constructors for YOLOv8 "S", "M", and "L" with widths/depths arrays
# You can tweak these arrays any way you like.
# -----------------------------------------------------------------------------------
def yolo_v8_s(num_classes: int = 20):
    """
    Small version: fewer channels, fewer repeated blocks
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 4, 8, 16,  64, 128]
    depths = [1, 1, 1, 1, 1]
    return YOLO(widths, depths, num_classes)

def yolo_v8_es(num_classes: int = 20):
    """
    Small version: fewer channels, fewer repeated blocks
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 4, 8, 16, 32, 64]
    depths = [1, 1, 1, 1, 1]
    return YOLO(widths, depths, num_classes)

def yolo_v8_m(num_classes: int = 20):
    """
    Medium version
    """
    # Example: 3 stages, with 2,4,4 repeated conv blocks
    # widths = [3, 48, 96, 192, 384]
    # depths = [2, 4, 4]
    widths = [3, 16, 32, 128, 256]
    depths = [1, 1, 1, 1]
    return YOLO(widths, depths, num_classes)


def yolo_v8_l(num_classes: int = 20):
    """
    Large version
    """
    # Example: 3 stages, with 3,6,6 repeated conv blocks
    # widths = [3, 64, 128, 256, 512, 512]
    # Note that we gave one extra channel stage here to illustrate flexibility.
    # You can adapt as needed.
    widths = [3, 64, 128, 256, 512, 512]
    depths = [3, 6, 6]
    return YOLO(widths, depths, num_classes)

