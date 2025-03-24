import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from utils.util import make_anchors



class SAMYOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes, img_size=(1024, 1024)):
        super().__init__()
        # Validate input size
        if img_size != (1024, 1024):
            print(f"Warning: SAM works best with 1024x1024 inputs. Using {img_size} may reduce performance")
            
        # Load SAM encoder (native 1024x1024)
        sam_checkpoint = "sam_checkpoint/sam_vit_b_01ec64.pth"
        sam_model = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        self.sam_encoder = sam_model.image_encoder
        
        # Freeze SAM encoder
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # SAM outputs [B, 256, 64, 64] for 1024x1024 input
        # Build FPN for multi-scale features
        self.fpn = SAMFPN(
            in_channels=256,
            out_channels=[width[3], width[4], width[5]],  # [128, 256, 512] for v8_s
            depth=depth[1]  # Use the FPN-specific depth from config
        )

        # Detection heads
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self._initialize_heads()

    def _initialize_heads(self):
        """Auto-configure strides based on SAM's native 1024x1024 processing"""
        # SAM's base stride is 16 (1024/64)
        strides = [16 * (2**i) for i in range(3)]  # [16, 32, 64]
        self.head.stride = torch.tensor(strides)
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        # SAM preprocessing (normalization)
        x = (x - torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)) \
            / torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        
        # Match SAM encoder dtype
        x = x.to(next(self.sam_encoder.parameters()).dtype)
        
        # Extract features (output shape: [B, 256, 64, 64])
        features = self.sam_encoder(x)
        
        # FPN processing
        p3, p4, p5 = self.fpn(features)  # [1/8, 1/16, 1/32] scales
        
        # Detection heads
        return self.head([p3, p4, p5])

# class SAMFPN(torch.nn.Module):
#     """FPN designed for SAM's 64x64 output (1024x1024 input)"""
#     def __init__(self, in_channels=256, out_channels=[128, 256, 512], depth: int = 2):
#         super().__init__()
        
#         # Top-down path
#         self.p5 = CSP(in_channels, out_channels[2], depth)
#         self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        
#         self.p4 = CSP(out_channels[2] + in_channels, out_channels[1], depth)
#         self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        
#         self.p3 = CSP(out_channels[1] + in_channels, out_channels[0], depth)
        
#         # Bottom-up path
#         self.down3 = Conv(out_channels[0], out_channels[0], 3, 2)
#         self.p4_out = CSP(out_channels[0] + out_channels[1], out_channels[1], depth)
        
#         self.down4 = Conv(out_channels[1], out_channels[1], 3, 2)
#         self.p5_out = CSP(out_channels[1] + out_channels[2], out_channels[2], depth)

#     def forward(self, x):
#         # Top-down
#         p5 = self.p5(x)          # [B, 512, 64, 64]
#         p4 = self.p4(torch.cat([self.up5(p5), x], 1))  # [B, 256, 128, 128]
#         p3 = self.p3(torch.cat([self.up4(p4), x], 1))  # [B, 128, 256, 256]
        
#         # Bottom-up
#         p4 = self.p4_out(torch.cat([self.down3(p3), p4], 1))  # [B, 256, 128, 128]
#         p5 = self.p5_out(torch.cat([self.down4(p4), p5], 1))  # [B, 512, 64, 64]
        
#         return p3, p4, p5



class SAMFPN(torch.nn.Module):
    """Fixed FPN for SAM's 64x64 features"""
    def __init__(self, in_channels=256, out_channels=[128, 256, 512], depth=2):
        super().__init__()
        # Top-down path
        self.p5 = CSP(in_channels, out_channels[2], depth)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Add projection layers for feature alignment
        self.proj4 = Conv(in_channels, out_channels[1], 1)
        self.proj3 = Conv(in_channels, out_channels[0], 1)
        
        # Modified CSP blocks with proper channel counts
        self.p4 = CSP(out_channels[2] + out_channels[1], out_channels[1], depth)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.p3 = CSP(out_channels[1] + out_channels[0], out_channels[0], depth)

    def forward(self, x):
        # Input x: [B, 256, 64, 64]
        
        # P5 branch
        p5 = self.p5(x)          # [B, 512, 64, 64]
        
        # P4 branch
        p5_up = self.up5(p5)     # [B, 512, 128, 128]
        x_proj4 = self.proj4(F.interpolate(x, scale_factor=2, mode='bilinear'))  # [B, 256, 128, 128]
        p4 = self.p4(torch.cat([p5_up, x_proj4], 1))  # [B, 512+256=768 → 256]
        
        # P3 branch
        p4_up = self.up4(p4)     # [B, 256, 256, 256]
        x_proj3 = self.proj3(F.interpolate(x, scale_factor=4, mode='bilinear'))  # [B, 128, 256, 256]
        p3 = self.p3(torch.cat([p4_up, x_proj3], 1))  # [B, 256+128=384 → 128]
        
        return p3, p4, p5

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

class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3), Conv(ch, ch, 3))
    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)

class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))
    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DFL(torch.nn.Module):
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

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        # Increased head capacity
        c1 = max(filters[0] * 2, self.nc)
        c2 = max((filters[0] // 2, self.ch * 4))
        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c1, 3), Conv(c1, c1, 3), Conv(c1, c1, 3),
            torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c2, 3), Conv(c2, c2, 3), Conv(c2, c2, 3),
            torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)
    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)
    def initialize_biases(self):
        for a, b, s in zip(self.box, self.cls, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


def sam_yolo_v8_s(num_classes: int = 80, img_size=(1024, 1024)):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return SAMYOLO(width, depth, num_classes, img_size)

def sam_yolo_v8_m(num_classes: int = 80, img_size=(1024, 1024)):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return SAMYOLO(width, depth, num_classes, img_size)







import torch
import torch.nn as nn

def flatten_model(model):
    """
    Flattens a model into a Sequential-like list of layers,
    handling custom blocks such as CSP, SPP, and Residuals recursively.
    """
    modules = []

    for name, layer in model.named_children():
        if isinstance(layer, (nn.Sequential, nn.ModuleList)):
            # Recursively flatten Sequential and ModuleList
            modules.extend(flatten_model(layer))
        elif isinstance(layer, nn.Conv2d):
            modules.append(layer)
            # Check for common attached submodules
            if hasattr(layer, 'norm'):
                modules.append(layer.norm)
            if hasattr(layer, 'relu'):
                modules.append(layer.relu)
        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.SiLU, nn.MaxPool2d, nn.LayerNorm, nn.GELU, nn.Upsample)):
            modules.append(layer)
        elif isinstance(layer, nn.Module):
            # For any custom or other nn.Module, recursively process its children
            if list(layer.children()):
                modules.extend(flatten_model(layer))
            else:
                modules.append(layer)
        else:
            # For any unknown types, just add directly
            modules.append(layer)

    return modules