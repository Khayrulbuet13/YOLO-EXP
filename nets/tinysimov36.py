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
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class DarkNet(torch.nn.Module):
    """
    A 'backbone' that combines Conv and downsampling in the last layer of each stage.
    """
    def __init__(self, widths=None, depths=None):
        super().__init__()
        layers = []
        in_ch = widths[0]
        stage_count = min(len(depths), len(widths) - 1)

        for i in range(stage_count):
            out_ch = widths[i + 1]
            nblocks = depths[i]

            for j in range(nblocks):
                # Use stride=4 for first block, stride=2 for second
                if i == 0 and j == 0:
                    stride = 4  # Aggressive downsampling
                else:
                    stride = 2 if (j == nblocks - 1) else 1
                
                layers.append(Conv(in_ch, out_ch, 3, stride))
                in_ch = out_ch

        self.backbone = torch.nn.ModuleList(layers)
        self.out_channels = in_ch

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=20, ch_in=128, head_ch=12, dfl_ch=8):
        """
        nc: number of classes
        ch_in: number of input channels for the first Conv in the head
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.dfl_ch = dfl_ch
        self.no = nc + self.dfl_ch * 4  # number of outputs per anchor
        self.nl = 1  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.conv = Conv(ch_in, head_ch, 1, 1)
        self.dfl = DFL(self.dfl_ch)

        # Detection head with correct output channels
        self.cls = torch.nn.Conv2d(head_ch, self.nc, 1)
        self.box = torch.nn.Conv2d(head_ch, 4 * self.dfl_ch, 1)  # 4 * 16 = 64 channels for DFL

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            return [torch.cat((self.box(x), self.cls(x)), 1)]

        # For inference
        anchor_points, stride_tensor = make_anchors([x], self.stride, 0.5)
        self.anchors, self.strides = anchor_points.transpose(0, 1), stride_tensor.transpose(0, 1)
        
        x = torch.cat([self.box(x), self.cls(x)], 1)
        x = x.view(x.shape[0], self.no, -1)
        box, cls = x.split((self.dfl_ch * 4, self.nc), 1)

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
    def __init__(self, widths=None, depths=None, num_classes=20, img_size=(256, 256),
                 head_ch=12, dfl_ch=8):
        """
        If widths/depths are None, use the original architecture (DarkNet hard-coded).
        Otherwise, build custom backbone from widths/depths.
        
        Args:
            widths: List of channel widths for each stage
            depths: List of block depths for each stage
            num_classes: Number of classes to detect
            img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
        """
        super().__init__()
        self.net = DarkNet(widths, depths)
        self.head = Head(num_classes, self.net.out_channels, head_ch, dfl_ch)

        # Handle img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        h, w = img_size

        # Initialize strides based on feature map dimensions
        original_mode = self.training
        self.eval()
        img_dummy = torch.zeros(1, 3, h, w)
        with torch.no_grad():
            darknet_out = self.net(img_dummy)
            feature_h = darknet_out.shape[2]
            feature_w = darknet_out.shape[3]
            stride_h = h / feature_h
            stride_w = w / feature_w
            if not math.isclose(stride_h, stride_w, rel_tol=1e-5):
                print(f"Warning: Stride mismatch: h={stride_h}, w={stride_w}. Using average.")
            self.head.stride = torch.tensor([stride_h])
        self.train(original_mode)

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
def yolo_v8_xl(num_classes: int = 20, img_size=(256, 256)):
    """
    This returns the original YOLO model exactly as before (the "XL" version).
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    widths =  [3, 16, 32, 64, 128]
    depths = [3, 2, 3, 2]
    return YOLO(widths, depths, num_classes, img_size=img_size)


# -----------------------------------------------------------------------------------
# Example custom constructors for YOLOv8 "S", "M", and "L" with widths/depths arrays
# You can tweak these arrays any way you like.
# -----------------------------------------------------------------------------------
def yolo_v8_es(num_classes: int = 20, img_size=(256, 256)):
    """
    Small version: fewer channels, fewer repeated blocks
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 4,  16]
    depths = [1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size=img_size)

def yolo_v8_s(num_classes: int = 20, img_size=(256, 256)):
    """
    Small version: fewer channels, fewer repeated blocks
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 4, 8,  16,  64, 128]
    depths = [1, 1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size=img_size)

def yolo_v8_sn(num_classes: int = 20, img_size=(256, 256)):
    """
    Small version: fewer channels, fewer repeated blocks
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 8,  16,  64, 128]
    depths = [1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size=img_size)

def yolo_v8_es(num_classes: int = 20, img_size=(256, 256)):
    """
    Extra Small version: fewer channels, fewer repeated blocks
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 1,2,2 repeated conv blocks
    # widths = [3(input), 32, 64, 128, 256]
    # depths = [1, 2, 2]
    widths = [3, 4, 8, 16, 32, 64]
    depths = [1, 1, 1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size=img_size)

def yolo_v8_m(num_classes: int = 20, img_size=(256, 256)):
    """
    Medium version
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 2,4,4 repeated conv blocks
    # widths = [3, 48, 96, 192, 384]
    # depths = [2, 4, 4]
    widths = [3, 16, 32, 128, 256]
    depths = [1, 1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size=img_size)


def yolo_v8_l(num_classes: int = 20, img_size=(256, 256)):
    """
    Large version
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 3,6,6 repeated conv blocks
    # widths = [3, 64, 128, 256, 512, 512]
    # Note that we gave one extra channel stage here to illustrate flexibility.
    # You can adapt as needed.
    widths = [3, 64, 128, 256, 512, 512]
    depths = [3, 6, 6]
    return YOLO(widths, depths, num_classes, img_size=img_size)

def yolo_v8_bn(num_classes: int = 20, img_size=(256, 256)):
    """
    Bottleneck version
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
    """
    # Example: 3 stages, with 3,6,6 repeated conv blocks
    # widths = [3, 64, 128, 256, 512, 512]
    # Note that we gave one extra channel stage here to illustrate flexibility.
    # You can adapt as needed.
    widths = [3, 64, 128]
    depths = [3, 6, 6]
    return YOLO(widths, depths, num_classes, img_size=img_size)

# Ultra-Efficient FPGA Model
def yolo_v8_fpga_light(num_classes: int = 20, img_size=(128, 256)):
    """
    Optimized for FPGA with:
    - Only 2 blocks
    - Stride-4 first convolution
    - Reduced head and DFL channels
    - Consistent 3x3 kernels
    """
    widths = [3,    # Input
              16,   # Stage 1 (stride-4)
              32]   # Stage 2 (stride-2)
    
    depths = [1, 1]  # Only 2 blocks total
    
    return YOLO(
        widths, 
        depths, 
        num_classes, 
        img_size=img_size,
        head_ch=8,     # Reduced head channels
        dfl_ch=6       # Reduced DFL channels
    )
    
def yolo_v8_fpga_mid(num_classes: int = 20, img_size=(128, 256)):
    """
    Optimized for FPGA with:
    - Only 2 blocks
    - Stride-4 first convolution
    - Reduced head and DFL channels
    - Consistent 3x3 kernels
    """
    widths = [3,    # Input
              16,   # Stage 1 (stride-4)
              64]   # Stage 2 (stride-2)
    
    depths = [1, 1]  # Only 2 blocks total
    
    return YOLO(
        widths, 
        depths, 
        num_classes, 
        img_size=img_size,
        head_ch=8,     # Reduced head channels
        dfl_ch=6       # Reduced DFL channels
    )