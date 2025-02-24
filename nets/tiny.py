import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
#  Helpers (fusing BN, anchor utils) 
#  (Replace these with your own or keep as-is)
# -------------------------------------------------------------------------
def fuse_conv(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d for inference."""
    fused = nn.Conv2d(conv.in_channels,
                      conv.out_channels,
                      kernel_size=conv.kernel_size,
                      stride=conv.stride,
                      padding=conv.padding,
                      groups=conv.groups,
                      bias=True)
    # Copy weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))
    # Copy bias
    b_conv = torch.zeros(conv.weight.size(0)) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused

def make_anchors(detect_outs, strides, grid_offset=0.5):
    """
    For each scale output:
       shape = [B, (ch), H, W]
       we produce center coords for each cell, e.g. (cx, cy)
    We return (anchors, strides) in list-of-tensors form, each Nx2 for (cx, cy).
    Exactly the same logic you used in your original 'utils.util.make_anchors'.
    """
    # Example implementation:
    anchors, s = [], []
    nl = len(detect_outs)
    for i in range(nl):
        _, ch, ny, nx = detect_outs[i].shape
        sx = strides[i]
        # Generate grid
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='xy')
        grid_xy = (torch.stack((xv, yv), 2).float() + grid_offset)  # shape [ny, nx, 2]
        anchors.append(grid_xy.view(-1, 2))
        s.append(torch.full((ny * nx, 1), sx))
    return anchors, s


# -------------------------------------------------------------------------
#  A simple (Conv+BN+ReLU) block with kernel=1 by default
# -------------------------------------------------------------------------
class Conv1x1(nn.Module):
    """1×1 Conv + BN + SiLU by default. 
       You can adjust 'k=3' if you want some 3×3 in backbone, 
       but 1×1 drastically reduces parameters."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, 
                              padding=p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        """Fuse Conv+BN for inference"""
        fused = fuse_conv(self.conv, self.bn)
        return fused


# -------------------------------------------------------------------------
#  1) Minimal Backbone
#     - 5 strided layers => final stride=32
#     - We take outputs at stride=8,16,32 => p3,p4,p5
# -------------------------------------------------------------------------
class MinimalBackbone(nn.Module):
    """
    Each layer is:
      conv1: (3 -> 8), stride=2 => out size 1/2
      conv2: (8 -> 8), stride=2 => out size 1/4
      conv3: (8 -> 8), stride=2 => out size 1/8 => p3
      conv4: (8 -> 8), stride=2 => out size 1/16 => p4
      conv5: (8 -> 8), stride=2 => out size 1/32 => p5

    p3, p4, p5 have 8 channels each.
    """
    def __init__(self):
        super().__init__()
        # For an even smaller param count, we do kernel=1
        # If you want more spatial mixing, set k=3 and p=1 for the earliest layers.
        self.conv1 = Conv1x1(3, 8, k=1, s=2, p=0)   # stride 2
        self.conv2 = Conv1x1(8, 8, k=1, s=2, p=0)   # stride 4
        self.conv3 = Conv1x1(8, 8, k=1, s=2, p=0)   # stride 8  => p3
        self.conv4 = Conv1x1(8, 8, k=1, s=2, p=0)   # stride 16 => p4
        self.conv5 = Conv1x1(8, 8, k=1, s=2, p=0)   # stride 32 => p5

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)
        return p3, p4, p5


# -------------------------------------------------------------------------
#  2) Tiny FPN
#     - merges p5→p4, p4→p3 with upsampling, all 1×1 conv
#     - minimal channels (8) throughout
# -------------------------------------------------------------------------
class TinyFPN(nn.Module):
    """
    We produce the same three outputs: f2, f4, f6
    corresponding to strides 8,16,32 for the Head.
    Each is 8 channels.
    """
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # p5->8, p4->8 => cat=16 => reduce to 8
        self.m5 = Conv1x1(8, 8, k=1, s=1, p=0)
        self.f5 = Conv1x1(16, 8, k=1, s=1, p=0)

        # f5->8, p3->8 => cat=16 => reduce to 8 => f2 is final p3-level
        self.m4 = Conv1x1(8, 8, k=1, s=1, p=0)
        self.f4 = Conv1x1(16, 8, k=1, s=1, p=0)

        # downward path again for p4-level
        self.down4 = Conv1x1(8, 8, k=1, s=2, p=0)   # f2-> => stride 16 => cat with f5 => 16-> f4
        self.f4_2 = Conv1x1(16, 8, k=1, s=1, p=0)

        # downward path for p5-level
        self.down5 = Conv1x1(8, 8, k=1, s=2, p=0)   # f4-> => stride 32 => cat with p5 => 16-> f6
        self.f6 = Conv1x1(16, 8, k=1, s=1, p=0)

    def forward(self, x):
        # x = (p3, p4, p5)
        p3, p4, p5 = x

        # Up path
        p5_up  = self.up(self.m5(p5))     # 8ch => up to stride16
        f5     = self.f5(torch.cat([p4, p5_up], dim=1))  # => 8ch

        f5_up  = self.up(self.m4(f5))     # => up to stride8
        f2     = self.f4(torch.cat([p3, f5_up], dim=1))  # => 8ch

        # Down path for final p4-level
        f2_down = self.down4(f2)         # => stride16
        f4_out  = self.f4_2(torch.cat([f2_down, f5], dim=1))  # => 8ch

        # Down path for final p5-level
        f4_down = self.down5(f4_out)     # => stride32
        f6      = self.f6(torch.cat([f4_down, p5], dim=1))    # => 8ch

        # Output: (stride8, stride16, stride32)
        return (f2, f4_out, f6)


# -------------------------------------------------------------------------
#  3) Distribution Focal Loss "DFL" (same as your code, preserving ch=16)
# -------------------------------------------------------------------------
class DFL(nn.Module):
    """
    Exactly the same logic as your original code,
    but we keep ch=16 so that final box dimension= 4*16 => 64.
    """
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        # This 1×1 conv is used to convert the DFL channels to continuous offsets
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=False)
        # Initialize it so that weight = [0,1,2,3,...ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(x)

    def forward(self, x):
        # x shape = [B, 4*ch, H*W]
        # => reshape -> softmax -> "integral" -> return shape [B,4,H*W]
        b, c, a = x.shape  # c=4*ch
        x = x.view(b, 4, self.ch, a).transpose(2, 1)  # => [b,ch,4,a]
        # softmax over ch dimension
        x = x.softmax(dim=1)  # [b,ch,4,a]
        x = self.conv(x)      # [b,1,4,a], weight is shape [1,ch,1,1]
        return x.view(b, 4, a)


# -------------------------------------------------------------------------
#  4) Tiny Head
#     - We keep the same final dimension => no = (nc + 4*dfl_ch)
#       but reduce from 2 intermediate convs -> 1 conv in each sub-branch
#     - The final shape is fully compatible with your existing YOLO
#       detection/loss routines (3 scales, same anchor logic, etc.)
# -------------------------------------------------------------------------
class TinyHead(nn.Module):
    """
    - ch=16 => DFL dimension => 64 box channels
    - We produce shape per scale = [ B, (64 + nc), H, W ]
    - We do it for 3 scales => then anchor creation is the same as your code.
    - We reduce the # of layers in each sub-block:

      BOX path:
        Conv1x1(x_in -> c2= max(x_in//4, 4*16=64))   # but here we override to 1 layer
        Conv1x1( c2 -> 4*dfl_ch=64 )                # final box pred

      CLS path:
        Conv1x1(x_in -> c1= max(x_in, nc=80) )      # at least 80
        Conv1x1( c1 -> nc )                         # final class pred

    - This yields the same final concat shape [B, 64+nc, H, W] per scale.
    """
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, channels_in=(8,8,8), dfl_ch=16):
        super().__init__()
        self.nc = nc
        self.ch = dfl_ch  # DFL channels
        self.nl = len(channels_in)  # 3 detection scales
        self.no = nc + self.ch * 4  # final out channels per scale
        self.stride = torch.zeros(self.nl)  # to be filled later

        # Build box sub-blocks
        self.box_convs = nn.ModuleList()
        for cin in channels_in:
            # c2 = max(cin//4, 4*dfl_ch) 
            # but because cin=8 < 80 < 64 anyway, let's just fix c2=64 if we want to keep the original logic
            c2 = max(cin // 4, self.ch * 4)  # usually 64
            block = nn.Sequential(
                Conv1x1(cin, c2, k=1, s=1, p=0),
                Conv1x1(c2, 4*self.ch, k=1, s=1, p=0)
            )
            self.box_convs.append(block)

        # Build cls sub-blocks
        self.cls_convs = nn.ModuleList()
        for cin in channels_in:
            # c1 = max(cin, nc). For COCO nc=80 => c1=80 if cin <80
            c1 = max(cin, nc)
            block = nn.Sequential(
                Conv1x1(cin, c1, k=1, s=1, p=0),
                Conv1x1(c1, nc,  k=1, s=1, p=0)
            )
            self.cls_convs.append(block)

        # DFL integral
        self.dfl = DFL(self.ch)

    def forward(self, feats):
        """
        feats: list of 3 feature maps => [f2, f4, f6], each [B, 8, H, W]
        Return either training-time shape = same list but each [B,no,H,W],
        or infer-time [B,  (x,y,w,h + cls),  1*Ncells total].
        """
        # 1) Combine the box + cls predictions
        #    Each scale i => shape [B, 64 + nc, H, W]
        outputs = []
        for i in range(self.nl):
            f = feats[i]
            box = self.box_convs[i](f)
            cls = self.cls_convs[i](f)
            out = torch.cat([box, cls], dim=1)  # => [B, 64+nc, H, W]
            outputs.append(out)

        if self.training:
            return outputs  # training code expects raw feature maps

        # 2) Inference mode => apply anchor offsets & DFL integral
        #    (same logic as your original Head.forward)
        # ---------------------------------------------------------
        # create anchor grids
        self.anchors, self.strides = (t.transpose(0,1) for t in make_anchors(outputs, self.stride, 0.5))
        # stack outputs => shape [B, (64+nc), Ncells_total]
        x_cat = torch.cat([o.view(o.shape[0], o.shape[1], -1) for o in outputs], dim=2)
        box, cls = x_cat.split((4*self.ch, self.nc), dim=1)  # => [B,4*dfl_ch,N], [B,nc,N]

        # DFL integral => decode x1,y1,x2,y2
        offsets = self.dfl(box)  # => [B,4,N]
        a, b = torch.split(offsets, 2, dim=1)  # => a=[B,2,N], b=[B,2,N]
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        wh = b - a
        xy = (a + b) * 0.5
        box_out = torch.cat([xy, wh], dim=1)  # => [B,4,N]
        # scale by stride
        box_out = box_out * self.strides  # broadcast each scale's stride

        # finalize => [B, 4+nc, N]
        cls_out = cls.sigmoid()
        return torch.cat([box_out, cls_out], dim=1)

    def initialize_biases(self):
        # Skip bias initialization if not needed
        return

# -------------------------------------------------------------------------
#  5) Putting it all together in a YOLO wrapper
# -------------------------------------------------------------------------
class YOLOTinyFPGA(nn.Module):
    """
    Overall flow:
      1) MinimalBackbone => p3,p4,p5
      2) TinyFPN => f2,f4,f6
      3) TinyHead => final box+cls
    We keep `ch=16` in DFL => output dimension is (80 + 64)=144 if nc=80.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = MinimalBackbone()
        self.fpn      = TinyFPN()
        self.head     = TinyHead(nc=num_classes, channels_in=(8,8,8), dfl_ch=16)

        # Set strides directly (8,16,32)
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)       # [B,8,h3,w3], [B,8,h4,w4], [B,8,h5,w5]
        f2, f4, f6 = self.fpn((p3, p4, p5)) # merges -> [B,8], ...
        return self.head([f2, f4, f6])      # either list of 3 outs (training) or final concat (inference)

    def fuse(self):
        """
        Fuse all Conv+BN for optimized inference
        (remove BN layers, store their effect in conv weights/bias).
        """
        for m in self.modules():
            if isinstance(m, Conv1x1):
                fused = m.fuse()  # returns a plain Conv2d
                # Replace the block
                m.conv = fused
                del m.bn
                # override forward
                def forward_fused(x, conv=fused, act=m.act):
                    return act(conv(x))
                m.forward = forward_fused
        return self


# -------------------------------------------------------------------------
#  Factory function (so it matches your style "def yolo_v8_n(...)")
# -------------------------------------------------------------------------
def yolo_fpga_tiny(num_classes=80):
    """
    Returns our tiny YOLO with ~ <50k parameters,
    same 3-scale output, final dimension = `nc + 64` for each anchor cell.
    """
    return YOLOTinyFPGA(num_classes=num_classes)
