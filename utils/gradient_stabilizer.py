import torch
import numpy as np

class AdaptiveScaler(torch.cuda.amp.GradScaler):
    """Dynamic scaler that adapts based on gradient statistics"""
    def __init__(self):
        super().__init__(init_scale=2.**10, growth_factor=1.1, backoff_factor=0.8)
        self.adaptive_factor = 1.0
        
    def update(self):
        super().update()
        # Dynamically adjust scale based on gradient magnitudes
        current_scale = self.get_scale()
        if any(torch.isinf(g).any() for g in self._per_optimizer_states.values()):
            self._scale = torch.full_like(self._scale, current_scale * 0.5)

class GradientStabilizer:
    """Advanced gradient stabilization with layer-specific constraints"""
    def __init__(self, model, initial_max_grad_value=1.0, initial_max_grad_norm=0.5):
        self.model = model
        self.layer_constraints = {
            'backbone': {'max_value': 2.0, 'max_norm': 1.0},
            'head': {'max_value': 0.5, 'max_norm': 0.3}
        }
        self.global_max_value = initial_max_grad_value
        self.global_max_norm = initial_max_grad_norm
        self.explosion_count = 0

    def clip_gradients(self):
        """Layer-aware gradient clipping"""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            # Get layer-specific constraints
            layer_type = 'backbone' if 'backbone' in name else 'head' if 'head' in name else 'other'
            constraints = self.layer_constraints.get(layer_type, {})
            
            # Value-based clipping
            max_val = constraints.get('max_value', self.global_max_value)
            param.grad.data.clamp_(-max_val, max_val)
            
            # Norm-based clipping
            max_norm = constraints.get('max_norm', self.global_max_norm)
            param_norm = param.grad.norm(2)
            clip_coef = max_norm / (param_norm + 1e-6)
            param.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, 1.0))

    def emergency_gradient_reset(self, threshold=1e4):
        """Handle extreme gradient values"""
        reset_count = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                param.grad.data.zero_()
                reset_count += 1
            elif param.grad.abs().max() > threshold:
                param.grad.data.clamp_(-threshold, threshold)
                reset_count += 1
                
        if reset_count > 0:
            self.explosion_count += 1
            print(f"Emergency reset affected {reset_count} parameters")

def create_optimizer_with_param_groups(model, base_lr=0.001, weight_decay=0.0001):
    """Create optimizer with separate parameter groups for different layers"""
    params = {
        'backbone': [],
        'head': [],
        'norm': [],
        'bias': []
    }
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name:
            params['backbone'].append(p)
        elif 'head' in name:
            params['head'].append(p)
        elif 'norm' in name or 'bn' in name:
            params['norm'].append(p)
        elif 'bias' in name:
            params['bias'].append(p)
    
    optimizer_groups = [
        {'params': params['backbone'], 'lr': base_lr * 0.1, 'weight_decay': weight_decay},
        {'params': params['head'], 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': params['norm'], 'lr': base_lr * 0.01, 'weight_decay': 0.0},
        {'params': params['bias'], 'lr': base_lr * 2.0, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(optimizer_groups)

class StabilizedConv(torch.nn.Module):
    """Conv block with built-in gradient scaling"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.act = torch.nn.ReLU(inplace=True)
        self.grad_scale = 0.2  # Reduced gradient flow

    def forward(self, x):
        x = self.conv(x * self.grad_scale)
        x = self.norm(x)
        return self.act(x)
