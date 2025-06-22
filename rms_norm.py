import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # gain parameter

    def _norm(self, x):
        # x: (..., dim)
        # RMS: sqrt(mean(x^2) + eps)
        # Output: x / RMS
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # We assume x comes in as (batch_size, ..., features)
        # And normalization is over the last 'features' dimension
        original_dtype = x.dtype
        # Calculate norm in float32 for stability
        x_normed = self._norm(x.float()).to(original_dtype)
        return x_normed * self.weight
