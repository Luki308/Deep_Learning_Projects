import numpy as np
import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import math


# Sinusoidal position embedding for timestep encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.pad(emb, [[0,0],[0,1]])
        assert emb.shape == (x.shape[0], self.dim), f"{x.shape} != {self.dim}"

        return emb

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x=self.conv(x)
        assert x.shape == (B, C, H // 2, W // 2)
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = nn.functional.interpolate(x, size=None, mode='nearest', scale_factor=2)
        x = self.conv(x)
        assert x.shape == (B, C, H * 2, W * 2)
        return x

class Nin(nn.Module):
    def __init__(self, in_dim, out_dim, scale=1e-10):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        # Works for [B, C, H, W] or [B, HW, C]
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
            x = x @ self.W + self.b  # [B, HW, out_dim]
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, out_dim, H, W]
            return x
        elif x.dim() == 3:  # [B, HW, C]
            return x @ self.W + self.b
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate = 0.1 ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.lin1 = nn.Linear(512, out_dim)
        if not (in_dim == out_dim):
            self.nin = Nin(in_dim, out_dim)
        self.dropout_rate = dropout_rate
        self.nonlin = nn.SiLU()
    def forward(self, x, temb):
        h = self.nonlin(nn.functional.group_norm(x, num_groups=1))
        h = self.conv1(h)

        h+=self.lin1(self.nonlin(temb))[:,:,None,None]
        h = self.nonlin(self.nonlin(nn.functional.group_norm(h, num_groups=1)))
        h = nn.functional.dropout(h, p=self.dropout_rate)
        h = self.conv2(h)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)

        assert x.shape == h.shape
        return x+h

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Q = Nin(dim, dim)
        self.K = Nin(dim, dim)
        self.V = Nin(dim, dim)
        self.dim = dim
        self.nin = Nin(dim, dim, scale=0.1)

    def forward_0(self, x):
        B, C, H, W = x.shape
        assert C ==self.dim

        h = nn.functional.group_norm(x, num_groups=1)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)

        w = torch.einsum('bchw,bcHw->bhwHW', q, k)*(int(C)**(-0.5))     #[B,H,W,H,W]
        w=torch.reshape(w, [B,H,W,H*W])
        w = nn.functional.softmax(w, dim=-1)
        w=torch.reshape(w, [B,H,W,H,W])

        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h=self.nin(h)

        assert h.shape==x.shape
        return x+h

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim

        h = nn.functional.group_norm(x, num_groups=min(32, C))

        # Reshape to sequence
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        q = self.Q(h)  # [B, HW, C]
        k = self.K(h)
        v = self.V(h)

        scale = self.dim ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, HW, HW]
        attn = attn.softmax(dim=-1)

        h = torch.bmm(attn, v)  # [B, HW, C]
        h = h.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        h = self.nin(h)
        return x + h

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResNetBlock(in_channels, base_channels)
        self.enc2 = ResNetBlock(base_channels, base_channels)
        self.down1 = Downsample(base_channels)

        self.enc3 = ResNetBlock(base_channels, base_channels * 2)
        self.enc4 = ResNetBlock(base_channels * 2, base_channels * 2)
        self.down2 = Downsample(base_channels * 2)

        # Bottleneck
        self.mid1 = ResNetBlock(base_channels * 2, base_channels * 4)
        self.attn = AttentionBlock(base_channels * 4)
        self.mid2 = ResNetBlock(base_channels * 4, base_channels * 2)

        # Decoder
        self.up1 = Upsample(base_channels * 2)
        self.dec1 = ResNetBlock(base_channels * 4, base_channels * 2)
        self.dec2 = ResNetBlock(base_channels * 2, base_channels)

        self.up2 = Upsample(base_channels)
        self.dec3 = ResNetBlock(base_channels * 2, base_channels)
        self.dec4 = ResNetBlock(base_channels, base_channels)

        # Final
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        temb = self.time_mlp(t)

        # Encoder
        x1 = self.enc1(x, temb)
        x2 = self.enc2(x1, temb)
        x3 = self.down1(x2)

        x4 = self.enc3(x3, temb)
        x5 = self.enc4(x4, temb)
        x6 = self.down2(x5)

        # Bottleneck
        x7 = self.mid1(x6, temb)
        x7 = self.attn(x7)
        x7 = self.mid2(x7, temb)

        # Decoder
        x8 = self.up1(x7)
        x8 = torch.cat([x5, x8], dim=1)
        x9 = self.dec1(x8, temb)
        x10 = self.dec2(x9, temb)

        x11 = self.up2(x10)
        x11 = torch.cat([x2, x11], dim=1)
        x12 = self.dec3(x11, temb)
        x13 = self.dec4(x12, temb)

        return self.out(x13)

def main():
    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 200, (4,))

    out = model(x, t)
    print(out.shape)  # Should be [4, 3, 64, 64]

if __name__ == '__main__':
    main()