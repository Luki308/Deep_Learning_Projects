import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import numpy as np

class DDIM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers for inference (no grad)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward(self, x0, t):
        noise = torch.randn_like(x0)
        x_t = (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 +
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )
        pred_noise = self.model(x_t, t)
        return nn.functional.l1_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape, eta=0.0, steps=50):
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        time_range = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long).to(device)

        for i, t in enumerate(time_range):
            t = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            alpha_t = self.alphas_cumprod[t][:, None, None, None]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            pred_noise = self.model(x, t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            if i == steps - 1:
                x = pred_x0
            else:
                next_t = torch.full_like(t, int(time_range[i + 1].item()))
                alpha_next = self.alphas_cumprod[next_t][:, None, None, None]
                sigma = eta * torch.sqrt((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t))
                noise = torch.randn_like(x) if eta > 0 else 0
                x = (
                    torch.sqrt(alpha_next) * pred_x0 +
                    torch.sqrt(1 - alpha_next - sigma ** 2) * pred_noise + sigma * noise
                )
        return x
