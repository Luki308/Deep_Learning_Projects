import torch
import intel_extension_for_pytorch as ipex
torch.manual_seed(280) # 0, 84, 85, 281, 333, 282, 280
import os
from torchvision.utils import save_image

from Project_3.src.model.UNet import UNet
from Project_3.src.model.ddim import DDIM

folder= '2025-06-02_23-01'
steps = 50
CHECKPOINT = f"checkpoints/{folder}/ddim_final.pth"
SAVE_PATH = f"checkpoints/{folder}/ddim_sample_s{steps}.png"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
SAMPLE_SHAPE = (16, 3, 64, 64)


def interpolate_latents(z1, z2, steps):
    return [(1 - t) * z1 + t * z2 for t in torch.linspace(0, 1, steps)]


def run_interpolation(ddim, out_dir='interpolation_results', steps_int=10, image_size=64, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)

    z1 = torch.randn(1, 3, image_size, image_size).to(device)
    z2 = torch.randn(1, 3, image_size, image_size).to(device)

    z_interp = interpolate_latents(z1, z2, steps_int)

    for idx, z in enumerate(z_interp):
        with torch.no_grad():
            img = ddim.sample(noise=z, steps=steps, shape=(1, 3, 64, 64))
            img = ((img + 1) / 2).clamp(0, 1)
            save_image(img, os.path.join(out_dir, f"interp_{idx:02d}.png"))


if __name__ == "__main__":
    unet = UNet(in_channels=3, out_channels=3).to(DEVICE)
    ddim = DDIM(unet, timesteps=1000).to(DEVICE)
    ddim.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    ddim.eval().to(DEVICE)

    run_interpolation(ddim, out_dir=f'interpolation_results/s{steps}/{folder}_mse_6', image_size=64, device=DEVICE)
