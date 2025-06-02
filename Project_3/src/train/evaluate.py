import os
from datetime import datetime

import torch
import torchvision.utils as vutils


from Project_3.src.model.UNet import UNet
from Project_3.src.model.ddim import DDIM
#
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
# os.makedirs(f"samples/{timestamp}", exist_ok=True)

# Configuration
folder= '2025-06-01_01-14'
steps = 50
CHECKPOINT = f"checkpoints/{folder}/ddim_final.pth"
SAVE_PATH = f"checkpoints/{folder}/ddim_sample_s{steps}.png"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
SAMPLE_SHAPE = (16, 3, 64, 64)

def main():
    # Load model
    unet = UNet(in_channels=3, out_channels=3).to(DEVICE)
    ddim = DDIM(unet, timesteps=1000).to(DEVICE)
    ddim.load_state_dict(torch.load(CHECKPOINT))
    ddim.eval()

    # Sample images
    with torch.no_grad():
        samples = ddim.sample(shape=SAMPLE_SHAPE, steps=steps)

    # Save grid
    vutils.save_image(samples, SAVE_PATH, nrow=4, normalize=True)
    print(f"[INFO] Saved generated samples to {SAVE_PATH}")

if __name__ == '__main__':
    main()
