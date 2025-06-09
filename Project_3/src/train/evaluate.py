import os
from datetime import datetime

import torch
import intel_extension_for_pytorch as ipex
from cleanfid import fid
from torch.utils.data import DataLoader

torch.manual_seed(0)
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from Project_3.src.model.UNet import UNet
from Project_3.src.model.ddim import DDIM


# Configuration
folder= '2025-06-02_23-01'  #'2025-06-08_23-44' # '2025-06-02_23-01'
steps = 100
CHECKPOINT = f"checkpoints/{folder}/ddim_final.pth"
SAVE_PATH = f"checkpoints/{folder}/ddim_sample_s{steps}.png"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
SAMPLE_SHAPE = (16, 3, 64, 64)
real_img_dir = f'generated_images/2025-06-08_23-44'
generated_dir = f'generated_images/s{steps}/{folder}'


def save_real_images(dataset_path, out_dir='real_images', img_size=64, num_images=500):
    os.makedirs(out_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    count = 0
    for imgs, _ in loader:
        for img in imgs:
            img = (img * 255).clamp(0, 255).byte()
            save_path = os.path.join(out_dir, f"real_{count:04d}.png")
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(save_path)
            count += 1
            if count >= num_images:
                return

def compute_fid(real_dir, fake_dir):
    return fid.compute_fid(real_dir, fake_dir, mode="legacy_pytorch", device=DEVICE, num_workers=0, use_dataparallel=False)

def fid_main():
    fid = compute_fid(real_dir=real_img_dir, fake_dir=generated_dir)
    print(fid)
    with open(f'{generated_dir}/fid.txt', 'w') as f:
        f.write(f"steps: {steps}, folder: {folder}\n")
        f.write(f"fid: {fid}\n")

def generate_and_save_images(ddim, num_images=500, batch_size=50, img_size=64, out_dir='generated_images',
                             device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    ddim.eval()

    n_batches = num_images // batch_size

    for i in tqdm(range(n_batches), desc="Generating images for FID"):
        with torch.no_grad():
            imgs = ddim.sample(shape=(batch_size, 3, 64, 64), steps=steps)
        imgs = ((imgs + 1) / 2).clamp(0, 1)  # scale to [0, 1]
        for j, img in enumerate(imgs):
            save_image(img, os.path.join(out_dir, f"img_{i * batch_size + j:04d}.png"))


def generate_main():
    #Load model checkpoint
    unet = UNet(in_channels=3, out_channels=3).to(DEVICE)
    ddim = DDIM(unet, timesteps=1000).to(DEVICE)
    ddim.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    generate_and_save_images(ddim, num_images=500, batch_size=32, out_dir=generated_dir, device=DEVICE)

    save_real_images("../../data", real_img_dir, num_images=500)

def generate_grid(ddim_model, n=100, save_dir="grid_check"):
    noise = torch.randn((n, 3, 64, 64), device=DEVICE)
    imgs = ddim_model.sample(noise=noise, steps=steps, shape=(n, 3, 64, 64))
    vutils.save_image(imgs, os.path.join(save_dir, f"grid_s{steps}.png"), nrow=10, normalize=True)

def grid_main():
    # Load model
    unet = UNet(in_channels=3, out_channels=3).to(DEVICE)
    ddim = DDIM(unet, timesteps=1000).to(DEVICE)
    ddim.load_state_dict(torch.load(CHECKPOINT))
    ddim.eval()
    generate_grid(ddim,save_dir=f"checkpoints/{folder}")

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
    #main()
    generate_main()
    #fid_main()
    #grid_main()
