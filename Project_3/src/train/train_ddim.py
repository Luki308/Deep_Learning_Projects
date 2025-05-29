import logging
from datetime import datetime

import torch
import intel_extension_for_pytorch as ipex
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from Project_3.src.model.UNet import UNet
from Project_3.src.model.ddim import DDIM

import os

# ---- Config ----
BATCH_SIZE = 32
IMAGE_SIZE = 64
LR = 2e-4
EPOCHS = 40
TIMESTEPS = 1000
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
# ---- Dataset ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


dataset = ImageFolder("../../data", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ---- Model ----
unet = UNet(in_channels=3, out_channels=3).to(DEVICE)
ddim = DDIM(unet, timesteps=TIMESTEPS).to(DEVICE)

optimizer = optim.Adam(ddim.parameters(), lr=LR)

ddim.train()
ddim, optimizer = ipex.optimize(ddim, optimizer=optimizer)

def main():
    file_path = f"checkpoints/{timestamp}/config.txt"
    with open(file_path, 'a') as file:
        file.write(f"Batch size:{BATCH_SIZE}, Image size: {IMAGE_SIZE}, Lr: {LR}, Epoches: {EPOCHS}, Timesteps: {TIMESTEPS}")

    # ---- Training Loop ----
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (real_imgs.size(0),), device=DEVICE)

            loss = ddim(real_imgs, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % 3 == 0:
            torch.save(ddim.state_dict(), f"checkpoints/{timestamp}/ddim_epoch_{epoch + 1}.pth")
            print(f"[INFO] Saved checkpoint at epoch {epoch + 1}")

    torch.save(ddim.state_dict(), f"checkpoints/{timestamp}/ddim_final.pth")

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(f"checkpoints/{timestamp}", exist_ok=True)
    main()