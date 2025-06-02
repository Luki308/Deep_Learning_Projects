import os
from datetime import datetime

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from Project_3.src.model.UNet import UNet
from Project_3.src.model.ddim import DDIM

torch.manual_seed(733)
np.random.seed(733)

# ---- Config ----
config = {"BATCH_SIZE": 32,
          "IMAGE_SIZE": 64,
          "LR": 2e-4,
          "EPOCHS": 40,
          "TIMESTEPS": 1000,
          "SUBSET": None,
          "DEVICE": torch.device("xpu" if torch.xpu.is_available() else "cpu"),
          "CHECKPOINT": "checkpoints/2025-06-01_01-14/ddim_final.pth"
}
# ---- Dataset ----
transform = transforms.Compose(
    [transforms.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),  # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), ])

dataset = ImageFolder("../../data", transform=transform)
if config["SUBSET"] is not None:
    dataset = Subset(dataset, np.random.choice(len(dataset), replace=False, size=config["SUBSET"]))
dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4)

def get_run_name(timestamp):
    """Create a unique name for each experiment run"""
    return f"{timestamp}---t{config["TIMESTEPS"]}-img{config["IMAGE_SIZE"]}-lr{config["LR"]}-ep{config["EPOCHS"]}-batch{config["BATCH_SIZE"]}-s{config["SUBSET"]}-previous{0 if config["CHECKPOINT"] is None else 1}"


def main():
    file_path = f"checkpoints/{timestamp}/config.txt"
    with open(file_path, 'a') as file:
        file.write(f"{config}")

    # ---- Model ----
    unet = UNet(in_channels=3, out_channels=3).to(config["DEVICE"])
    ddim = DDIM(unet, timesteps=config["TIMESTEPS"]).to(config["DEVICE"])
    if config["CHECKPOINT"] is not None:
        ddim.load_state_dict(torch.load(config["CHECKPOINT"]))
    optimizer = optim.Adam(ddim.parameters(), lr=config["LR"])

    ddim.train()
    ddim, optimizer = ipex.optimize(ddim, optimizer=optimizer)

    writer = SummaryWriter(f'runs/{get_run_name(timestamp)}')
    #writer.add_graph(ddim, (torch.randn([16, 3, 64, 64], device=config["DEVICE"]), torch.randint(0, config["TIMESTEPS"], (16,), device=config["DEVICE"])))

    # ---- Training Loop ----
    for epoch in range(config["EPOCHS"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config["EPOCHS"]}")
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(config["DEVICE"])
            t = torch.randint(0, config["TIMESTEPS"], (real_imgs.size(0),), device=config["DEVICE"])

            loss = ddim(real_imgs, t)
            writer.add_scalar('Loss/train', loss.item(), epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % 5 == 0:
            torch.save(ddim.state_dict(), f"checkpoints/{timestamp}/ddim_epoch_{epoch + 1}.pth")
            print(f"[INFO] Saved checkpoint at epoch {epoch + 1}")

    torch.save(ddim.state_dict(), f"checkpoints/{timestamp}/ddim_final.pth")
    writer.add_hparams({key: value for key, value in config.items() if key not in ["DEVICE", "CHECKPOINT"]}, {})
    writer.flush()
    writer.close()


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(f"checkpoints/{timestamp}", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    #config["EPOCHS"] = 10
    #config["LR"] = 1e-3
    main()
