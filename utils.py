import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from PIL import Image

def show_images(images, title=""):
    images = [(data - torch.min(data)) / (torch.max(data) - torch.min(data)) for data in images]
    Grid = make_grid(images, nrow=5, padding=0)
    plt.imshow(Grid.permute(1, 2, 0), cmap="gray") 
    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0])
        break

def show_forward(ddpm, loader):
    device = ddpm.device
    for batch in loader:
        batch = batch[0].to(device)
        show_images(batch, "0% Noisy")
        values = [0.25,  0.5, 0.75, 1]
        for val in values:
            imgs = ddpm.forward(batch, [int(val*ddpm.n_steps-1) for _ in range(len(batch))])
            show_images(imgs, title=f"{int(val*100)}% Noisy")
        break