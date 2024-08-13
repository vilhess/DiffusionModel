import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def show_images(images, title=""):
    fig, axs = plt.subplots(4, 5, figsize=(10, 6))
    for i, data in enumerate(images):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(data.permute(1, 2, 0), cmap="gray")
        axs[row, col].axis('off')
        
        if i == 19:
            break
    fig.suptitle(title, fontsize=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuster la mise en page pour le titre
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