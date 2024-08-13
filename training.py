import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader
from  tqdm import tqdm
from copy import deepcopy

from UNet import ContextUNET
from DDPM import ContextDDPM
from utils import *

DEVICE="mps"

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = ContextDDPM(ContextUNET(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

trainset = FashionMNIST("../coding/Dataset", train=True, transform=transform, download=False)
dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

EPOCHS=100
LEARNING_RATE=3e-4
n_steps = ddpm.n_steps

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ddpm.parameters(), lr=LEARNING_RATE)


best_loss = float("inf")
best_model = None
best_epoch = None

for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, data in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS}", colour="#005500")):
        x = data[0].to(DEVICE)
        classes = data[1].to(DEVICE)
        n = len(x)

        eta = torch.randn(x.shape).to(DEVICE)
        t = torch.randint(0, n_steps, (n, )).to(DEVICE)

        noisy_images = ddpm(x, t, eta)
        eta_theta = ddpm.backward(noisy_images, t.reshape(n, -1), classes)

        loss = criterion(eta_theta, eta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()*len(x)/len(dataloader.dataset)

    log_string = f"Loss at epoch {epoch+1}: {epoch_loss:.3f}"

    if epoch_loss<best_loss:
        best_loss=epoch_loss
        best_epoch=epoch
        best_model=deepcopy(ddpm.state_dict())
        checkpoints = {
            "model": ddpm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss
        }
        torch.save(checkpoints, "checkpoints/best_model_mnist_context.pkl")
        log_string+=" --> model saved"

    ddpm.show_sample()

    print(log_string)