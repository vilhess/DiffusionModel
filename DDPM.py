import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import show_images

class DDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=1e-4, max_beta=0.02, device="cpu"):
        super(DDPM, self).__init__()
        self.device = device
        self.network = network.to(device)
        self.n_steps = n_steps
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward(self, input, t, eta=None):
        if eta is None:
            eta = torch.randn(size=(input.shape)).to(self.device)
        n, _, _, _ = input.shape
        return self.alpha_bar[t].sqrt().reshape(n, 1, 1, 1)*input + (1 - self.alpha_bar[t]).sqrt().reshape(n, 1, 1, 1)*eta

    def backward(self, noise, t):
        return self.network(noise, t)

    def show_sample(self):
      with torch.no_grad():
        self.eval()
        x = torch.randn(size=(1, 1, 28, 28)).to(self.device)
        for t in tqdm(list(range(self.n_steps))[::-1], desc="Generating sample"):
          time = torch.tensor([t]).to(self.device)
          eta_theta = self.backward(x, time)
          alpha_t = self.alphas[t].to(self.device)
          alpha_bar_t = self.alpha_bar[t].to(self.device)
          x = 1/torch.sqrt(alpha_t) * (x - (1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eta_theta)
          if t>0:
            z = torch.randn(size=(x.shape)).to(self.device)
            beta_t = self.betas[t].to(self.device)
            sigma_t = torch.sqrt(beta_t).to(self.device)
            x = x + sigma_t*z

      plt.imshow(x[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")
      plt.show()

    def generate_sample(self):
        device = self.device
        with torch.no_grad():
            x = torch.randn(15, 1, 28, 28).to(device)

            vals = [int(0*self.n_steps), int(0.25*self.n_steps), int(0.5*self.n_steps), int(0.75*self.n_steps), self.n_steps-1]

            for idx, t in enumerate(tqdm(list(range(self.n_steps))[::-1])):
                time_tensor = (t*torch.ones((15, 1))).to(device).long()
                eta_theta = self.backward(x, time_tensor)
                alphas_t = self.alphas[t]
                alphas_bar_t = self.alpha_bar[t]

                x = (1 / alphas_t.sqrt()) * (x - (1 - alphas_t) / (1 - alphas_bar_t).sqrt() * eta_theta)
                if t>0:
                    z = torch.randn(15, 1, 28, 28).to(device)
                    beta_t=self.betas[t]
                    sigma_t=beta_t.sqrt()
                    x = x+sigma_t*z


                if t in vals:
                    show_images(x.cpu(), f"{int(t/self.n_steps * 100)}% Noisy")

class ContextDDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=1e-4, max_beta=0.02, device="cpu"):
        super(ContextDDPM, self).__init__()
        self.device = device
        self.network = network.to(device)
        self.n_steps = n_steps
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward(self, input, t, eta=None):
        if eta is None:
            eta = torch.randn(size=(input.shape)).to(self.device)
        n, _, _, _ = input.shape
        return self.alpha_bar[t].sqrt().reshape(n, 1, 1, 1)*input + (1 - self.alpha_bar[t]).sqrt().reshape(n, 1, 1, 1)*eta

    def backward(self, noise, t, c):
        return self.network(noise, t, c)

    def show_sample(self):
      with torch.no_grad():
        self.eval()
        x = torch.randn(size=(1, 1, 28, 28)).to(self.device)
        c = torch.tensor(2).long().to(self.device)
        for t in tqdm(list(range(self.n_steps))[::-1], desc="Generating sample"):
          time = torch.tensor([t]).to(self.device)
          eta_theta = self.backward(x, time, c)
          alpha_t = self.alphas[t].to(self.device)
          alpha_bar_t = self.alpha_bar[t].to(self.device)
          x = 1/torch.sqrt(alpha_t) * (x - (1-alpha_t)/torch.sqrt(1-alpha_bar_t)*eta_theta)
          if t>0:
            z = torch.randn(size=(x.shape)).to(self.device)
            beta_t = self.betas[t].to(self.device)
            sigma_t = torch.sqrt(beta_t).to(self.device)
            x = x + sigma_t*z

      plt.imshow(x[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")
      plt.show()

    def generate_sample(self):
        device = self.device
        with torch.no_grad():
            x = torch.randn(20, 1, 28, 28).to(device)
            c = torch.Tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]).to(device).long()

            vals = [int(0*self.n_steps), int(0.25*self.n_steps), int(0.5*self.n_steps), int(0.75*self.n_steps), self.n_steps-1]

            for idx, t in enumerate(tqdm(list(range(self.n_steps))[::-1])):
                time_tensor = (t*torch.ones((20, 1))).to(device).long()
                eta_theta = self.backward(x, time_tensor, c)
                alphas_t = self.alphas[t]
                alphas_bar_t = self.alpha_bar[t]

                x = (1 / alphas_t.sqrt()) * (x - (1 - alphas_t) / (1 - alphas_bar_t).sqrt() * eta_theta)
                if t>0:
                    z = torch.randn(20, 1, 28, 28).to(device)
                    beta_t=self.betas[t]
                    sigma_t=beta_t.sqrt()
                    x = x+sigma_t*z


                if t in vals:
                    show_images(x.cpu(), f"{int(t/self.n_steps * 100)}% Noisy")
