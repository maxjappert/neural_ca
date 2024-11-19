import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import torchvision.transforms as transforms
import imageio
from datetime import datetime

from functions import create_initial_grid
from network import NeuralCAComplete

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_h = 32
grid_w = 32

num_channels = 16

# if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
init_state_grid = torch.zeros((num_channels, grid_h, grid_w)).to(device)
# set seed
init_state_grid[:, grid_h//2, grid_w//2] = 1

init_state_grid = create_initial_grid(num_channels, grid_h, grid_w, device=device)
init_state_grid[3:, grid_h // 2, grid_w // 2] = 1

net = NeuralCAComplete().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

num_epochs = 200000

target_rgba = transforms.ToTensor()(Image.open('image32.png').convert('RGBA')).to(device)
target_rgba.requires_grad_()# Ensure it's RGBA

def generate_session_code():
    # Get the current time and format it as a human-readable string
    session_code = datetime.now().strftime('%Y%m%d_%H%M%S')
    return session_code

session_code = generate_session_code()

os.makedirs('images', exist_ok=True)
os.makedirs('videos', exist_ok=True)
os.makedirs(f'videos/{session_code}', exist_ok=True)
os.makedirs(f'images/{session_code}', exist_ok=True)
os.makedirs('models', exist_ok=True)

def normalize_grads(model):  # makes training more stable, especially early on
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

best_loss = float('inf')

for epoch_idx in range(num_epochs):
    print(f'Epoch {epoch_idx+1}')
    state_grid = init_state_grid
    optimizer.zero_grad()
    N = random.randint(64, 96)

    states = net(state_grid.unsqueeze(0), N)
    frames = states[:, :, :4, :, :]
    estimated_rgba = frames[-1].squeeze()

    mse = (estimated_rgba - target_rgba).pow(2).mean()
    mse.backward()
    normalize_grads(net)
    optimizer.step()

    if mse < best_loss:
        best_loss = mse
        torch.save(net.state_dict(), 'models/' + session_code + '.pth')

    print(f'MSE: {mse.item()}')

    rgba_image = transforms.ToPILImage()(estimated_rgba)

    # Save the image

    if epoch_idx % 1000 == 0:
        rgba_image.save(f'images/{session_code}/epoch{epoch_idx + 1}.png', format='PNG')

        # Create a writer object for saving the video
        with imageio.get_writer(f'videos/{session_code}/epoch{epoch_idx}.mp4', fps=12) as writer:
            for frame in frames:
                # Convert the RGBA frame (C, H, W) to (H, W, C)
                # rgba_image = transforms.ToPILImage()(frame)

                # Write the frame to the video file
                frame = frame.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                frame *= 255
                frame = frame.astype(np.uint8)

                writer.append_data(frame)
