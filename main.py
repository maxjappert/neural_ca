import json
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
from network import NeuralCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(input_image_path,
          lr=0.0001,
          num_epochs=20000,
          num_channels=16,
          hidden_dim=128,
          weight_decay=0,
          verbose=True,
          session_code=None):

    target_rgba = transforms.ToTensor()(Image.open(input_image_path).convert('RGBA')).to(device)
    target_rgba.requires_grad_()

    # target_rgba has shape (rgba_channels, height, width)

    grid_h = target_rgba.shape[1]
    grid_w = target_rgba.shape[2]

    def export_hyperparameters(session_code):
        os.makedirs('hps', exist_ok=True)

        hps = {
            'lr': lr,
            'num_channels': num_channels,
            'hidden_dim': hidden_dim,
            'height': grid_h,
            'width': grid_w,
            'input_image_path': input_image_path
        }

        with open(f'hps/{session_code}.json', 'w') as outfile:
            json.dump(hps, outfile)

    # if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
    init_state_grid = torch.zeros((num_channels, grid_h, grid_w)).to(device)
    # set seed
    init_state_grid[:, grid_h//2, grid_w//2] = 1

    init_state_grid = create_initial_grid(num_channels, grid_h, grid_w, device=device)
    init_state_grid[3:, grid_h // 2, grid_w // 2] = 1

    net = NeuralCA(num_channels=num_channels, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    def generate_session_code():
        # Get the current time and format it as a human-readable string
        session_code = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = input_image_path.split('/')[-1].split('.')[0]
        return name + '_' + session_code

    if session_code is not None:
        session_code = generate_session_code()

    export_hyperparameters(session_code)

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

        # Save the image

        if epoch_idx % 1000 == 0 and verbose:
            rgba_image = transforms.ToPILImage()(estimated_rgba)
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

train('flower_small.png')
