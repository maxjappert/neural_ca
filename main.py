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

from torch.optim.lr_scheduler import MultiStepLR

from functions import create_initial_grid
from network import NeuralCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_mse(epochs, mse_values, title="MSE over Epochs", save_path=None):
    """Plot MSE values over epochs using Matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("MSE (Mean Squared Error)")
    plt.grid(True)
    plt.tight_layout()

    # Optionally save the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    # Display the plot
    # plt.show()

def train(input_image_path,
          lr=0.0001,
          num_epochs=100000,
          num_channels=16,
          hidden_dim=128,
          weight_decay=0,
          verbose=True,
          session_code=None,
          min_steps=64,
          max_steps=96,
          h=128,
          w=128,
          generate_plots=True,
          milestones=[3000, 6000, 9000],  # lr scheduler milestones
          gamma=0.2,  # lr scheduler gamma
          pool_size=1024,
          batch_size=32,
          save_step=1000
          ):

    img = Image.open(input_image_path).convert('RGBA')

    img = img.resize((w, h), Image.LANCZOS)
    img.save('test.png')
    target_rgba = transforms.ToTensor()(img).to(device)
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

    init_state_grid = create_initial_grid(num_channels, grid_h, grid_w)
    init_state_grid[3:, grid_h // 2, grid_w // 2] = 1

    pool_state_grids = init_state_grid.unsqueeze(0).repeat(pool_size, 1, 1, 1)

    net = NeuralCA(num_channels=num_channels, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    def generate_session_code():
        # Get the current time and format it as a human-readable string
        session_code = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = input_image_path.split('/')[-1].split('.')[0]
        return name + '_' + session_code

    if session_code is None:
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
    mses = []

    def mse(estimated, target, per_item=False):
        squared_diff = (estimated - target).pow(2)

        if per_item:
            return squared_diff.mean(dim=(1, 2, 3))
        else:
            return squared_diff.mean()

    for epoch_idx in range(num_epochs):
        sample_idxs = torch.randint(0, pool_size, (batch_size,))
        batch = pool_state_grids[sample_idxs].to(device)

        # target_rgba.unsqueeze(0).repeat(32, 1, 1, 1)
        key_values = mse(batch[:, :4, :, :], target_rgba, per_item=True)
        sorted_indices = torch.argsort(key_values, descending=True)
        batch = batch[sorted_indices]
        batch[0] = init_state_grid

        optimizer.zero_grad()
        N = random.randint(min_steps, max_steps)

        states = net(batch, N).permute((1, 0, 2, 3, 4))
        frames = states[:, :, :4, :, :]
        estimated_rgbas = frames[:, -1, :, :, :].squeeze()

        loss = mse(estimated_rgbas, target_rgba)
        loss.backward()
        normalize_grads(net)
        optimizer.step()
        scheduler.step()

        pool_state_grids[sample_idxs] = batch.to('cpu')

        if loss < best_loss:
            best_loss = loss
            torch.save(net.state_dict(), 'models/' + session_code + '.pth')

        mses.append(loss.item())

        # Save the image

        if epoch_idx % save_step == 0 and verbose:
            rgba_image = transforms.ToPILImage()(estimated_rgbas[-1])
            rgba_image.save(f'images/{session_code}/epoch{epoch_idx + 1}.png', format='PNG')

            print(f'Epoch {epoch_idx + 1}')
            print(f'MSE: {loss.item()}\n')

            # Create a writer object for saving the video
            with imageio.get_writer(f'videos/{session_code}/epoch{epoch_idx}.mp4', fps=12) as writer:
                for frame in frames[-1]:
                    # Convert the RGBA frame (C, H, W) to (H, W, C)
                    # rgba_image = transforms.ToPILImage()(frame)

                    # Write the frame to the video file
                    frame = frame.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                    frame *= 255
                    frame = frame.astype(np.uint8)

                    writer.append_data(frame)

            if generate_plots:
                plot_path = f'images/{session_code}/plots'
                os.makedirs(plot_path, exist_ok=True)
                plot_mse(list(range(len(mses))), mses, save_path=os.path.join(plot_path, str(epoch_idx+1) + '.png'))


train('gecko.png',
      lr=2e-03,# lr=0.000001,
      session_code='gecko_l',
      min_steps=64,
      max_steps=96,
      h=96,
      w=96,
      generate_plots=True,
      pool_size=1024,
      batch_size=8,
      milestones=[3000, 6000, 9000],
      gamma=0.2,
      save_step=1000)
