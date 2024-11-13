import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import torchvision.transforms as transforms
from scipy.ndimage import maximum_filter
import imageio

grid_h = 32
grid_w = 32

sobel_x = torch.tensor([[-1, 0, +1],
                        [-2, 0, +2],
                        [-1, 0, +1]], dtype=torch.float32)
sobel_y = sobel_x.T

# first four channels are rgba, the rest are hidden and learned
num_channels = 16

sobel_x = torch.stack([sobel_x]*num_channels, dim=0)
sobel_x = torch.stack([sobel_x]*num_channels, dim=0)
sobel_y = torch.stack([sobel_y]*num_channels, dim=0)
sobel_y = torch.stack([sobel_y]*num_channels, dim=0)


class NeuralCA(nn.Module):
    def __init__(self):
        super(NeuralCA, self).__init__()
        self.fc1 = nn.Linear(num_channels*3, 128)
        self.fc2 = nn.Linear(128, 16)
        # self.conv1 = nn.Conv2d(num_channels*3, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def perceive(state_grid, sobel_x=sobel_x, sobel_y=sobel_y):
    # Convolve sobel filters with states
    # in x, y and channel dimension.
    grad_x = F.conv2d(state_grid, sobel_x, stride=1, padding=1)
    grad_y = F.conv2d(state_grid, sobel_y, stride=1, padding=1)
    # Concatenate the cell’s state channels,
    # the gradients of channels in x and
    # the gradient of channels in y.
    perception_grid = torch.stack([state_grid, grad_x, grad_y]).reshape(-1, grid_h, grid_w)
    return perception_grid


def stochastic_update(state_grid, ds_grid):
    # Zero out a random fraction of the updates.
    rand_mask = (torch.rand(grid_h, grid_w) < 0.5).to(torch.float32)
    ds_grid = ds_grid * rand_mask
    return state_grid + ds_grid


def alive_masking(state_grid):
    # Take the alpha channel as the measure of “life”.
    alpha_channel = state_grid[3, :, :]
    alive = F.max_pool2d(alpha_channel.unsqueeze(1), kernel_size=3, stride=1, padding=1) > 0.1
    alive = alive.squeeze().float()
    alive_mask = torch.stack([alive]*num_channels, dim=0)
    state_grid = state_grid * alive_mask
    return state_grid


# if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
init_state_grid = torch.zeros((num_channels, grid_h, grid_w))
# set seed
init_state_grid[3:, grid_h//2, grid_w//2] = 1

net = NeuralCA()

optimizer = torch.optim.Adam(net.parameters(), lr=0.00002)

num_epochs = 200

target_rgba = transforms.ToTensor()(Image.open('image32.png').convert('RGBA'))  # Ensure it's RGBA

for epoch_idx in range(num_epochs):
    print(f'Epoch {epoch_idx+1}')
    state_grid = init_state_grid
    optimizer.zero_grad()
    N = random.randint(64, 96)
    video_tensor = torch.zeros(N, 4, grid_h, grid_w)  # Example: 100 frames
    for n in range(N):
        perception_grid = perceive(state_grid)

        ds_grid = torch.zeros((num_channels, grid_h, grid_w))
        for i in range(grid_h):
            for j in range(grid_w):
                perceived_cell = perception_grid[:, i, j]
                ds = net(perceived_cell)
                ds_grid[:, i, j] = ds

        state_grid = stochastic_update(state_grid, ds_grid)
        state_grid = alive_masking(state_grid)
        state_grid = torch.clamp(state_grid, 0, 1)

        video_tensor[n] = state_grid[:4, :, :]

    mse = (state_grid[:4, :, :] - target_rgba).pow(2).mean()
    mse.backward()
    optimizer.step()

    print(f'MSE: {mse.item()}')

    rgba_image = transforms.ToPILImage()(state_grid[:4, :, :])

    # Save the image
    rgba_image.save(f'epoch{epoch_idx+1}.png', format='PNG')

    # Create a writer object for saving the video
    with imageio.get_writer(f'epoch{epoch_idx}.mp4', fps=12) as writer:
        for frame in video_tensor:
            # Convert the RGBA frame (C, H, W) to (H, W, C)
            # rgba_image = transforms.ToPILImage()(frame)

            # Write the frame to the video file
            frame = frame.detach().numpy().transpose(1, 2, 0)
            frame *= 255
            frame = frame.astype(np.uint8)

            writer.append_data(frame)
