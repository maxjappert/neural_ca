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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_h = 32
grid_w = 32

# Implement the Neural Cellular Automata as a PyTorch module
class NeuralCAComplete(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128):
        super(NeuralCAComplete, self).__init__()
        self.state_dim = state_dim
        self.update = nn.Sequential(nn.Conv2d(state_dim, 3 * state_dim, 3, padding=1, groups=state_dim, bias=False),
                                    # perceive
                                    nn.Conv2d(3 * state_dim, hidden_dim, 1, bias=False),  # process perceptual inputs
                                    nn.ReLU(),  # nonlinearity
                                    nn.Conv2d(hidden_dim, state_dim, 1, bias=False))  # output a residual update
        self.update[-1].weight.data *= 0  # initial residual updates should be close to zero

        # First conv layer will use fixed Sobel filters to perceive neighbors
        identity = np.outer([0, 1, 0], [0, 1, 0])  # identity filter
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel x filter
        kernel = np.stack([identity, dx, dx.T], axis=0)  # stack (identity, dx, dy) filters
        kernel = np.tile(kernel, [state_dim, 1, 1])  # tile over channel dimension
        self.update[0].weight.data[...] = torch.Tensor(kernel)[:, None, :, :]
        self.update[0].weight.requires_grad = False

    def forward(self, x, num_steps):
        alive_mask = lambda alpha: nn.functional.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
        frames = []
        for i in range(num_steps):
            alive_mask_pre = alive_mask(alpha=x[:, 3:4])
            update_mask = torch.rand(*x.shape, device=x.device) > 0.5  # run a state update 1/2 of time
            x = x + update_mask * self.update(x)  # state update!
            x = x * alive_mask_pre * alive_mask(alpha=x[:, 3:4])  # a cell is either living or dead
            frames.append(x.clone())
        return torch.stack(frames)  # axes: [N, B, C, H, W] where N is # of steps

num_channels = 16

# if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
init_state_grid = torch.zeros((num_channels, grid_h, grid_w)).to(device)
# set seed
init_state_grid[:, grid_h//2, grid_w//2] = 1

net = NeuralCAComplete().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

num_epochs = 200000

target_rgba = transforms.ToTensor()(Image.open('image32.png').convert('RGBA')).to(device)
target_rgba.requires_grad_()# Ensure it's RGBA

os.makedirs('images', exist_ok=True)
os.makedirs('videos', exist_ok=True)

def normalize_grads(model):  # makes training more stable, especially early on
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

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

    print(f'MSE: {mse.item()}')

    rgba_image = transforms.ToPILImage()(estimated_rgba)

    # Save the image
    rgba_image.save(f'images/epoch{epoch_idx+1}.png', format='PNG')

    if epoch_idx % 1000 == 0:
        # Create a writer object for saving the video
        with imageio.get_writer(f'videos/epoch{epoch_idx}.mp4', fps=12) as writer:
            for frame in frames:
                # Convert the RGBA frame (C, H, W) to (H, W, C)
                # rgba_image = transforms.ToPILImage()(frame)

                # Write the frame to the video file
                frame = frame.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                frame *= 255
                frame = frame.astype(np.uint8)

                writer.append_data(frame)
