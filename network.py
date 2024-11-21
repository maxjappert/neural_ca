from torch import nn
import numpy as np
import torch

# Implement the Neural Cellular Automata as a PyTorch module
class NeuralCA(nn.Module):
    def __init__(self, num_channels=16, hidden_dim=128):
        super(NeuralCA, self).__init__()
        self.state_dim = num_channels
        self.update = nn.Sequential(nn.Conv2d(num_channels, 3 * num_channels, 3, padding=1, groups=num_channels, bias=False),
                                    # perceive
                                    nn.Conv2d(3 * num_channels, hidden_dim, 1, bias=False),  # process perceptual inputs
                                    nn.ReLU(),  # nonlinearity
                                    nn.Conv2d(hidden_dim, num_channels, 1, bias=False))  # output a residual update
        self.update[-1].weight.data *= 0  # initial residual updates should be close to zero

        # First conv layer will use fixed Sobel filters to perceive neighbors
        identity = np.outer([0, 1, 0], [0, 1, 0])  # identity filter
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel x filter
        kernel = np.stack([identity, dx, dx.T], axis=0)  # stack (identity, dx, dy) filters
        kernel = np.tile(kernel, [num_channels, 1, 1])  # tile over channel dimension
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
