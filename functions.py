import torch


def create_initial_grid(num_channels=16, grid_h=32, grid_w=32, device='cpu'):
    # if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
    init_state_grid = torch.zeros((num_channels, grid_h, grid_w)).to(device)
    # set seed
    return init_state_grid
