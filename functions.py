import json

import torch


def create_initial_grid(num_channels=16, grid_h=32, grid_w=32):
    # if cell has alpha > 0.1 then it is mature, whereby its neighbours with alpha <= 0.1 are growing
    init_state_grid = torch.zeros((num_channels, grid_h, grid_w))
    # set seed
    return init_state_grid

def get_hps(session_id):
    with open(f'hps/{session_id}.json', 'r') as f:
        hps = json.load(f)

    return hps