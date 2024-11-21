import tkinter as tk
from tkinter import Canvas
import random

import torch
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from functions import create_initial_grid, get_hps
from network import NeuralCA

# Create a function to generate a random RGB color
def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
# Function to update the entire grid with random colors
def update_grid():
    global grid, canvas

    with torch.no_grad():
        grid = net(grid.unsqueeze(0), 1)[-1].squeeze() # rgba

    frame = grid[:3].clamp(0, 1)

    # plt.imshow(frame.permute(1,2,0))
    # plt.show()

    for i in range(grid_h):
        for j in range(grid_w):
            update_pixel(i, j, color=list(int(x*255) for x in frame[:3, i, j]))

    # Schedule the grid to update every 500ms
    root.after(1, update_grid)

# Function to convert RGBA to hex format for Tkinter usage
def rgba_to_hex(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

# Function to update the color of a specific pixel
def update_pixel(i, j, color=(0, 0, 0)):
    color_hex = rgba_to_hex(color)
    canvas.create_rectangle(j * pixel_size, i * pixel_size,
                            (j + 1) * pixel_size, (i + 1) * pixel_size,
                            outline=color_hex, fill=color_hex)

# Function to handle pixel click event
def on_pixel_click(event):
    col = event.x // pixel_size  # Determine column
    row = event.y // pixel_size  # Determine row

    print('pixel clicked')

    grid[3:, row, col] = 1

# Initialize Tkinter window
root = tk.Tk()
root.title("RGB Pixel Grid")

session_id = 'flower_small_20241121_071301'

hps = get_hps(session_id)

# Grid dimensions and size
grid_h = hps['height']
grid_w = hps['width']
pixel_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralCA().to(device)
net.load_state_dict(torch.load(f'models/{session_id}.pth'))

num_channels = hps['num_channels']
grid_h = 128
grid_w = 128

grid = create_initial_grid(num_channels=num_channels, grid_h=grid_h, grid_w=grid_w, device=device)
# grid[3:, 16, 16] = 1

# Create a canvas widget
canvas = Canvas(root, width=grid_w * pixel_size, height=grid_h * pixel_size, bg='white')
canvas.pack()

# Bind mouse click event on the canvas
canvas.bind("<Button-1>", on_pixel_click)

# Start continuous update of the grid
update_grid()

# Start the Tkinter main loop
root.mainloop()
