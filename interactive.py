import tkinter as tk
from tkinter import Canvas
import random

import torch
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from functions import create_initial_grid
from network import NeuralCAComplete


# Create a function to generate a random RGB color
def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def rescale_rgb_up(rgb):
    new_rgb = list(int(x*255) for x in rgb)

    for i in range(len(new_rgb)):
        if new_rgb[i] > 255:
            new_rgb[i] = 255
        elif new_rgb[i] < 0:
            new_rgb[i] = 0

    return tuple(new_rgb)

# Function to update the entire grid with random colors
def update_grid(grid):
    global pixel_grid, canvas

    print(grid.max())

    with torch.no_grad():
        grid = net(grid.unsqueeze(0), 1)[-1].squeeze() # rgba
    frame = grid[:3]

    # plt.imshow(frame.permute(1,2,0))
    # plt.show()

    for i in range(32):
        for j in range(32):
            update_pixel(i, j, color=rescale_rgb_up(frame[:3, i, j]))

    # Schedule the grid to update every 500ms
    root.after(10, lambda: update_grid(grid))

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

    grid[:, row, col] = 1

    print(grid.max())

# Initialize Tkinter window
root = tk.Tk()
root.title("RGB Pixel Grid")

# Grid dimensions and size
grid_size = 32
pixel_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralCAComplete().to(device)
net.load_state_dict(torch.load('models/model.pth'))

num_channels = 16
grid_h = 32
grid_w = 32

grid = create_initial_grid(num_channels=num_channels, grid_h=grid_h, grid_w=grid_w, device=device)
grid[3:, 16, 16] = 1

# Create a canvas widget
canvas = Canvas(root, width=grid_size * pixel_size, height=grid_size * pixel_size, bg='white')
canvas.pack()

# Bind mouse click event on the canvas
canvas.bind("<Button-1>", on_pixel_click)

# Start continuous update of the grid
update_grid(grid)

# Start the Tkinter main loop
root.mainloop()
