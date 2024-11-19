import tkinter as tk
from tkinter import Canvas
import random

import torch
from PIL import Image, ImageTk

from functions import create_initial_grid
from network import NeuralCAComplete


# Create a function to generate a random RGB color
def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Function to update the entire grid with random colors
def update_grid():
    global pixel_grid, canvas

    with torch.no_grad():
        grid = net(grid, 1).squeeze() # rgba
    print('here')
    frame = grid[:4]

    for i in range(32):
        for j in range(32):
            if frame[-1, i, j] < 0.5:
                update_pixel(i, j, (255, 255, 255))
            else:
                int_tuple = tuple(int(x*255) for x in frame[:3, i, j])
                update_pixel(i, j, color=int_tuple)

    # Schedule the grid to update every 500ms
    root.after(500, update_grid)

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

    grid[3:, row, col] = 1

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

# Draw the initial grid
for i in range(grid_size):
    for j in range(grid_size):
        update_pixel(i, j, (0, 0, 0))

# Bind mouse click event on the canvas
canvas.bind("<Button-1>", on_pixel_click)

# Start continuous update of the grid
update_grid()

# Start the Tkinter main loop
root.mainloop()
