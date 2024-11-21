import tkinter as tk
from tkinter import Canvas
import random
import torch
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from functions import create_initial_grid, get_hps
from network import NeuralCA

# Function to convert frame to a Tkinter-compatible pixel format
def frame_to_photoimage(frame, canvas_w, canvas_h):
    # Convert to (H, W, 3) uint8 format
    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(frame_np, 'RGB')
    # Scale to canvas size
    img_resized = img.resize((canvas_w, canvas_h), Image.NEAREST)
    return ImageTk.PhotoImage(img_resized)

# Function to update the grid and display on the canvas
def update_grid():
    global grid, photo, canvas_image

    with torch.no_grad():
        grid = net(grid.unsqueeze(0), 1)[-1].squeeze()  # rgba

    frame = grid[:3].clamp(0, 1)  # Extract RGB channels and clamp to [0, 1]

    # Convert frame to PhotoImage and update the canvas
    photo = frame_to_photoimage(frame, canvas_width, canvas_height)
    canvas.itemconfig(canvas_image, image=photo)

    # Schedule the grid to update every 1 ms
    root.after(1, update_grid)

# Function to handle pixel click event
def on_pixel_click(event):
    col = event.x // pixel_size  # Determine column
    row = event.y // pixel_size  # Determine row
    grid[3:, row, col] = 1  # Add some perturbation
    print(f"Pixel clicked at ({row}, {col})")

# Initialize Tkinter window
root = tk.Tk()
root.title("Optimized RGB Pixel Grid")

# Load session configuration
session_id = 'image32_20241121_070617'
hps = get_hps(session_id)

# Grid dimensions and size
grid_h = hps['height']
grid_w = hps['width']
pixel_size = 4  # Use smaller pixels for better rendering performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NeuralCA().to(device)
net.load_state_dict(torch.load(f'models/{session_id}.pth'))

num_channels = hps['num_channels']
grid_h = 128
grid_w = 128

grid = create_initial_grid(num_channels=num_channels, grid_h=grid_h, grid_w=grid_w, device=device)

# Create a canvas widget
canvas = Canvas(root, width=grid_w * pixel_size, height=grid_h * pixel_size, bg='white')
canvas.pack()

canvas_width = grid_w * pixel_size
canvas_height = grid_h * pixel_size

# Add an image placeholder on the canvas
photo = None
canvas_image = canvas.create_image(0, 0, anchor="nw", image=photo)

# Bind mouse click event on the canvas
canvas.bind("<Button-1>", on_pixel_click)

# Start continuous update of the grid
update_grid()

# Start the Tkinter main loop
root.mainloop()
