import tkinter as tk
from tkinter import Canvas
import random
from PIL import Image, ImageTk

# Create a function to generate a random RGB color
def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Function to update the entire grid with random colors
def update_grid():
    global pixel_grid, canvas
    for i in range(32):
        for j in range(32):
            pixel_grid[i][j] = (0, 0, 0)
            update_pixel(i, j)

    # Schedule the grid to update every 500ms
    # root.after(500, update_grid)

# Function to update the color of a specific pixel
def update_pixel(i, j, color=(0, 0, 0)):
    color_hex = '#%02x%02x%02x' % color  # Convert RGB to hex
    canvas.create_rectangle(j * pixel_size, i * pixel_size,
                            (j + 1) * pixel_size, (i + 1) * pixel_size,
                            outline=color_hex, fill=color_hex)

# Function to handle pixel click event
def on_pixel_click(event):
    col = event.x // pixel_size  # Determine column
    row = event.y // pixel_size  # Determine row
    # When a pixel is clicked, update all pixels
    update_pixel(row, col, random_color())

# Initialize Tkinter window
root = tk.Tk()
root.title("RGB Pixel Grid")

# Grid dimensions and size
grid_size = 32
pixel_size = 20

# Initialize the pixel grid with random colors
pixel_grid = [[(255, 255, 255) for _ in range(grid_size)] for _ in range(grid_size)]

# Create a canvas widget
canvas = Canvas(root, width=grid_size * pixel_size, height=grid_size * pixel_size)
canvas.pack()

# Draw the initial grid
for i in range(grid_size):
    for j in range(grid_size):
        update_pixel(i, j)

# Bind mouse click event on the canvas
canvas.bind("<Button-1>", on_pixel_click)

# Start continuous update of the grid
update_grid()

# Start the Tkinter main loop
root.mainloop()
