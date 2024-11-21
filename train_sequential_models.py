import os

from main import train

path = 'frames'

frames = []

for filename in os.listdir(path):
    if filename.endswith('.png'):
        frames.append(filename)

for idx, frame in enumerate(frames):
    train(os.path.join(path, frame), num_epochs=5000, verbose=True, session_code=f'frame1')