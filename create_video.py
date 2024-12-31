import os
import sys

import cv2
import argparse


def create_video_from_images(input_dir):
    # Get list of files in the directory
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

    if not files:
        print("No PNG files found in the directory.")
        return

    # Read the first image to get the width and height for the video
    first_image_path = os.path.join(input_dir, files[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create a VideoWriter object
    output_path = os.path.join(input_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for AVI format
    video_writer = cv2.VideoWriter(output_path, fourcc, 60, (width, height))

    # Write each image as a frame in the video
    for file in files:
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)
        video_writer.write(img)
        # print(f"Adding {file} to video...")

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video successfully created: {output_path}")


if __name__ == "__main__":
    # Set up argument parsing
    # parser = argparse.ArgumentParser(description="Convert image sequence to video.")
    # parser.add_argument("directory", help="Path to the directory containing the images.")

    # args = parser.parse_args()
    input_dir = sys.argv[1]

    # Check if the directory exists
    if not os.path.isdir(input_dir):
        print(f"The directory {input_dir} does not exist.")
    else:
        create_video_from_images(input_dir)
