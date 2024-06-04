import os
from skimage import io, transform
from skimage.util import img_as_ubyte
import numpy as np
import random

# Define the source and destination directories
src_dir = '/data/wyli/data/glas/images/'
dst_dir = '/data/wyli/data/glas/images_64/'

os.makedirs(dst_dir, exist_ok=True)

# Get a list of all the image files in the source directory
image_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Define the size of the crop box
crop_size = np.array([64, 64])

# Define the number of crops per image
K = 5

for image_file in image_files:
    # Load the image
    image = io.imread(os.path.join(src_dir, image_file))

    # Get the size of the image
    image_size = np.array(image.shape[:2])

    for i in range(K):
        # Calculate a random start point for the crop box
        start = np.array([random.randint(0, image_size[0] - crop_size[0]), random.randint(0, image_size[1] - crop_size[1])])

        # Calculate the end point of the crop box
        end = start + crop_size

        # Crop the image
        cropped_image = img_as_ubyte(image[start[0]:end[0], start[1]:end[1]])

        # Save the cropped image to the destination directory
        io.imsave(os.path.join(dst_dir, f"{image_file}_{i}.png"), cropped_image)