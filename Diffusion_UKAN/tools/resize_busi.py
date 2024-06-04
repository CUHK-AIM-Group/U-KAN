import os
from skimage import io, transform
from skimage.util import img_as_ubyte
import numpy as np

# Define the source and destination directories
src_dir = '/data/wyli/data/busi/images/'
dst_dir = '/data/wyli/data/busi/images_64/'

os.makedirs(dst_dir, exist_ok=True)

# Get a list of all the image files in the source directory
image_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Define the size of the crop box
crop_size = np.array([400 ,400])

# Define the size of the resized image
# resize_size = (64, 64)
resize_size = (64, 64)

for image_file in image_files:
    # Load the image
    image = io.imread(os.path.join(src_dir, image_file))
    print(image.shape)


    # Calculate the center of the image
    center = np.array(image.shape[:2]) // 2

    # Calculate the start and end points of the crop box
    start = center - crop_size // 2
    end = start + crop_size

    # Crop the image
    cropped_image = img_as_ubyte(image[start[0]:end[0], start[1]:end[1]])

    # Resize the cropped image
    resized_image = transform.resize(cropped_image, resize_size, mode='reflect')

    # Save the resized image to the destination directory
    io.imsave(os.path.join(dst_dir, image_file), img_as_ubyte(resized_image))