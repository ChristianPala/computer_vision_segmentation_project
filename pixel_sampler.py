# Auxiliary library to sample 10000 pixel from the training dataset, taking
# 5000 from the sky class and 5000 from the rest of the classes.

import os
from pathlib import Path
import numpy as np
import glob
import skimage

# Global variables:
from config import TRAINING_DATASET_PATH


def get_images():
    """
    Gets all the images from the training dataset
    :return: a list of all the images
    """
    images = []
    for img_path in glob.glob(os.path.join(TRAINING_DATASET_PATH, 'image_*.png')):
        images.append(skimage.io.imread(img_path))
    return images

def compute_total_sky_pixels(images):
    """
    Computes the total number of pixels in the sky class in the training dataset
    @param
    """
    pass