# Auxiliary library to sample 10000 pixel from the training dataset, taking
# 5000 from the sky class and 5000 from the rest of the classes.

import os
import numpy as np
import glob
import skimage
import pandas as pd

# Global variables:
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


def compute_sampling_proportions(train: bool = True) -> (int, int):
    """
    Computes the total number of pixels in the sky class in the training dataset
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the total number of pixels in the sky class
    """
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # get all the binary masks in the path:
    binary_masks = []
    for mask_path in glob.glob(os.path.join(path, 'mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # compute the total number of pixels in the sky class
    total_sky_pixels = 0
    total_non_sky_pixels = 0
    for mask in binary_masks:
        total_sky_pixels += np.sum(mask == 1)
        total_non_sky_pixels += np.sum(mask != 1)

    # compute the sampling proportions to sample for each images
    ski_proportion = total_sky_pixels / (total_sky_pixels + total_non_sky_pixels)
    non_ski_proportion = total_non_sky_pixels / (total_sky_pixels + total_non_sky_pixels)

    return ski_proportion, non_ski_proportion


def sample_pixels(total_count: int, train: bool = True) -> np.ndarray:
    """
    Sample a balanced dataset, from two classes, from the images in the dataset.
    @param total_count: the total number of pixels to sample
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the sampled pixels
    """
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # get all the images in the path:
    images = []
    for image_path in glob.glob(os.path.join(path, 'image_*.png')):
        images.append(skimage.io.imread(image_path))

    # get all the binary masks in the path:
    binary_masks = []
    for mask_path in glob.glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # compute the sampling proportions to sample for each images
    sky_proportion, non_sky_proportion = compute_sampling_proportions(train)

    # for each image, sample sky_proportion * total_count pixels from the sky class
    # and non_sky_proportion * total_count pixels from the rest of the classes
    sampled_pixels = []
    for image, mask in zip(images, binary_masks):
        # get the sky pixels
        sky_pixels = image[mask == 1]
        # get the non-sky pixels
        non_sky_pixels = image[mask != 1]
        # sample the pixels
        sky_pixels = np.array(sky_pixels)
        non_sky_pixels = np.array(non_sky_pixels)
        sky_pixels = sky_pixels[np.random.choice(sky_pixels.shape[0],
                                                 int(sky_proportion * total_count),
                                                 replace=False), :, :]
        non_sky_pixels = non_sky_pixels[np.random.choice(non_sky_pixels.shape[0],
                                                         int(non_sky_proportion * total_count),
                                                         replace=False), :, :]
        # append both to the sampled pixels
        sampled_pixels.append(sky_pixels)
        sampled_pixels.append(non_sky_pixels)

    return np.array(sampled_pixels)


def main() -> None:
    """
    Main function to sample 10000 pixels from the training dataset,
    taking 5000 from the sky class and 5000 from the rest of the classes.
    """
    # sample 10000 pixels from the training dataset
    sampled_pixels_tr = sample_pixels(10000, train=True)
    train_df = pd.DataFrame(sampled_pixels_tr.reshape(-1, 3))
    train_df.to_csv('train.csv', index=False)

    # sample 5000 pixels from the testing dataset
    sampled_pixels_te = sample_pixels(5000, train=False)
    test_df = pd.DataFrame(sampled_pixels_te.reshape(-1, 3))
    test_df.to_csv('test.csv', index=False)


