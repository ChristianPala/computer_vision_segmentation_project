# Auxiliary library to sample pixels from the dataset
# Libraries:
# Data Manipulation:
import os
from pathlib import Path

import numpy as np
from glob import glob
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
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # compute the total number of pixels in the sky class
    total_sky_pixels = 0
    total_non_sky_pixels = 0
    for mask in binary_masks:
        total_sky_pixels += np.sum(mask == 255)
        total_non_sky_pixels += np.sum(mask != 255)

    # compute the sampling proportions to sample for each images
    return total_sky_pixels, total_non_sky_pixels


def sample_pixels(total_count: int, train: bool = True) -> pd.DataFrame:
    """
    Sample a balanced dataset, from two classes, from the images in the dataset.
    @param total_count: the total number of pixels to sample
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the sampled pixels
    """
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # compute the sampling proportions to sample for each images
    sky_count, non_sky_count = compute_sampling_proportions(train)

    sky_sample = total_count / sky_count
    non_sky_sample = total_count / non_sky_count

    # create the training dataset
    df = pd.DataFrame(columns=['image', 'r', 'g', 'b', 'class'])

    # get all the images in the path:
    images = []
    for image_path in glob(os.path.join(path, 'image_*.png')):
        images.append(skimage.io.imread(image_path))

    # get all the binary masks in the path:
    binary_masks = []
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # the binary mask only requires 1 channel, so we can remove the other 2
    binary_masks = [mask[:, :, 0] for mask in binary_masks]

    # convert all images and binary masks to float
    images = [skimage.img_as_float(image) for image in images]
    binary_masks = [skimage.img_as_float(mask) for mask in binary_masks]

    # sample the pixels
    image_count = 0
    for image, mask in zip(images, binary_masks):
        # divide the image into 3 channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        # flatten the pixels of each channel:
        r, g, b = r.flatten(), g.flatten(), b.flatten()
        # flatten the mask:
        mask = mask.flatten()
        # sample the pixels from the sky class
        sky_pixels = np.random.choice(np.arange(len(mask)), size=int(sky_sample * np.sum(mask == 1)),
                                      replace=False)
        # sample the pixels from the non-sky class
        non_sky_pixels = np.random.choice(np.arange(len(mask)), size=int(non_sky_sample * np.sum(mask == 0)),
                                          replace=False)

        # create the dataframe for the sky pixels
        sky_df = pd.DataFrame({'image': [image_count] * len(sky_pixels),
                               'r': r[sky_pixels],
                               'g': g[sky_pixels],
                               'b': b[sky_pixels],
                               'class': [1] * len(sky_pixels)})

        # create the dataframe for the non-sky pixels
        non_sky_df = pd.DataFrame({'image': [image_count] * len(non_sky_pixels),
                                   'r': r[non_sky_pixels],
                                   'g': g[non_sky_pixels],
                                   'b': b[non_sky_pixels],
                                   'class': [0] * len(non_sky_pixels)})
        # concatenate the dataframes under the main dataframe
        df = pd.concat([df, sky_df, non_sky_df], ignore_index=True)

        image_count += 1

    return df


def main() -> None:
    """
    Main function to sample pixels from the training dataset, and save them to a csv file
    """
    # sample amount of pixels from the training and testing dataset:
    training_pixels: int = 10000
    testing_pixels: int = 5000

    # sample 10000 pixels from the training dataset
    train_df: pd.DataFrame = sample_pixels(training_pixels, train=True)
    # sample 10000 pixels from the testing dataset
    test_df: pd.DataFrame = sample_pixels(testing_pixels, train=False)

    # save the sampled pixels
    train_df.to_csv(Path(TRAINING_DATASET_PATH, 'train_by_pixel.csv'), index=False)
    test_df.to_csv(Path(TESTING_DATASET_PATH, 'test_by_pixel.csv'), index=False)


if __name__ == '__main__':
    main()
