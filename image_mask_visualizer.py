import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skimage
from typing import Union

from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


def get_image_and_segmentation(img_number: Union[str, int], train: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gets the image corresponding to the given number and its full segmentation.
    @param: img_number: Index of the image desired to be plotted
    @param: train: True if the image is from the training dataset, False if it is from the testing dataset
    :return image and segmentation as numpy ndarrays.
    """
    # Check that the given image number is a string
    img_number = str(img_number)

    im_path = str(Path(TRAINING_DATASET_PATH, 'image_' + img_number + '.png')) \
        if train else str(Path(TESTING_DATASET_PATH, 'image_' + img_number + '.png'))
    segmentation_path = str(Path(TRAINING_DATASET_PATH, 'mask_' + img_number + '.png')) \
        if train else str(Path(TESTING_DATASET_PATH, 'mask_' + img_number + '.png'))

    # Get the actual image and segmentation as numpy ndarrays
    im = skimage.io.imread(im_path)
    segmentation = skimage.io.imread(segmentation_path)
    return im, segmentation


def visualize_image_and_segmentation(img_number: Union[str, int], train: bool = True) -> None:
    """
    Plots side-by-side the image corresponding to the given number and its full segmentation.
    @param: img_number: Index of the image desired to be plotted
    :return None. It plots directly the corresponding images.
    """
    # Get the actual image and segmentation as numpy ndarrays
    im, segmentation = get_image_and_segmentation(img_number, train)

    # Plot the image and its segmentation side-by-side
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.set_title('Original image')
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.imshow(im, vmin=0, vmax=255)

    ax1.set_title('Image Segmentation')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.imshow(segmentation, vmin=0, vmax=255)
    fig.show()


def main():
    visualize_image_and_segmentation(img_number='1')


if __name__ == '__main__':
    main()
