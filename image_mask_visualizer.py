# Libraries:
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage
from typing import Union

# Global variables:
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
    @param: img_number: Index of the image desired to be plotted.
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


def binary_mask(mask) -> np.ndarray:
    """
    Converts the mask to a binary mask, keeping only the sky class as 1 and the rest as 0
    @param: param mask: the mask to convert
    :return: the binary mask
    """
    # from the docs, sky has the following RGB values: 70,130,180.
    # we noticed the images are saved with 4 channels, so we need to ignore the alpha channel.
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    # sky mask: 70, 130, 180
    sky = ((mask[:, :, 0] == 70) & (mask[:, :, 1] == 130) & (mask[:, :, 2] == 180))

    # set to 1 the sky pixels and 0 the rest, ignoring the alpha channel.
    mask[sky, :] = np.array([255, 255, 255, 255])
    mask[~sky, :] = np.array([0, 0, 0, 255])
    return mask/255


def save_binary_mask_images(label: str = 'sky', imgs_path: str = ''):
    """
    Saves the binary mask images
    @param: label: The label of the element that is wanted to be selected.
    @param: imgs_path: Desired path where the binary mask images will be saved.
    :return: None. It directly saves the images in the desired path.
    """
    n_images = 174
    imgs_path += f'binary_mask_{label}_'
    [skimage.io.imsave(fname=f'{imgs_path}{i}', arr=binary_mask(get_image_and_segmentation(i)[1]))
     for i in range(n_images)]


def main():
    visualize_image_and_segmentation(img_number='1')
    img1, segm1 = get_image_and_segmentation('1')

    file_name = 'binary_sky_mask'
    binary_mask1 = binary_mask(segm1)
    plt.imshow(binary_mask1, vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    main()
