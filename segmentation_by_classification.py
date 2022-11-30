import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from typing import Union


def get_image_and_segmentation(img_number: Union[str, int]) -> (np.ndarray, np.ndarray):
    """
    Gets the image corresponding to the given number and its full segmentation.
    Args:
        img_number: Index of the image desired to be plotted
    Returns: image and segmentation as numpy ndarrays.
    """
    # Check that the given image number is a string
    img_number = str(img_number)

    # Get the paths of the desired image and segmentation
    cwd_path = os.getcwd()
    img_path = cwd_path + '/initial_dataset/' + 'image_' + img_number + '.png'
    segmentation_path = cwd_path + '/initial_dataset/' + 'mask_' + img_number + '.png'

    # Get the actual image and segmentation as numpy ndarrays
    im = skimage.io.imread(img_path)
    segmentation = skimage.io.imread(segmentation_path)
    return im, segmentation


def visualize_image_and_segmentation(img_number: Union[str, int]) -> None:
    """
    Plots side-by-side the image corresponding to the given number and its full segmentation.
    Args:
        img_number: Index of the image desired to be plotted
    Returns: None. It plots directly the corresponding images.
    """
    # Get the actual image and segmentation as numpy ndarrays
    im, segmentation = get_image_and_segmentation(img_number)

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
