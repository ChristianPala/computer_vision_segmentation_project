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


def binary_mask(mask) -> np.ndarray:
    """
    Converts the mask to a binary mask, keeping only the sky class as 1 and the rest as 0
    :param mask: the mask to convert
    :return: the binary mask
    """
    # from the docs, sky has the following RGB values: 70,130,180.
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    # sky mask: 70,130,180
    sky = ((mask[:, :, 0] == 70) & (mask[:, :, 1] == 130) & (mask[:, :, 2] == 180))

    # set to 1 the sky pixels and 0 the rest
    mask[sky, 0] = 255
    mask[sky, 1] = 255
    mask[sky, 2] = 255
    mask[~sky, 0] = 0
    mask[~sky, 1] = 0
    mask[~sky, 2] = 0
    return mask/255


def save_binary_mask_images():
    binary_masks = [binary_mask(get_image_and_segmentation(i)[1]) for i in range(174)]
    #skimage.io.imsave()


def main():
    visualize_image_and_segmentation(img_number='1')
    img1, segm1 = get_image_and_segmentation('1')

    file_name = 'binary_sky_mask'
    binary_mask1 = binary_mask(segm1)
    plt.imshow(binary_mask1, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    main()
