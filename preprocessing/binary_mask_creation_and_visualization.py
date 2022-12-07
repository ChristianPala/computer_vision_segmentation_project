# Libraries:
import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage
from typing import Union
import glob


# Global variables:
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


def count_number_of_files(path: Union[Path, str]) -> int:
    """
    Counts the number of files in a directory
    @param path: the path to the directory
    :return: the number of files in the directory
    """
    return len(glob.glob(os.path.join(path, '*image_*.png')))


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
    :return: None. It plots directly the corresponding images.
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


def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Removes the alpha channel from the image
    @param image: the image to remove the alpha channel from
    :return: the image without the alpha channel
    """
    if image.shape[-1] == 4:
        return image[..., :3]
    else:
        return image


def binary_mask(mask) -> np.ndarray:
    """
    Converts the mask to a binary mask, keeping only the sky class as 1 and the rest as 0
    @param: param mask: the mask to convert
    :return: the binary mask
    """
    # from the docs, sky has the following RGB values: 70,130,180.
    # we noticed the images are saved with 4 channels, so we need to ignore the alpha channel.
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    # Remove the alpha channel from the segmentation image
    mask = remove_alpha_channel(mask)

    # Compute the sky mask: 70, 130, 180
    sky = ((mask[:, :, 0] == 70) & (mask[:, :, 1] == 130) & (mask[:, :, 2] == 180))

    # set to 1 the sky pixels and 0 the rest, ignoring the alpha channel.
    mask[sky, :] = 255
    mask[~sky, :] = 0
    return mask


def save_binary_mask_images(label: str = 'sky', train: bool = True) -> None:
    """
    Saves the binary mask images
    @param: label: The label of the element that is wanted to be selected.
    @param: train: Flag to define whether the binary masks are for training or testing.
    :return: None. It directly saves the images in the desired path.
    """
    if train:
        n_images = count_number_of_files(TRAINING_DATASET_PATH)
        imgs_path = Path(TRAINING_DATASET_PATH, f'binary_mask_{label}_')
        [skimage.io.imsave(fname=f'{imgs_path}{i}.png', arr=binary_mask(get_image_and_segmentation(i, train=True)[1]))
         for i in range(n_images)]
    else:
        n_images = count_number_of_files(TESTING_DATASET_PATH)
        imgs_path = Path(TESTING_DATASET_PATH, f'binary_mask_{label}_')
        [skimage.io.imsave(fname=f'{imgs_path}{i}.png',
                           arr=binary_mask(get_image_and_segmentation(i, train=False)[1]))
         for i in range(n_images)]


def main():
    # Visualize the images and their segmentation
    visualize_image_and_segmentation(img_number='1')

    # Save the binary masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        save_binary_mask_images(label='sky', train=True)
        save_binary_mask_images(label='sky', train=False)


if __name__ == '__main__':
    main()
