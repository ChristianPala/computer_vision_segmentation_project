# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder

# Libraries:
from glob import glob
from pathlib import Path
import shutil
from typing import Union
import numpy as np
import skimage
import os
import glob
import warnings

# Import functions from other python files
from image_mask_visualizer import get_image_and_segmentation

# Global variables:
# Path to the initial dataset folder
from config import INITIAL_DATASET_PATH, TRAINING_CITY, TESTING_CITY
# Path to the training and testing datasets folders
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


def create_initial_dataset_folder_with_images_and_masks(city: str, train: bool = True) -> None:
    """
    Creates the initial dataset folder with the images from the given city
    @param city: the city to use, must be present in the dataset, we selected Aachen and Zurich for
    our training and testing datasets respectively.
    @param train: whether we are creating the training dataset or the testing dataset.
    :return: None. Populates the initial dataset folder with the images and masks from the given city
    """
    # Select training or testing dataset folder:
    im_path = Path(INITIAL_DATASET_PATH, "train") if train else Path(INITIAL_DATASET_PATH, "test")
    im_path.mkdir(exist_ok=True, parents=True)

    # if the images are already in the folder, we don't need to copy them
    if len(glob(str(Path(im_path, "*.png")))) == 0:
        for image_path in glob(f"leftImg8bit/train/{city}/*.png"):
            shutil.copy(str(image_path), str(im_path))
    # if the masks are already in the folder, we don't need to copy it
    if len(glob(str(Path(im_path, "*color.png")))) == 0:
        for image_path in glob(f"gtFine/train/{city}/*color.png"):
            shutil.copy(str(image_path), str(im_path))


def rename_images_and_masks(train: bool = True) -> None:
    """
    Renames the images and masks to a more succinct format:
    @param train: whether we are renaming the training dataset or the testing dataset.
    :return: None. Renames the images and masks
    """
    im_path = Path(INITIAL_DATASET_PATH, "train") if train else Path(INITIAL_DATASET_PATH, "test")

    # if the images are already correctly named, we don't need to rename them:
    if len(glob(str(im_path / "mask_0.png"))) == 0:
        for i, image_path in enumerate(glob(str(im_path / "*color.png"))):
            # rename the mask
            Path(image_path).rename(im_path / f"mask_{i}.png")
        for i, image_path in enumerate(glob(str(im_path / "*leftImg8bit.png"))):
            # rename the image
            Path(image_path).rename(im_path / f"image_{i}.png")


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


def count_number_of_files(path: Union[Path, str]) -> int:
    """
    Counts the number of files in a directory
    @param path: the path to the directory
    :return: the number of files in the directory
    """
    return len(glob.glob(os.path.join(path, '*image_*.png')))


def binary_mask(mask: np.ndarray) -> np.ndarray:
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
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning)  # Suppress warnings about low contrast images
        if train:
            n_images = count_number_of_files(TRAINING_DATASET_PATH)
            imgs_path = Path(TRAINING_DATASET_PATH, f'binary_mask_{label}_')
            [skimage.io.imsave(fname=f'{imgs_path}{i}.png', arr=binary_mask(get_image_and_segmentation(i, train=True)[1]))
             for i in range(n_images)]
        else:
            n_images = count_number_of_files(TESTING_DATASET_PATH)
            imgs_path = Path(TESTING_DATASET_PATH, f'binary_mask_{label}_')
            [skimage.io.imsave(fname=f'{imgs_path}{i}.png', arr=binary_mask(get_image_and_segmentation(i, train=False)[1]))
             for i in range(n_images)]


def main() -> None:
    """
    Moves the masks to the initial dataset path and saves the binary masks
    :return: None. moves the masks and saves the binary masks
    """
    # Create the initial dataset folder containing images and their full segmentation
    create_initial_dataset_folder_with_images_and_masks(city=TRAINING_CITY, train=True)
    create_initial_dataset_folder_with_images_and_masks(city=TESTING_CITY, train=False)
    rename_images_and_masks(train=True)
    rename_images_and_masks(train=False)

    # Save the binary masks into the training and testing folders
    save_binary_mask_images(label='sky', train=True)
    save_binary_mask_images(label='sky', train=False)


if __name__ == "__main__":
    main()

