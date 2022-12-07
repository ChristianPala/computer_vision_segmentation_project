# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder
import warnings
# Libraries:
from glob import glob
from pathlib import Path
import shutil
import numpy as np


# Global variables:
# Path to the initial dataset folder
from config import INITIAL_DATASET_PATH, TRAINING_CITY, TESTING_CITY


def create_initial_dataset_folder_with_images_and_masks(city: str, train: bool = True) -> None:
    """
    Creates the initial dataset folder with the images from the given city
    @param city: the city to use, must be present in the dataset, we selected Aachen and Zurich for
    our training and testing datasets respectively.
    @param train: whether we are creating the training dataset or the testing dataset.
    :return: None. Populates the initial dataset folder with the images and masks from the given city
    """
    # Select training or testing dataset folder:
    im_path = Path(INITIAL_DATASET_PATH,"train") if train else Path(INITIAL_DATASET_PATH,"test")
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


def main() -> None:
    """
    Moves the masks to the initial dataset path
    :return: None. moves the masks
    """
    create_initial_dataset_folder_with_images_and_masks(city=TRAINING_CITY, train=True)
    create_initial_dataset_folder_with_images_and_masks(city=TESTING_CITY, train=False)
    rename_images_and_masks(train=True)
    rename_images_and_masks(train=False)


if __name__ == "__main__":
    main()

