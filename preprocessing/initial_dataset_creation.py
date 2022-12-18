# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder
import warnings
# Libraries:
# Data manipulation:
from glob import glob
from pathlib import Path
import shutil
import numpy as np

# Global variables:
# Path to the initial dataset folders
from config import INITIAL_DATASET_PATH, LEFT_IMG_8_BIT_PATH, GT_FINE_PATH, TRAINING_CITY, VALIDATION_CITY, TESTING_CITY


def is_populated(path: Path) -> bool:
    """
    Checks if the folder is already populated
    @param path: the path to check
    :return: bool: whether the folder is populated or not
    """
    return len(list(path.iterdir())) != 0


def create_initial_dataset_folder_with_images_and_masks(city: str, train: bool = True, val_city: str = None) \
        -> None:
    """
    Creates the initial dataset folder with the images from the given city
    @param city: the city to use, must be present in the dataset, we selected Aachen and Zurich for
    our training and testing datasets respectively.
    @param train: bool: default (True) whether we are creating the training dataset or the testing dataset.
    @param val_city: str: if the validation dataset is to be created, the city to use for the validation dataset.
    :return: None. Populates the initial dataset folder with the images and masks from the given city
    """

    # Select the path where the images and masks will be copied to, depending on whether we are creating
    # the training dataset or the testing dataset
    im_path = Path(INITIAL_DATASET_PATH, "train") if train else Path(INITIAL_DATASET_PATH, "test")
    im_path.mkdir(parents=True, exist_ok=True)

    # We will use the city of Aachen for our training dataset and the city of Zürich for our testing dataset
    # both are present in the training dataset
    sub_folder = "train"

    # if the folder is already populated, delete it and create a new one:
    if is_populated(im_path):
        shutil.rmtree(im_path)
        im_path.mkdir(parents=True, exist_ok=True)

    # Copy the training or testing images to the initial dataset folder:
    for image_path in glob(str(Path(LEFT_IMG_8_BIT_PATH, sub_folder, city, "*.png"))):
        shutil.copy(str(image_path), str(im_path))

    # Copy the training or testing masks to the initial dataset folder:
    for image_path in glob(str(Path(GT_FINE_PATH, sub_folder, city, "*color.png"))):
        shutil.copy(str(image_path), str(im_path))

    # if a validation city is provided, we create the validation dataset:
    if val_city:
        sub_folder = "val"
        val_path = Path(INITIAL_DATASET_PATH, "val")
        val_path.mkdir(exist_ok=True, parents=True)

        # if the folder is already populated, delete it and create a new one:
        if is_populated(val_path):
            shutil.rmtree(val_path)
            val_path.mkdir(parents=True, exist_ok=True)

        for image_path in glob(str(Path(LEFT_IMG_8_BIT_PATH, sub_folder, val_city, "*.png"))):
            shutil.copy(str(image_path), str(val_path))

        for image_path in glob(str(Path(GT_FINE_PATH, sub_folder, val_city, "*color.png"))):
            shutil.copy(str(image_path), str(val_path))


def rename_images_and_masks(train: bool = True, val: bool = True) -> None:
    """
    Renames the images and masks to a more succinct format:
    @param train: whether we are renaming the training dataset or the testing dataset.
    @param val: whether we are renaming the validation dataset or not.
    :return: None. Renames the images and masks
    """
    im_path = Path(INITIAL_DATASET_PATH, "train") if train else Path(INITIAL_DATASET_PATH, "test")

    # Rename the images and masks:
    for i, image_path in enumerate(glob(str(im_path / "*leftImg8bit.png"))):
        # rename the image
        Path(image_path).rename(im_path / f"image_{i}.png")
    # rename the mask
    for i, image_path in enumerate(glob(str(im_path / "*color.png"))):
        # rename the mask
        Path(image_path).rename(im_path / f"mask_{i}.png")

    # rename the validation images and masks:
    if val:
        im_path = Path(INITIAL_DATASET_PATH, "val")

        for i, image_path in enumerate(glob(str(im_path / "*leftImg8bit.png"))):
            # rename the image
            Path(image_path).rename(im_path / f"image_{i}.png")

        for i, image_path in enumerate(glob(str(im_path / "*color.png"))):
            # rename the mask
            Path(image_path).rename(im_path / f"mask_{i}.png")


def main() -> None:
    """
    Moves the masks to the initial dataset path
    :return: None. moves the masks
    """
    create_initial_dataset_folder_with_images_and_masks(city=TRAINING_CITY, val_city=VALIDATION_CITY)
    create_initial_dataset_folder_with_images_and_masks(city=TESTING_CITY, train=False)
    rename_images_and_masks(train=True, val=True)
    rename_images_and_masks(train=False)


if __name__ == "__main__":
    main()

