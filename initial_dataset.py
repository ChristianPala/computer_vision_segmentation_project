# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder

# Libraries:
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np
import skimage


def create_initial_dataset_folder_with_aachen_images() -> None:
    """
    Creates the initial dataset folder with the images from Aachen
    :return: None. Creates the initial dataset folder
    """
    Path("initial_dataset").mkdir(exist_ok=True)
    # if the images are already in the folder, we don't need to copy them
    if len(glob("initial_dataset/*.png")) == 0:
        for image_path in glob("leftImg8bit/train/aachen/*.png"):
            Path(image_path).rename(Path("initial_dataset") / Path(image_path).name)


def get_image_segmentation(image_path, segmentation_path) -> Optional[Path]:
    """Get the segmentation for the image.

    Args:
        image_path (str): Path to the image
        segmentation_path (str): Path to the segmentation

    Returns:
        str: Path to the segmentation
    """
    # if the color.png is already in the folder, we don't need to copy it
    if len(glob("initial_dataset/*color.png")) == 0:
        image_name = Path(image_path).name
        segmentation_name = image_name.replace("_leftImg8bit", "_gtFine_color")
        segmentation = Path(segmentation_path) / segmentation_name
        return segmentation


def rename_images_and_masks() -> None:
    """
    Renames the images and masks to a more succinct format:
    """
    Path("initial_dataset").mkdir(exist_ok=True)

    # if the images are already correctly named, we don't need to rename them:
    if len(glob("initial_dataset/image_0.png")) == 0:
        for i, image_path in enumerate(glob("initial_dataset/*.png")):
            if i % 2 == 1:
                Path(image_path).rename(Path("initial_dataset") / f"image_{i // 2}.png")
            else:
                Path(image=Path(image_path).rename(Path("initial_dataset") / f"mask_{i // 2}.png"))


def binary_mask(mask) -> np.ndarray:
    """
    Converts the mask to a binary mask, keeping only the sky class as 1 and the rest as 0
    :param mask: the mask to convert
    :return: the binary mask
    """
    # from the docs, sky has the following RGB values: 70,130,180.
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    # sky mask: 70,130,180
    sky = np.all(mask == [70, 130, 180], axis=-1)
    # set to 1 the sky pixels and 0 the rest
    mask[sky] = [1., 1., 1.]
    mask[~sky] = [0., 0., 0.]
    return mask


def main() -> None:
    """
    Moves the masks to the initial dataset path
    :return: None. moves the masks
    """
    create_initial_dataset_folder_with_aachen_images()

    for image_path in glob("initial_dataset/*.png"):
        segmentation_path = get_image_segmentation(image_path, "gtFine/train/aachen")
        if segmentation_path is None:
            break
        Path(segmentation_path).rename(Path("initial_dataset") / Path(segmentation_path).name)

    rename_images_and_masks()

    for mask_path in glob("initial_dataset/*.png"):
        mask = skimage.io.imread(mask_path)
        mask = binary_mask(mask)
        skimage.io.imsave(mask_path, mask)


if __name__ == "__main__":
    main()

