# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder


from glob import glob
from pathlib import Path


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


def get_image_segmentation(image_path, segmentation_path):
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
    for i, image_path in enumerate(glob("initial_dataset/*.png")):
        if i % 2 == 1:
            Path(image_path).rename(Path("initial_dataset") / f"image_{i // 2}.png")
        else:
            Path(image=Path(image_path).rename(Path("initial_dataset") / f"mask_{i // 2}.png"))


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


if __name__ == "__main__":
    main()

