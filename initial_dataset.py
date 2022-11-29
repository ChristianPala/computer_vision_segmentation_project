# start with a sample of 174 images from Aachen for our dataset:
# We need to get the matching masks in the gtFine folder


from glob import glob
from pathlib import Path


def get_image_segmentation(image_path, segmentation_path):
    """Get the segmentation for the image.

    Args:
        image_path (str): Path to the image
        segmentation_path (str): Path to the segmentation

    Returns:
        str: Path to the segmentation
    """
    image_name = Path(image_path).name
    segmentation_name = image_name.replace("_leftImg8bit", "_gtFine_color")
    segmentation = Path(segmentation_path) / segmentation_name
    return segmentation


def main() -> None:
    """
    Moves the masks to the initial dataset path
    :return: None. moves the masks
    """
    for image_path in glob("initial_dataset/*.png"):
        segmentation_path = get_image_segmentation(image_path, "gtFine/train/aachen")
        Path(segmentation_path).rename(Path("initial_dataset") / Path(segmentation_path).name)


if __name__ == "__main__":
    main()

