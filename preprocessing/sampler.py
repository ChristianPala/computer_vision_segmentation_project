# Auxiliary library to sample pixels from the dataset
# Libraries:
# Data Manipulation:
import os
from pathlib import Path
import numpy as np
from glob import glob
import skimage
import pandas as pd
# Randomness:
from random import randint
# Plotting:
from matplotlib import pyplot as plt
# Progress bar:
from tqdm import tqdm

# Global variables:
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH, VALIDATION_DATASET_PATH, PATCH_SIZE


def get_images_and_binary_masks(train: bool = True) -> (list, list):
    """
    Auxiliary function to get the images and the binary masks from the dataset.
    @param train: bool: True if the image is from the training dataset, False if it is from the testing dataset
    :return:
    """
    # get the correct path:
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # get all the images:
    images = []
    for image_path in glob(os.path.join(path, 'image_*.png')):
        images.append(skimage.io.imread(image_path))

    # get all the grayscale binary masks in the path:
    binary_masks = []
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))
    # cast the binary mask to grayscale
    binary_masks = [skimage.color.rgb2gray(mask) for mask in binary_masks]

    return images, binary_masks


def get_val_images_and_binary_masks() -> (list, list):
    """
    Auxiliary function to get the images and the binary masks from the validation dataset.
    :return: the images and the binary masks from the validation dataset.
    """
    # get the correct path:
    path = VALIDATION_DATASET_PATH

    # get all the images:
    images = []
    for image_path in glob(os.path.join(path, 'image_*.png')):
        images.append(skimage.io.imread(image_path))

    # get all the grayscale binary masks in the path:
    binary_masks = []
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    binary_masks = [skimage.color.rgb2gray(mask) for mask in binary_masks]

    return images, binary_masks


def compute_validation_proportions() -> (int, int):
    """
    Computes the total number of pixels in the sky class in the validation dataset
    :return: the total number of pixels in the sky class and the total number of pixels in the non-sky class
    for the validation dataset.
    """
    # get the binary masks:
    binary_masks = get_val_images_and_binary_masks()[1]

    # compute the total number of pixels in the sky class
    total_sky_pixels = 0
    total_non_sky_pixels = 0
    for mask in binary_masks:
        total_sky_pixels += np.sum(mask == 1)
        total_non_sky_pixels += np.sum(mask != 1)

    # compute the sampling proportions to sample for each images
    return total_sky_pixels, total_non_sky_pixels


def validation_pixel_sampler(total_count: int) -> pd.DataFrame:
    """
    Sample pixels from the validation dataset.
    @param total_count: the total number of pixels to sample.
    :return: None. Saves the sampled pixels in a csv file.
    """
    # get the images and the binary masks:
    images, binary_masks = get_val_images_and_binary_masks()

    # compute the sampling proportions:
    total_sky_pixels, total_non_sky_pixels = compute_validation_proportions()

    sky_sample = total_count / total_sky_pixels
    non_sky_sample = total_count / total_non_sky_pixels

    # create the dataset
    df = pd.DataFrame(columns=['image_nr', 'r', 'g', 'b', 'x', 'y', 'class'])

    # sample the pixels
    image_count = 0
    for image, mask in zip(images, binary_masks):
        sky_row, sky_column, sky_r, sky_g, sky_b, non_sky_row, non_sky_column, non_sky_r, non_sky_g, non_sky_b = \
            binary_sampler(image, mask, sky_sample, non_sky_sample)

        # create the sky dataframe
        sky_df = pd.DataFrame(
            {'image_nr': image_count, 'r': sky_r, 'g': sky_g, 'b': sky_b, 'x': sky_column, 'y': sky_row, 'class': 1})

        # create the non-sky dataframe
        non_sky_df = pd.DataFrame(
            {'image_nr': image_count, 'r': non_sky_r, 'g': non_sky_g, 'b': non_sky_b, 'x': non_sky_column,
             'y': non_sky_row, 'class': 0})

        # concatenate the dataframes
        df = pd.concat([df, sky_df, non_sky_df], ignore_index=True)

        # increment the image count
        image_count += 1

    return df


def compute_sampling_proportions(train: bool = True) -> (int, int):
    """
    Computes the total number of pixels in the sky class in the training dataset
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the total number of pixels in the sky class
    """
    # get the binary masks:
    binary_masks = get_images_and_binary_masks(train=train)[1]

    # compute the total number of pixels in the sky class
    total_sky_pixels = 0
    total_non_sky_pixels = 0
    for mask in binary_masks:
        total_sky_pixels += np.sum(mask == 1)
        total_non_sky_pixels += np.sum(mask != 1)

    # compute the sampling proportions to sample for each images
    return total_sky_pixels, total_non_sky_pixels


def binary_sampler(image: np.ndarray, binary_mask: np.ndarray, sky_fraction: float, non_sky_fraction: float) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray):
    """
    Samples the rows, columns, and rgb values of an image for the sky and non-sky classes.
    @param image: np.ndarray: the image to sample
    @param binary_mask: np.ndarray: the binary mask of the image
    @param sky_fraction: float: the fraction of the sky pixels to sample
    @param non_sky_fraction: float: the fraction of the non-sky pixels to sample
    :return: the rows, columns, and rgb values of the sampled pixels for the sky and non-sky classes.
    """
    sky_pixels = np.argwhere(binary_mask == 1)
    # select the row index of the sky pixels
    sky_row = sky_pixels[:, 0]
    # select the column index of the sky pixels
    sky_column = sky_pixels[:, 1]
    # create tuples of the row and column index of the sky pixels
    sky_pixels = list(zip(sky_row, sky_column))
    # sample the sky pixels
    sky_pixels_index = np.random.choice(len(sky_pixels), int(len(sky_pixels) * sky_fraction), replace=False)
    sky_pixels = [sky_pixels[i] for i in sky_pixels_index]
    # get the row and column index of the sky pixels
    sky_row = [pixel[0] for pixel in sky_pixels]
    sky_column = [pixel[1] for pixel in sky_pixels]
    # get the r, g, b values of the sampled pixels
    sky_r = image[sky_row, sky_column, 0]
    sky_g = image[sky_row, sky_column, 1]
    sky_b = image[sky_row, sky_column, 2]

    # get the non-sky pixels
    non_sky_pixels = np.argwhere(binary_mask == 0)
    # select the row index of the non-sky pixels
    non_sky_row = non_sky_pixels[:, 0]
    # select the column index of the non-sky pixels
    non_sky_column = non_sky_pixels[:, 1]
    # create tuples of the x and y index of the non-sky pixels
    non_sky_pixels = list(zip(non_sky_row, non_sky_column))
    # sample the non-sky pixels
    non_sky_pixels_index = np.random.choice(len(non_sky_pixels), int(len(non_sky_pixels) * non_sky_fraction),
                                            replace=False)
    non_sky_pixels = [non_sky_pixels[i] for i in non_sky_pixels_index]
    # get the row and column index of the non-sky pixels
    non_sky_row = [pixel[0] for pixel in non_sky_pixels]
    non_sky_column = [pixel[1] for pixel in non_sky_pixels]
    # get the r, g, b values of the sampled pixels
    non_sky_r = image[non_sky_row, non_sky_column, 0]
    non_sky_g = image[non_sky_row, non_sky_column, 1]
    non_sky_b = image[non_sky_row, non_sky_column, 2]
    return sky_row, sky_column, sky_r, sky_g, sky_b, non_sky_row, non_sky_column, non_sky_r, non_sky_g, non_sky_b


def pixel_sampler(total_count: int, train: bool = True) -> pd.DataFrame:
    """
    Sample a balanced dataset, from two classes, from the images in the dataset.
    @param total_count: the total number of pixels to sample
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the sampled pixels
    """

    # get the images and the binary masks:
    images, binary_masks = get_images_and_binary_masks(train=train)

    # compute the sampling proportions of the two classes in the dataset:
    sky_count, non_sky_count = compute_sampling_proportions(train)
    sky_sample = total_count / sky_count
    non_sky_sample = total_count / non_sky_count

    # create the dataset
    df = pd.DataFrame(columns=['image_nr', 'r', 'g', 'b', 'x', 'y', 'class'])

    # sample the pixels
    image_count = 0
    for image, mask in zip(images, binary_masks):
        sky_row, sky_column, sky_r, sky_g, sky_b, non_sky_row, non_sky_column, non_sky_r, non_sky_g, non_sky_b = \
            binary_sampler(image, mask, sky_sample, non_sky_sample)

        # create the sky dataframe
        sky_df = pd.DataFrame(
            {'image_nr': image_count, 'r': sky_r, 'g': sky_g, 'b': sky_b, 'x': sky_column, 'y': sky_row, 'class': 1})

        # create the non-sky dataframe
        non_sky_df = pd.DataFrame(
            {'image_nr': image_count, 'r': non_sky_r, 'g': non_sky_g, 'b': non_sky_b, 'x': non_sky_column,
             'y': non_sky_row, 'class': 0})

        # concatenate the dataframes
        df = pd.concat([df, sky_df, non_sky_df], ignore_index=True)

        # increment the image count
        image_count += 1

    return df


def plot_binary_mask_and_sampled_pixels(pixel_dataframe: pd.DataFrame,
                                        image_nr: int, train: bool = True) -> None:
    """
    Auxiliary function to plot the binary mask and the sampled pixels
    @param pixel_dataframe: pd.DataFrame: the dataframe containing the sampled pixels
    @param image_nr: the image number: int: the image number in the dataset
    @param train: bool: True if the image is from the training dataset, False if it is from the testing dataset
    :return: None. Plots the binary mask and the sampled pixels and saves the plot to the plots' folder.
    """
    # select and create the path:
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH
    plots_path = Path(os.path.join(path, 'plots'))
    plots_path.mkdir(parents=True, exist_ok=True)

    name = 'training' if train else 'testing'

    # get the corresponding grayscale binary mask:
    binary_mask = skimage.io.imread(os.path.join(path, f'binary_mask_sky_{image_nr}.png'))

    # get the row of the image number:
    pixels: pd.DataFrame = pixel_dataframe[pixel_dataframe['image_nr'] == image_nr]

    # plot the binary mask
    plt.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    # plot the sampled pixels, plot colorblind friendly colors
    plt.scatter(x=pixels['x'], y=pixels['y'], c=pixels['class'], cmap='spectral', s=1, alpha=0.8)
    # add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=['sky', 'non-sky'],
               title='Pixel Class')
    # add a title
    plt.title(f'Sampled pixels over the binary mask, extracted from {name} image: {image_nr}.')
    # save the plot
    plt.savefig(Path(plots_path, f'sampled_pixels_from_image_{image_nr}.png'))
    # show the plot
    plt.show()


def sampler_visual_inspector(training: bool = True) -> None:
    """
    Visually inspect the sampler to make sure it is working properly.
    @param training: bool: True if the image is from the training dataset, False if it is from the testing dataset
    :return: None. Displays the sampled pixels on the first image in the dataset, as a sanity check.
    """
    # get the correct path:
    path = TRAINING_DATASET_PATH if training else TESTING_DATASET_PATH
    name = 'train' if training else 'test'
    df: pd.DataFrame = pd.read_csv(Path(path, f'{name}_by_pixel.csv'))
    # plot the binary mask and the sampled pixels
    plot_binary_mask_and_sampled_pixels(pixel_dataframe=df, image_nr=0, train=training)


def dataset_explorer(dataframe: pd.DataFrame, sampling_type: str, train: bool = True) -> None:
    """
    Function to print information on the generated dataset
    @param dataframe: the dataframe to explore
    @param sampling_type: the type of sampling used to generate the dataset, either 'pixel' or 'patch'.
    @param train: whether the dataframe is the training dataset or not
    :return: None. Prints some information.
    """
    name = "Training" if train else "Testing"

    sample_type = "pixels" if sampling_type == "pixel" else "patches"

    # print the number of rows and columns
    print("*" * 50)
    print(f"{name} dataframe sampled by {sampling_type} has {dataframe.shape[0]} rows and "
          f"{dataframe.shape[1]} columns.")
    # print the number of sky and non-sky pixels
    print(f"{name} dataframe sampled by {sampling_type} has {dataframe[dataframe['class'] == 1].shape[0]} "
          f"sky {sample_type} and "
          f"{dataframe[dataframe['class'] == 0].shape[0]} non-sky {sampling_type}.")

    if sampling_type == "pixel":
        # print the number of sky and non-sky pixels per image
        print(f"{name} dataframe sampled by {sampling_type} has approximately "
              f"{int(dataframe.groupby('image_nr')['class'].sum().mean())} "
              f"sky {sample_type} per image and approximately "
              f"{int(dataframe.groupby('image_nr')['class'].count().mean() - dataframe.groupby('image_nr')['class'].sum().mean())} "
              f"non-sky {sample_type} per image.")

    elif sampling_type == "patch":
        # print the patch size
        print(f"{name} dataframe sampled by {sampling_type} has {dataframe['patch'].shape[0]} patches of size "
              f"{dataframe['patch'].iloc[0].shape[0]}x{dataframe['patch'].iloc[0].shape[0]}.")
    # print the number of images
    print(f"{name} dataframe sampled by {sampling_type} has {dataframe['image_nr'].nunique()} images.")
    print("*" * 50)


def has_sky_delta(binary_mask: np.ndarray, delta: int = 256) -> bool:
    """
    Auxiliary function to check if a binary mask has sky pixels.
    @param binary_mask: the binary mask to check the presence of sky pixels.
    @param delta: the number of pixels to check around the center of the image.
    :return: bool: True if the image has sky pixels, False otherwise.
    """
    # get the mask only inside the delta for the decision:
    binary_mask = binary_mask[delta:-delta, delta:-delta]
    sky_pixels = np.argwhere(binary_mask == 1)
    return True if len(sky_pixels > 0) else False


def get_patches(image: np.ndarray, binary_mask: np.ndarray, n_patches: int = 6, delta: int = PATCH_SIZE // 2) \
        -> ([np.ndarray], [np.ndarray], [int], [int], [int]):
    patches: [np.ndarray] = []
    mask_labels: [np.ndarray] = []
    classes: [int] = []
    centers_row: [int] = []
    centers_column: [int] = []
    sky_count, non_sky_count = 0, 0
    for i in range(n_patches):
        while sky_count < 3:  # Get 3 sky patches
            center_row = randint(delta, image.shape[0] - delta)
            center_column = randint(delta, image.shape[1] - delta)

            class_value = binary_mask[center_row, center_column]
            if class_value == 1:
                classes.append(class_value)
                patch = image[center_row - delta:center_row + delta, center_column - delta:center_column + delta, :]
                mask_label = binary_mask[center_row - delta:center_row + delta,
                             center_column - delta:center_column + delta]
                centers_row.append(center_row)
                centers_column.append(center_column)
                patches.append(patch)
                mask_labels.append(mask_label)
                sky_count += 1

        while non_sky_count < 3:  # Get 3 non-sky patches
            center_row = randint(delta, image.shape[0] - delta)
            center_column = randint(delta, image.shape[1] - delta)

            class_value = binary_mask[center_row, center_column]
            if class_value == 0:
                classes.append(class_value)
                patch = image[center_row - delta: center_row + delta, center_column - delta: center_column + delta, :]
                mask_label = binary_mask[center_row - delta: center_row + delta,
                             center_column - delta: center_column + delta]
                centers_row.append(center_row)
                centers_column.append(center_column)
                patches.append(patch)
                mask_labels.append(mask_label)
                non_sky_count += 1

    return patches, mask_labels, classes, centers_row, centers_column


def patch_sampler(train: bool = True) -> pd.DataFrame:
    # get all the images in the path:
    images, binary_masks = get_images_and_binary_masks(train=train)

    df = pd.DataFrame(columns=['image_nr', 'patch', 'center_x', 'center_y', 'class', 'mask_label'])
    img_num: int = 0
    for img, mask in tqdm(zip(images, binary_masks), desc='Images', total=len(images)):
        if not has_sky_delta(mask):
            img_num += 1# Skip images with no sky
            continue

        patches, mask_labels, classes, centers_row, centers_column = get_patches(img, mask)

        temp_df = pd.DataFrame({'image_nr': img_num, 'patch': patches, 'mask_label': mask_labels,
                                'center_x': centers_column, 'center_y': centers_row, 'class': classes})
        df = pd.concat([df, temp_df], ignore_index=True)

        img_num += 1

    return df


def val_patch_sampler() -> pd.DataFrame:
    # get all the images in the path:
    images, binary_masks = get_val_images_and_binary_masks()

    df = pd.DataFrame(columns=['image_nr', 'patch', 'center_x', 'center_y', 'class', 'mask_label'])

    img_num: int = 0
    for img, mask in tqdm(zip(images, binary_masks), desc='Images', total=len(images)):
        if not has_sky_delta(mask):
            img_num += 1# Skip images with no sky
            continue

        patches, mask_labels, classes, centers_row, centers_column = get_patches(img, mask)

        temp_df = pd.DataFrame({'image_nr': img_num, 'patch': patches, 'mask_label': mask_labels,
                                'center_x': centers_column, 'center_y': centers_row, 'class': classes})
        df = pd.concat([df, temp_df], ignore_index=True)

        img_num += 1

    return df


def main() -> None:
    """
    Main function to sample pixels from the training dataset, and save them to a csv file.
    Confirms that the sampling is working properly by plotting the sampled pixels over the binary mask.
    Gives some information on the generated dataset.
    :return: None. Saves the sampled datasets to csv files and displays some information about them.
    """
    # Sample amount of pixels per class from the training and testing dataset for the pixel classifier:
    training_pixels: int = 15000
    testing_pixels: int = 5000

    # Sample 15000 pixels from the training dataset:
    train_pixels_df: pd.DataFrame = pixel_sampler(total_count=training_pixels, train=True)
    # Sample 5000 pixels from the validation dataset:
    val_pixels_df: pd.DataFrame = validation_pixel_sampler(total_count=testing_pixels)
    # Sample 5000 pixels from the testing dataset:
    test_pixels_df: pd.DataFrame = pixel_sampler(total_count=testing_pixels, train=False)

    # Save the sampled pixels:
    train_pixels_df.to_csv(Path(TRAINING_DATASET_PATH, 'train_by_pixel.csv'), index=False)
    val_pixels_df.to_csv(Path(VALIDATION_DATASET_PATH, 'val_by_pixel.csv'), index=False)
    test_pixels_df.to_csv(Path(TESTING_DATASET_PATH, 'test_by_pixel.csv'), index=False)

    # Inspect the sampled pixels:
    sampler_visual_inspector(training=True)

    # Explore the pixel datasets:
    dataset_explorer(dataframe=train_pixels_df, sampling_type="pixel", train=True)
    dataset_explorer(dataframe=val_pixels_df, sampling_type="pixel", train=False)
    dataset_explorer(dataframe=test_pixels_df, sampling_type="pixel", train=False)

    # Sample patches from the training and testing dataset for the patch classifier:
    train_patches_df = patch_sampler(train=True)
    val_patches_df = val_patch_sampler()
    test_patches_df = patch_sampler(train=False)

    # Save the sampled patches as pickle files to preserve the image data:
    train_patches_df.to_pickle(Path(TRAINING_DATASET_PATH, 'train_by_patch.pkl'))
    val_patches_df.to_pickle(Path(VALIDATION_DATASET_PATH, 'val_by_patch.pkl'))
    test_patches_df.to_pickle(Path(TESTING_DATASET_PATH, 'test_by_patch.pkl'))

    # Explore the patch datasets:
    dataset_explorer(dataframe=train_patches_df, sampling_type="patch", train=True)
    dataset_explorer(dataframe=val_patches_df, sampling_type="patch", train=False)
    dataset_explorer(dataframe=test_patches_df, sampling_type="patch", train=False)


# Driver Code
if __name__ == '__main__':
    main()
