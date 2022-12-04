# Auxiliary library to sample pixels from the dataset
# Libraries:
# Data Manipulation:
import os
from pathlib import Path
import numpy as np
from glob import glob
import skimage
import pandas as pd
from matplotlib import pyplot as plt

# Global variables:
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


def compute_sampling_proportions(train: bool = True) -> (int, int):
    """
    Computes the total number of pixels in the sky class in the training dataset
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the total number of pixels in the sky class
    """
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # get all the binary masks in the path:
    binary_masks = []
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # compute the total number of pixels in the sky class
    total_sky_pixels = 0
    total_non_sky_pixels = 0
    for mask in binary_masks:
        total_sky_pixels += np.sum(mask == 255)
        total_non_sky_pixels += np.sum(mask != 255)

    # compute the sampling proportions to sample for each images
    return total_sky_pixels, total_non_sky_pixels


def sample_pixels(total_count: int, train: bool = True) -> pd.DataFrame:
    """
    Sample a balanced dataset, from two classes, from the images in the dataset.
    @param total_count: the total number of pixels to sample
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the sampled pixels
    """
    path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH

    # compute the sampling proportions to sample for each images
    sky_count, non_sky_count = compute_sampling_proportions(train)

    sky_sample = total_count / sky_count
    non_sky_sample = total_count / non_sky_count

    # create the training dataset
    df = pd.DataFrame(columns=['image', 'r', 'g', 'b', 'x', 'y', 'class'])

    # get all the images in the path:
    images = []
    for image_path in glob(os.path.join(path, 'image_*.png')):
        images.append(skimage.io.imread(image_path))

    # get all the binary masks in the path:
    binary_masks = []
    for mask_path in glob(os.path.join(path, 'binary_mask_*.png')):
        binary_masks.append(skimage.io.imread(mask_path))

    # cast the binary mask to grayscale
    binary_masks = [skimage.color.rgb2gray(mask) for mask in binary_masks]

    # sample the pixels
    image_count = 0
    for image, mask in zip(images, binary_masks):
        # get the sky pixels
        sky_pixels = np.argwhere(mask == 1)
        # select the x index of the sky pixels
        sky_x = sky_pixels[:, 0]
        # select the y index of the sky pixels
        sky_y = sky_pixels[:, 1]
        # create tuples of the x and y index of the sky pixels
        sky_pixels = list(zip(sky_x, sky_y))
        # sample the sky pixels
        sky_pixels_index = np.random.choice(len(sky_pixels), int(len(sky_pixels) * sky_sample), replace=False)
        sky_pixels = [sky_pixels[i] for i in sky_pixels_index]
        # get the x and y coordinates of the sampled tuples
        sky_x = [pixel[0] for pixel in sky_pixels]
        sky_y = [pixel[1] for pixel in sky_pixels]
        # get the r, g, b values of the sampled pixels
        sky_r = image[sky_x, sky_y, 0]
        sky_g = image[sky_x, sky_y, 1]
        sky_b = image[sky_x, sky_y, 2]

        # get the non-sky pixels
        non_sky_pixels = np.argwhere(mask == 0)
        # select the x index of the non-sky pixels
        non_sky_x = non_sky_pixels[:, 0]
        # select the y index of the non-sky pixels
        non_sky_y = non_sky_pixels[:, 1]
        # create tuples of the x and y index of the non-sky pixels
        non_sky_pixels = list(zip(non_sky_x, non_sky_y))
        # sample the non-sky pixels
        non_sky_pixels_index = np.random.choice(len(non_sky_pixels), int(len(non_sky_pixels) * non_sky_sample),
                                                replace=False)
        non_sky_pixels = [non_sky_pixels[i] for i in non_sky_pixels_index]
        # get the x and y coordinates of the sampled tuples
        non_sky_x = [pixel[0] for pixel in non_sky_pixels]
        non_sky_y = [pixel[1] for pixel in non_sky_pixels]
        # get the r, g, b values of the sampled pixels
        non_sky_r = image[non_sky_x, non_sky_y, 0]
        non_sky_g = image[non_sky_x, non_sky_y, 1]
        non_sky_b = image[non_sky_x, non_sky_y, 2]

        # create the sky dataframe
        # TODO: understand why the inversion of the x and y coordinates was necessary.
        sky_df = pd.DataFrame(
            {'image': image_count, 'r': sky_r, 'g': sky_g, 'b': sky_b, 'x': sky_y, 'y': sky_x, 'class': 1})

        # create the non-sky dataframe
        non_sky_df = pd.DataFrame(
            {'image': image_count, 'r': non_sky_r, 'g': non_sky_g, 'b': non_sky_b, 'x': non_sky_y, 'y': non_sky_x,
             'class': 0})

        # concatenate the dataframes
        df = pd.concat([df, sky_df, non_sky_df], ignore_index=True)

        # increment the image count
        image_count += 1

    return df


def inspect_sampler() -> None:
    """
    Visually inspect the sampler to make sure it is working properly.
    """
    # get the training dataset
    train_df: pd.DataFrame = pd.read_csv(Path(TRAINING_DATASET_PATH, 'train_by_pixel.csv'))
    # get the first binary mask
    mask: np.ndarray = skimage.io.imread(Path(TRAINING_DATASET_PATH, 'binary_mask_sky_0.png'))
    # get the pixels of the first image:
    pixels: pd.DataFrame = train_df[train_df['image'] == 0]
    # map the target to sky and non-sky
    # plot the pixels over the binary mask:
    plt.imshow(mask)
    plt.scatter(x=pixels['x'], y=pixels['y'], c=pixels['class'], cmap='bwr', s=1, alpha=0.5)
    # add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=['sky', 'non-sky'],
               title='Pixel Class')
    # add a title
    plt.title('Sampled pixels over the binary mask, extracted from the first image')
    # show the plot
    plt.show()


def explore_dataset(dataframe: pd.DataFrame, train: bool = True) -> None:
    """
    Function to print information on the generated dataset
    :return: None. Prints some information.
    """
    name = "Training" if train else "Testing"

    # print the number of rows and columns
    print(f"{name} dataframe has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.")
    # print the number of sky and non-sky pixels
    print(f"{name} dataframe has {dataframe[dataframe['class'] == 1].shape[0]} sky pixels and "
          f"{dataframe[dataframe['class'] == 0].shape[0]} non-sky pixels.")
    # print the number of images
    print(f"{name} dataframe has {dataframe['image'].nunique()} images.")


def main() -> None:
    """
    Main function to sample pixels from the training dataset, and save them to a csv file.
    Confirms that the sampling is working properly by plotting the sampled pixels over the binary mask.
    Gives some information on the generated dataset.
    """
    # sample amount of pixels from the training and testing dataset:
    training_pixels: int = 15000
    testing_pixels: int = 5000

    # sample 10000 pixels from the training dataset
    train_df: pd.DataFrame = sample_pixels(training_pixels, train=True)
    # sample 10000 pixels from the testing dataset
    test_df: pd.DataFrame = sample_pixels(testing_pixels, train=False)

    # save the sampled pixels
    train_df.to_csv(Path(TRAINING_DATASET_PATH, 'train_by_pixel.csv'), index=False)
    test_df.to_csv(Path(TESTING_DATASET_PATH, 'test_by_pixel.csv'), index=False)

    # inspect the sampled pixels
    inspect_sampler()

    # explore the dataset
    explore_dataset(train_df)
    explore_dataset(test_df)


# Driver Code
if __name__ == '__main__':
    main()
