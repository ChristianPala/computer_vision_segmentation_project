# File to segment an image by classifying each pixel with a feed forward neural network
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
import os
# Modelling:
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape
# Typings
from typing import Union
# Plots
import matplotlib.pyplot as plt
# Metrics:
from sklearn.metrics import roc_auc_score
# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset

# Global variables:
from config import RESULTS_PATH, SAMPLE_IMAGE_RESULTS_PATH
PATCH_SIZE = 512

# Ensure the directory exists:
Path(SAMPLE_IMAGE_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

# Tensorflow logging level:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model():

    # parameters:
    dropout_rate = 0.1
    dense_layer_size_1 = 512
    dense_layer_size_2 = 256

    # Create a fully connected model:
    model = Sequential()
    # the input layer are 369 patches of 512x512 pixels, each with 3 channels (RGB):
    model.add(Flatten(input_shape=(PATCH_SIZE, PATCH_SIZE, 3)))
    # add a dense layer:
    model.add(Dense(dense_layer_size_1, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # second dense layer:
    model.add(Dense(dense_layer_size_2, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # the output layer is a binary mask of 512x512 pixels, 0 for background and 1 for sky:
    model.add(Dense(PATCH_SIZE * PATCH_SIZE, activation='sigmoid'))
    # reshape the output to be a 512x512 mask:
    model.add(Reshape((PATCH_SIZE, PATCH_SIZE, 1)))

    # compile the model:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    return model


def visualize_segmentation(image: np.ndarray, generated_segmentation: np.ndarray, binary_mask: np.ndarray,
                           title: str, save: bool = False,
                           path: Union[str, Path] = None):

    # plot the image and superimpose the segmentation, add the binary mask as a subplot:
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, vmin=0, vmax=1)
    ax1.imshow(generated_segmentation, alpha=0.5, vmin=0, vmax=1, cmap='jet')
    ax1.set_title('Generated segmentation')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(binary_mask, vmin=0, vmax=1, cmap='gray')
    # center the title:
    fig.suptitle(title, fontsize=16, y=0.95)
    # remove the axis:
    plt.axis('off')
    # save the plot:
    if save:
        plt.imsave(path, image)

    # show the plot:
    plt.show()


def main():

    train = load_dataset(classification_type='by_patch', train=True)
    test = load_dataset(classification_type='by_patch', train=False)

    # Preprocess the data:
    # get the raw features and labels:
    x_train = train['patch']
    y_train = train['mask_label']
    x_test = test['patch']
    y_test = test['mask_label']

    # preprocess the data, convert to float32 and normalize to [0, 1]:
    x_train = np.array(x_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    x_test = np.array(x_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    y_train = np.array(y_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)
    y_test = np.array(y_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)

    # create the model:
    model = create_model()

    # train the model:
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # evaluate the model on the AUC metric:
    y_pred = model.predict(x_test)
    # get the categorical predictions from the sigmoid output:
    y_pred = np.where(y_pred > 0.5, 1, 0)
    # flatten for the AUC metric:
    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test.ravel()
    # calculate the AUC:
    auc_ff_nn = roc_auc_score(y_test_flat, y_pred_flat)

    print(f'The AUC on the test set is: {auc_ff_nn}')

    # ensure the results directory exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model':  ['dense'], 'auc': [auc_ff_nn]})
    results.to_csv(Path(RESULTS_PATH, 'patch_segmentation_by_pixel_classification.csv'), index=False)

    # get 3 random images from the test set:
    random_indexes = np.random.choice(range(len(x_test)), 3)
    for i in random_indexes:
        # get the image and the segmentation:
        image = x_test[i]
        # get the binary mask:
        binary_mask = y_test[i]
        generated_segmentation = y_pred[i]
        # visualize the image and the segmentation:
        visualize_segmentation(image=image, generated_segmentation=generated_segmentation,
                               binary_mask=binary_mask, title=f'Test image {i} segmentation by pixel classification',
                               save=True, path=Path(SAMPLE_IMAGE_RESULTS_PATH,
                                                    f'test_image_{i}_segmentation_by_pixel_classification.png'))


if __name__ == '__main__':
    main()
