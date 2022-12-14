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
from config import RESULTS_PATH

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
    model.add(Flatten(input_shape=(512, 512, 3)))
    # add a dense layer:
    model.add(Dense(dense_layer_size_1, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # second dense layer:
    model.add(Dense(dense_layer_size_2, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # the output layer is a binary mask of 512x512 pixels, 0 for background and 1 for sky:
    model.add(Dense(512 * 512, activation='sigmoid'))
    # reshape the output to be a 512x512 mask:
    model.add(Reshape((512, 512, 1)))

    # compile the model:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    return model


def visualize_segmentation(image: np.ndarray, segmentation: np.ndarray, title: str):

    # plot the image and the segmentation:
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image, vmin=0, vmax=1)
    ax[1].imshow(segmentation, vmin=0, vmax=1)
    # center the title:
    fig.suptitle(title, fontsize=16, y=0.95)
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

    # preprocess the data:
    x_train = np.array(x_train.tolist()).reshape((-1, 512, 512, 3)).astype(np.float32) / 255
    x_test = np.array(x_test.tolist()).reshape((-1, 512, 512, 3)).astype(np.float32) / 255

    y_train = np.array(y_train.tolist()).reshape((-1, 512, 512, 1)).astype(np.float32)
    y_test = np.array(y_test.tolist()).reshape((-1, 512, 512, 1)).astype(np.float32)

    # create the model:
    model = create_model()

    # train the model:
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # evaluate the model on the AUC metric:
    y_pred = model.predict(x_test)
    y_pred_flat = y_pred.ravel()
    y_true = y_test
    y_true_flat = y_true.ravel()
    # calculate the AUC:
    auc = roc_auc_score(y_true_flat, y_pred_flat)

    print(f'The AUC on the test set is: {auc}')

    # ensure the results directory exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': ['dense'], 'auc': [auc]})
    results.to_csv(Path(RESULTS_PATH, 'patch_segmentation_by_pixel_classification.csv'), index=False)

    # get the first image and the first resulting segmentation:
    image = x_test[0]
    segmentation = y_pred[0]

    # visualize the segmentation:
    visualize_segmentation(image, segmentation, title='Segmentation by pixel classification on test image 1')


if __name__ == '__main__':
    main()
