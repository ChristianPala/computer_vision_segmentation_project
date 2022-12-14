# File to segment an image by classifying each pixel with a feed forward neural network
# Libraries:
# Data manipulation:
from pathlib import Path
import glob
import pandas as pd
import numpy as np
# Modelling:
import skimage
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
# Typings
from typing import Union
# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset
from preprocessing.binary_mask_creation_and_visualization import visualize_image_and_segmentation,\
                                                                    get_image_and_segmentation


# Global variables:
from config import RESULTS_PATH


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



def classify_image(model: Sequential, img_number: Union[str, int] = 1):

    # Get the requested image and its segmentation
    # todo: use data from test_by_pixel.pkl
    img, segm = get_image_and_segmentation(img_number=img_number, train=False)
    x_dim = img.shape[0]
    y_dim = img.shape[1]

    # Classify by pixel the image
    classified_img = np.zeros(x_dim, y_dim)
    for r in range(x_dim):
        for c in range(y_dim):
            pred = model.predict(img)
            classified_img[r][c] = pred

    # Visualize the classified image and the true segmentation side-ny-side
    visualize_image_and_segmentation(im=classified_img, segmentation=segm)

    return classified_img


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
    # first ravel the predictions and the true labels:
    y_pred = model.predict(x_test).ravel()
    y_true = y_test.ravel()
    # calculate the AUC:
    auc = roc_auc_score(y_true, y_pred)

    print(f'The AUC on the test set is: {auc}')


if __name__ == '__main__':
    main()
