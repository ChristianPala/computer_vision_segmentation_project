# Library to classify a single pixel with a convoluted neural network
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Modelling:
from keras import Input, Model
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D, \
    UpSampling2D, Concatenate
from keras.optimizers import Adam


# Typings
from typing import Union

# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset


# Global variables:
from config import RESULTS_PATH, SAMPLE_IMAGE_RESULTS_PATH

def mkmodel():
    input_layer = Input(shape=(512, 512, 3))
    x = Conv2D(filters=2, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    return Model(input_layer, x)


def main():
    """
    Main function to segment an image by classifying each pixel with a feed forward neural network.
    :return: None, saves the AUC score in the results' folder.
    """

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

    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model:
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Plot the ROC curve:
    plt.plot(*roc_curve(y_test, y_pred))

    # print the AUC:
    print(roc_auc_score(y_test, y_pred))


if __name__ == '__main__':
    main()






def create_convolutional_model() -> Model:
    """
    Create a convolutional neural network with 2 hidden layers and a sigmoid output layer.
    :return: Sequential: the model.
    """
    pass