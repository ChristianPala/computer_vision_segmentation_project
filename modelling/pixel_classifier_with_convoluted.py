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


def main() -> None:

    training: pd.DataFrame = load_dataset(classification_type='by_patch', train=True)
    testing: pd.DataFrame = load_dataset(classification_type='by_patch', train=False)

    # split the data into X and y:
    x_train = training[['r', 'g', 'b']]

    # For now, the only feature is the average RGB value:
    x_train = np.array(x_train.mean(axis=1)).reshape(-1, 1)
    y_train = training['class']

    x_test = testing[['r', 'g', 'b']]
    x_test = np.array(x_test.mean(axis=1)).reshape(-1, 1)
    y_test = testing['class']

    df_train = pd.DataFrame(x_train, y_train)

    model = mkmodel()
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













