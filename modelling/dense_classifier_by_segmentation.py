# File to classify by RGB with single pixels as features using a Dense Neural Network
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
from keras.layers import Dense, Activation
# Typings
from typing import Union
# Plots
import matplotlib.pyplot as plt
# Utility functions
from pixel_classifier_by_rgb_features import evaluate_model, train_model
from modelling.pixel_classifier_by_average_rgb import load_dataset
from preprocessing.binary_mask_creation_and_visualization import visualize_image_and_segmentation,\
                                                                    get_image_and_segmentation


# Global variables:
from config import RESULTS_PATH


def create_model():

    df = load_dataset(train=True)
    rgb = df[['r', 'g', 'b']]
    X = np.array(rgb).reshape(-1, 3)

    model = Sequential()
    model.add(Dense(100, input_shape=X[0].shape))
    model.add(Activation("sigmoid"))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # For a multi-class classification problem
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['auc'])
    return model


def classify_image(model: Sequential, img_number: Union[str, int] = 1):

    # Get the requested image and its segmentation
    # todo: use data from test_by_pixel.csv
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
    dense_nn = create_model()
    dense_nn = train_model(dense_nn)
    auc_dense_nn = evaluate_model(dense_nn)
    print(f'Dense neural network AUC with RGB as features: {auc_dense_nn}')

    # ensure that the results folder exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': 'dense_nn', 'auc': auc_dense_nn})
    results.to_csv(Path(RESULTS_PATH, 'sense_pixel_classifier_by_rgb_as_feature.csv'), index=False)

    # Get an image classified by pixel using the model and compare it to its segmentation
    classified_img = classify_image(model=dense_nn, img_number=1)


if __name__ == '__main__':
    main()
