# Library to classify a single pixel with a convoluted neural network
# Libraries:
# Data manipulation:
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import os

# Modelling:
from keras import Input, Model
from keras.layers import Conv2D

# Typings
from typing import Union

# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset


# Global variables:
from config import RESULTS_PATH, TENSORBOARD_LOGS_PATH
from tensorboard import program

log_path = Path(TENSORBOARD_LOGS_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
log_path.mkdir(parents=True, exist_ok=True)


def create_model():
    input_layer = Input(shape=(None, None, 3))
    x = Conv2D(filters=2, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    return Model(input_layer, x)


def main() -> None:

    training: pd.DataFrame = load_dataset(classification_type='by_patch', split_type='train')
    validation: pd.DataFrame = load_dataset(classification_type='by_patch', split_type='val')
    testing: pd.DataFrame = load_dataset(classification_type='by_patch', split_type='test')

    # split the data into X and y:
    x_train = training[['r', 'g', 'b']]
    y_train = training['class']
    x_val = validation[['r', 'g', 'b']]
    y_val = validation['class']

    # For now, the only feature is the average RGB value:
    x_train = np.array(x_train.mean(axis=1)).reshape(-1, 1)
    x_val = np.array(x_val.mean(axis=1)).reshape(-1, 1)

    x_test = testing[['r', 'g', 'b']]
    x_test = np.array(x_test.mean(axis=1)).reshape(-1, 1)
    y_test = testing['class']

    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.TensorBoard(log_path, histogram_freq=1)])

    # Evaluate the model:
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # print the AUC:
    auc = roc_auc_score(y_test, y_pred)
    print(f'The AUC on the test set is {auc}')

    # Save the AUC score:
    results_path = Path(RESULTS_PATH)
    results_path.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({'model': ['cnn'],
                            'test_auc': [auc]})
    results.to_csv(Path(RESULTS_PATH, 'pixel_classifier_with_convoluted.csv'), index=False)


if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    main()













