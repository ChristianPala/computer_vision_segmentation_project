# Library to classify a single pixel with a convoluted neural network
# Libraries:
# Data manipulation:
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
# Modelling:
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D
# Tensorboard:
from tensorboard import program
# Plots
from sklearn.metrics import roc_auc_score
# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset

# Global variables:
from config import RESULTS_PATH, TENSORBOARD_LOGS_PATH, PATCH_SIZE

# Ensure the log directory exists:
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


def main():
    """
    Main function to segment an image by classifying each pixel with a feed forward neural network.
    :return: None, saves the AUC score in the results' folder.
    """

    train = load_dataset(classification_type='by_patch', split_type='train')
    validation = load_dataset(classification_type='by_patch', split_type='val')
    test = load_dataset(classification_type='by_patch', split_type='test')

    # Preprocess the data:
    # get the raw features and labels:
    x_train = train['patch']
    y_train = train['mask_label']
    x_validation = validation['patch']
    y_validation = validation['mask_label']
    x_test = test['patch']
    y_test = test['mask_label']

    # preprocess the data, convert to float32 and normalize to [0, 1]:
    x_train = np.array(x_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    x_validation = np.array(x_validation.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    x_test = np.array(x_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    y_train = np.array(y_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)
    y_validation = np.array(y_validation.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)
    y_test = np.array(y_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)

    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_validation, y_validation),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir=str(log_path), histogram_freq=1)])

    # Evaluate the model:
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # flatten for the AUC metric:
    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test.ravel()
    # calculate the AUC:
    auc_cnn_standard = roc_auc_score(y_test_flat, y_pred_flat)

    print(f'The AUC on the test set is: {auc_cnn_standard}')

    # Save the results:
    results_path = Path(RESULTS_PATH)
    results_path.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({'auc': [auc_cnn_standard]})
    results_path /= 'auc_ff_nn.csv'
    results.to_csv(results_path, index=False)


if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(log_path)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    main()
