# Library to classify a single pixel with a convoluted neural network
# Libraries:
# Data manipulation:
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Modelling:
from keras import Input
from keras import Model
from keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorboard import program
import tensorflow as tf
# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset

# Global variables:
from config import RESULTS_PATH, SAMPLE_IMAGE_RESULTS_PATH, TENSORBOARD_LOGS_PATH, PATCH_SIZE
log_path = Path(TENSORBOARD_LOGS_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
log_path.mkdir(parents=True, exist_ok=True)


def create_model() -> Model:
    """
    Creates a UNet model, copied from professor Giusti's jupyter notebook fcn.ipynb.
    @return: Model, the UNet model.
    """
    input_layer = Input(shape=(None, None, 3), name='input')
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = SeparableConv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x1 = x
    x = MaxPooling2D(2)(x)
    x = SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x2 = x
    x = MaxPooling2D(2)(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x2]) # Skip connections
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x1])  # Skip connections
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    return Model(input_layer, x)


def main() -> None:
    """
    Main function to segment an image by classifying by patches with a UNet convoluted neural network.
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

    model.fit(x_train, y_train, epochs=3, batch_size=128,
              validation_data=(x_validation, y_validation),
              callbacks=[tf.keras.callbacks.TensorBoard(str(log_path))])

    # Evaluate the model:
    y_pred = model.predict(x_test, batch_size=32)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # flatten for the AUC metric:
    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test.ravel()
    # calculate the AUC:
    auc_cnn_unet = roc_auc_score(y_test_flat, y_pred_flat)

    print(f'The AUC on the test set is: {auc_cnn_unet}')

    # Save the results:
    results_path = Path(RESULTS_PATH)
    results_path.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({'auc': [auc_cnn_unet]})
    results_path /= 'auc_cnn_unet.csv'
    results.to_csv(results_path, index=False)


if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(log_path)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    main()
