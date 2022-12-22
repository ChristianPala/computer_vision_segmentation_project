# Library to classify a single pixel with a convoluted neural network
# Libraries:
# Data manipulation:
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import skimage
# Modelling:
from sklearn.metrics import f1_score
from keras import Input
from keras import Model
from keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate
from plot_keras_history import plot_history
from tensorboard import program
import tensorflow as tf
# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset

# Global variables:
from config import RESULTS_PATH, TENSORBOARD_LOGS_PATH, PATCH_SIZE, TRAINING_DATASET_PATH, VALIDATION_DATASET_PATH, \
    TESTING_DATASET_PATH

log_path = Path(TENSORBOARD_LOGS_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
LEARNING_CURVE_GRAPHS = os.path.join(RESULTS_PATH, 'learning_curves')
log_path.mkdir(parents=True, exist_ok=True)
Path(LEARNING_CURVE_GRAPHS).mkdir(parents=True, exist_ok=True)


def my_plot_history(history, fig_name: str) -> None:
    # Plot the epochs history with accuracy and loss function values at each epoch
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    save_fig(fig_name)
    plt.xlabel('Epoch')
    plt.ylabel('Scores')
    plt.show()

def save_fig(fig_name, tight_layout=True, fig_extension="png", resolution=300):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_name, dpi=resolution)


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
    x = Concatenate()([x, x2])  # Skip connections
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
    # load the car pickle file
    train = pd.read_pickle(Path(TRAINING_DATASET_PATH, 'train_by_patch_car.pkl'))
    validation = pd.read_pickle(Path(VALIDATION_DATASET_PATH, 'val_by_patch_car.pkl'))
    test = pd.read_pickle(Path(TESTING_DATASET_PATH, 'test_by_patch_car.pkl'))

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

    history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                        validation_data=(x_validation, y_validation),
                        callbacks=[tf.keras.callbacks.TensorBoard(str(log_path))])

    # Evaluate the model:
    y_pred = model.predict(x_test, batch_size=16)

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

    # Save the model as a pickle file:
    model_path = Path(RESULTS_PATH, 'models')
    model_path.mkdir(parents=True, exist_ok=True)

    model_path /= 'cnn_unet.h5'
    pickle.dump(model, open(model_path, 'wb'))

    # Plot the training history:
    my_plot_history(history, str(Path(LEARNING_CURVE_GRAPHS, 'auc_cnn_unet')))


if __name__ == '__main__':
    # main()

    # load the pickled model:
    model_path = Path(RESULTS_PATH, 'models', 'cnn_unet.h5')

    # get the first original image:
    image_path = Path(TESTING_DATASET_PATH, 'image_0.png')

    # get the first original mask:
    mask_path = Path(TESTING_DATASET_PATH, 'binary_mask_car_0.png')

    # get the first original mask:
    mask = skimage.io.imread(mask_path, as_gray=True)

    image = skimage.io.imread(image_path)

    # preprocess the image:
    image = image.astype(np.float32) / 255

    # add a batch dimension:
    image = np.expand_dims(image, axis=0)

    # predict the mask:
    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(image, batch_size=16)
    y_test = np.array(mask).ravel()

    # find the threshold with the best AUC:
    thresholds = np.linspace(0, 1, 100)

    #f1_scores = []
    #for threshold in thresholds:
        #y_pred_threshold = np.where(y_pred > threshold, 1, 0)
        #y_pred_threshold = y_pred_threshold.ravel()
        #f1_scores.append(f1_score(y_test, y_pred_threshold))

    #print(f1_scores.index(max(f1_scores)))

    # best threshold 0.64:
    y_pred_threshold = np.where(y_pred > 0.64, 1, 0)

    # plot the generated mask:
    plt.imshow(y_pred_threshold[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Predicted mask for the first image in the test set')
    plt.show()
