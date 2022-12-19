# File to segment an image by classifying each pixel with a feed forward neural network
# Libraries:
# Data manipulation:
from datetime import datetime
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
# Tensorboard:
import tensorflow as tf
from tensorboard import program
# Global variables:
from config import RESULTS_PATH, SAMPLE_IMAGE_RESULTS_PATH, TENSORBOARD_LOGS_PATH, PATCH_SIZE
# Ensure the directory exists:
Path(SAMPLE_IMAGE_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
log_path = Path(TENSORBOARD_LOGS_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
log_path.mkdir(parents=True, exist_ok=True)
# Tensorflow logging level:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(nr_neurons_layer_1: int = 512, nr_neurons_layer_2: int = 256, dropout_rate: float = 0.1) -> Sequential:
    """
    Create a feed forward neural network with 2 hidden layers and a sigmoid output layer.
    @param nr_neurons_layer_1: number of neurons in the first hidden layer
    @param nr_neurons_layer_2: number of neurons in the second hidden layer
    @param dropout_rate: dropout rate
    :return: Sequential: the model.
    """

    # Create a fully connected model:
    model = Sequential()
    # the input layer are 369 patches of 512x512 pixels, each with 3 channels (RGB):
    model.add(Flatten(input_shape=(PATCH_SIZE, PATCH_SIZE, 3)))
    # add a dense layer:
    model.add(Dense(nr_neurons_layer_1, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # second dense layer:
    model.add(Dense(nr_neurons_layer_2, activation='relu'))
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
                           title: str, path: Union[str, Path], save: bool = False) -> None:
    """
    Visualize the segmentation of an image superimposed on the original image.
    show the binary mask on the right.
    @param image: np.ndarray: the original image
    @param generated_segmentation: np.ndarray: the generated segmentation
    @param binary_mask: np.ndarray: the binary mask of the image
    @param title: str: the title of the plot
    @param save: bool: whether to save the plot
    @param path: Path or str: the path to save the plot to
    :return: None, shows the plot and optionally saves it.
    """

    # plot the image and superimpose the segmentation on the left, the segmentation on its own in the center as
    # a subplot and the binary mask on the right as a subplot:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.imshow(image, vmin=0, vmax=1)
    ax1.imshow(generated_segmentation, alpha=0.4, vmin=0, vmax=1, cmap='jet')
    # remove the axis:
    ax1.axis('off')
    ax1.set_title('Superimposed segmentation')
    ax2.imshow(generated_segmentation, vmin=0, vmax=1, cmap='gray')
    # remove the axis:
    ax2.axis('off')
    ax2.set_title(f'Generated segmentation')
    ax3.imshow(binary_mask, vmin=0, vmax=1, cmap='gray')
    # remove the axis:
    ax3.axis('off')
    ax3.set_title(f'Original binary mask')

    # set the title of the plot:
    fig.suptitle(title, fontsize=22, y=0.95)
    plt.axis('off')

    # save the plot if required:
    if save:
        plt.savefig(path)

    # show the plot:
    plt.show()


def main():
    """
    Main function to segment an image by classifying each pixel with a feed forward neural network.
    :return: None, saves the AUC score in the results' folder.
    """

    # Load training, validation and testing sets
    train = load_dataset(classification_type='by_patch', split_type='train')
    val = load_dataset(classification_type='by_patch', split_type='val')
    test = load_dataset(classification_type='by_patch', split_type='test')

    # Preprocess the data:
    # get the raw features and labels:
    x_train = train['patch']
    y_train = train['mask_label']
    x_val = val['patch']
    y_val = val['mask_label']
    x_test = test['patch']
    y_test = test['mask_label']

    # preprocess the data, convert to float32 and normalize to [0, 1]:
    x_train = np.array(x_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    x_val = np.array(x_val.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    x_test = np.array(x_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32) / 255
    y_train = np.array(y_train.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)
    y_val = np.array(y_val.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)
    y_test = np.array(y_test.tolist()).reshape((-1, PATCH_SIZE, PATCH_SIZE, 1)).astype(np.float32)

    # Create the Feed-Forward Neural Network model
    model = create_model()

    # Train the model:
    model.fit(x_train, y_train, epochs=10, batch_size=32,
              validation_data=(x_val, y_val), callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_path)])

    # Evaluate the model on the AUC metric:
    y_pred = model.predict(x_test)

    # Get the categorical predictions from the sigmoid output:
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Flatten for the AUC metric:
    y_pred_flat = y_pred.ravel()
    y_test_flat = y_test.ravel()

    # Calculate the AUC:
    auc_ff_nn = roc_auc_score(y_test_flat, y_pred_flat)
    print(f'The AUC on the test set is: {auc_ff_nn}')

    # Ensure the results directory exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # Save the results:
    results = pd.DataFrame({'model': ['dense'], 'auc': [auc_ff_nn]})
    results.to_csv(Path(RESULTS_PATH, 'segmentation_by_patch_classification_feed_forward.csv'), index=False)

    # Get 3 random images from the test set:
    random_indexes = np.random.choice(range(len(x_test)), 3)
    for i in random_indexes:
        # get the image and the segmentation:
        image = x_test[i]
        # get the binary mask:
        binary_mask = y_test[i]
        generated_segmentation = y_pred[i]
        # visualize the image and the segmentation:
        visualize_segmentation(image=image, generated_segmentation=generated_segmentation, binary_mask=binary_mask,
                               title=f'Test image {i} segmentation by patch classification', save=True,
                               path=Path(SAMPLE_IMAGE_RESULTS_PATH,
                                         f'test_image_{i}_segmentation_by_patch_classification_feed_forward.png'))


# Driver Code:
if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    main()
