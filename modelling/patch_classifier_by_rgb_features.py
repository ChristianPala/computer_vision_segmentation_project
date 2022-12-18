# File to classify by RGB with single pixels as features
# Libraries:
# Data manipulation:
import numpy as np
from pathlib import Path
import pandas as pd
# Modelling:
from sklearn.metrics import roc_auc_score
from modelling.pixel_classifier_by_average_rgb import load_dataset, create_model
from tensorboard import program
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow as tf

# Global variables:
from config import RESULTS_PATH, TENSORBOARD_LOGS_PATH

Path(TENSORBOARD_LOGS_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)


# Functions:
def preprocess_data(x, y):
    """
    Preprocess the data to be used in the model
    :param x: the features
    :param y: the labels
    :return: the preprocessed features and labels
    """
    # reshape the data:
    x = np.array(x.tolist()).reshape((-1, 512, 512, 3)).astype(np.float32) / 255
    y = np.array(y.tolist()).reshape((-1, 1)).astype(np.float32)

    return x, y


def create_feed_forward_model(x_shape: int, y_shape: int, channels: int) -> Sequential:
    """
    Create a feed forward model
    :param x_shape: the shape of the x-axis of the image
    :param y_shape: the shape of the y-axis of the image
    :param channels: the number of channels in the image
    :return: Sequential: the model
    """
    # parameters:
    dropout_rate = 0.1
    dense_layer_size_1 = 512
    dense_layer_size_2 = 256

    # Create a fully connected model:
    model = Sequential()
    # the input layer are 369 patches of 512x512 pixels, each with 3 channels (RGB):
    model.add(Flatten(input_shape=(x_shape, y_shape, channels)))
    # add a dense layer:
    model.add(Dense(dense_layer_size_1, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # second dense layer:
    model.add(Dense(dense_layer_size_2, activation='relu'))
    # add a dropout layer to prevent over-fitting:
    model.add(Dropout(dropout_rate))
    # the output layer has 1 neuron since it's a binary classification:
    model.add(Dense(1, activation='sigmoid'))
    # compile the model:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    return model


def main() -> None:
    """
    Main routine to run the evaluation of the second classifier
    :return: None. Prints the AUC score of the model
    """
    # Load the dataset:
    train = load_dataset(classification_type="by_patch", split_type='train')
    val = load_dataset(classification_type="by_patch", split_type='val')
    test = load_dataset(classification_type="by_patch", split_type='test')

    # get the raw features and labels:
    x_train = train['patch']
    y_train = train['class']
    x_val = val['patch']
    y_val = val['class']
    x_test = test['patch']
    y_test = test['class']

    # preprocess the data:
    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)
    x_test, y_test = preprocess_data(x_test, y_test)

    # get the shape of a single image for the network input:
    x_shape, y_shape, channels = x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2]

    # create the model:
    model = create_feed_forward_model(x_shape, y_shape, channels)

    # train the model:
    model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_PATH)])

    # predict the test data:
    y_pred = model.predict(x_test)
    # categorize the predictions:
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Evaluate the model on the AUC score:
    auc = roc_auc_score(y_test, y_pred)

    print(f'Fully connected AUC with patch RGB values as features: {auc}')

    # ensure that the results folder exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': ['Feed forward neural network with RGB values as features'],
                            'auc': [auc]})
    results.to_csv(Path(RESULTS_PATH, 'patch_classifier_by_rgb_as_feature.csv'), index=False)


# Driver code:
if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', TENSORBOARD_LOGS_PATH])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    main()
