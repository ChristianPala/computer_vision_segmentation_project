# File to classify by RGB with single pixels as features
# Libraries:
# Data manipulation:
import numpy as np
from pathlib import Path
import pandas as pd
# Modelling:
from sklearn.metrics import roc_auc_score
from modelling.pixel_classifier_by_average_rgb import load_dataset, create_model
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Global variables:
from config import RESULTS_PATH


# Functions:
def main():
    """
    Main routine to run the evaluation of the second classifier
    :return: None. Prints the AUC score of the model
    """
    # Load the dataset:
    train = load_dataset(classification_type="by_patch", train=True)
    test = load_dataset(classification_type="by_patch", train=False)

    # get the features and labels:
    x_train = train['patch']
    y_train = train['class']
    x_test = test['patch']
    y_test = test['class']

    # reshape the data:
    x_train = np.array(x_train.tolist()).reshape((-1, 512, 512, 3)).astype(np.float32) / 255
    x_test = np.array(x_test.tolist()).reshape((-1, 512, 512, 3)).astype(np.float32) / 255

    # Create a fully connected model:
    model = Sequential()
    # the input layer are 369 patches of 512x512 pixels, each with 3 channels (RGB):
    model.add(tf.keras.layers.Flatten(input_shape=(512, 512, 3)))
    # the hidden layer has 128 neurons:
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # the output layer has 1 neuron:
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # compile the model:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model:
    model.fit(x_train, y_train, epochs=10)

    # Evaluate the model on the AUC score:
    auc = roc_auc_score(y_test, model.predict(x_test))

    print(f'Fully connected AUC with patch RGB values as features: {auc}')

    # ensure that the results folder exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': ['logistic_regression'],
                            'auc': [auc]})
    results.to_csv(Path(RESULTS_PATH, 'patch_classifier_by_rgb_as_feature.csv'), index=False)


if __name__ == '__main__':
    main()
