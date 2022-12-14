# File to classify by RGB with single pixels as features
# Libraries:
# Data manipulation:
import numpy as np
from pathlib import Path
import pandas as pd
# Modelling:
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from modelling.pixel_classifier_by_average_rgb import load_dataset, create_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential


# Global variables:
from config import RESULTS_PATH


# Functions:
def train_model(model: LogisticRegression or KNeighborsClassifier or Sequential, train: bool = True) \
        -> LogisticRegression or KNeighborsClassifier:
    """
    Trains the model on the dataset
    @param model: the model to train
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the trained model
    """
    df = load_dataset(train=train)
    rgb = df[['r', 'g', 'b']]
    y = df['class']
    # for this second classifier we take the r, g, b values of each pixel as features:
    x = np.array(rgb).reshape(-1, 3)
    # split the data into train and test:
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
    # train the model:
    if model is Sequential:
        model.fit(x_train, y_train, epochs=100, batch_size=32)
    else:
        model.fit(x_train, y_train)
    return model


def evaluate_model(model: LogisticRegression or KNeighborsClassifier) -> float:
    """
    Evaluates the model on the dataset
    @param model: the model to evaluate
    :return: the AUC score of the model
    """
    df = load_dataset(train=False)
    rgb = df[['r', 'g', 'b']]
    x = np.array(rgb).reshape(-1, 3)
    y = df['class']
    # split the data into train and test:
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # Evaluate the model on the AUC score:
    auc = roc_auc_score(y_test, model.predict(x_test))
    return auc


def main():
    """
    The main function to run the evaluation of the second classifier
    :return: None. Prints the AUC score of the model
    """
    log_reg = create_model(model_type='logistic_regression')
    log_reg = train_model(log_reg)
    knn = create_model(model_type='knn')
    knn = train_model(knn)
    auc_log = evaluate_model(log_reg)
    print(f'Logistic Regression AUC with RGB as features: {auc_log}')
    auc_knn = evaluate_model(knn)
    print(f'KNN AUC with RBG as features: {auc_knn}')

    # ensure that the results folder exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': ['logistic_regression', 'knn'],
                            'auc': [auc_log, auc_knn]})
    results.to_csv(Path(RESULTS_PATH, 'pixel_classifier_by_rgb_as_feature.csv'), index=False)


if __name__ == '__main__':
    main()
