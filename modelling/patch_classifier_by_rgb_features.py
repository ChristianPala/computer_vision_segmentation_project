# File to classify by RGB with single pixels as features
# Libraries:
# Data manipulation:
import numpy as np
from pathlib import Path
import pandas as pd
# Modelling:
from sklearn.metrics import roc_auc_score
from modelling.pixel_classifier_by_average_rgb import load_dataset, create_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Global variables:
from config import RESULTS_PATH


# Functions:
def train_model(model: LogisticRegression or KNeighborsClassifier, train: bool = True) \
        -> LogisticRegression or KNeighborsClassifier:
    """
    Trains the model on the dataset
    @param model: the model to train
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the trained model
    """
    df = load_dataset(classification_type="by_patch", train=train)
    x = df['patch']
    y = df['class']
    model.fit(x, y)

    return model


def evaluate_model(model: LogisticRegression or KNeighborsClassifier) -> float:
    """
    Evaluates the model on the dataset
    @param model: the model to evaluate
    :return: the AUC score of the model
    """
    df = load_dataset(train=False)
    x = df['patch']
    y = df['class']
    # Evaluate the model on the AUC score:
    auc = roc_auc_score(y, model.predict(x))
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
    results.to_csv(Path(RESULTS_PATH, 'patch_classifier_by_rgb_as_feature.csv'), index=False)


if __name__ == '__main__':
    main()
