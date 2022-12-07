# Libraries:
# Data Manipulation:
import numpy as np
import pandas as pd
from pathlib import Path

# Modelling:
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Global Variables:
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH


# Functions:
def load_dataset(train: bool = True) -> pd.DataFrame:
    """
    Loads the dataset from the path
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the dataset
    """
    path: Path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH
    name: str = 'train' if train else 'test'
    classification_type: str = 'by_pixel'
    df = pd.read_csv(Path(path, f'{name}_{classification_type}.csv'))
    # print the class distribution:
    print(f'{name} class distribution:')
    print(df['class'].value_counts())

    return df


def create_model(model_type: str = None) -> LogisticRegression or KNeighborsClassifier:
    """
    Creates a model from the dataset
    @param model_type: the type of model to create
    """
    random_seed: int = 42

    if model_type == 'logistic_regression' or type is None:
        # binary classification
        model = LogisticRegression(random_state=random_seed)
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model


def train_model(model: LogisticRegression or KNeighborsClassifier, train: bool = True) \
        -> LogisticRegression or KNeighborsClassifier:
    """
    Trains the model on the dataset
    @param model: the model to train
    @param train: True if the image is from the training dataset, False if it is from the testing dataset
    :return: the trained model
    """
    df = load_dataset(train)
    rgb = df[['r', 'g', 'b']]
    y = df['class']
    # for the initial classifier we only care about the average RGB value of each pixel:
    x = np.array(rgb.mean(axis=1)).reshape(-1, 1)
    model.fit(x, y)
    return model


def evaluate_model(model: LogisticRegression or KNeighborsClassifier) -> float:
    """
    Evaluates the model on the dataset
    @param model: the model to evaluate
    :return: the AUC score of the model
    """
    df = load_dataset(train=False)
    rgb = df[['r', 'g', 'b']]
    x = np.array(rgb.mean(axis=1)).reshape(-1, 1)
    y = df['class']
    # Evaluate the model on the AUC score:
    auc = roc_auc_score(y, model.predict(x))
    return auc


def main():
    """
    The main function to run the initial classifier
    :return: None. Prints the AUC score of the model
    """
    log_reg = create_model(model_type='logistic_regression')
    log_reg = train_model(log_reg)
    knn = create_model(model_type='knn')
    knn = train_model(knn)
    auc = evaluate_model(log_reg)
    print(f'AUC: {auc:.3f}')


if __name__ == '__main__':
    main()
