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
from config import TRAINING_DATASET_PATH, VALIDATION_DATASET_PATH, \
    TESTING_DATASET_PATH, RESULTS_PATH


# Functions:
def load_dataset(classification_type: str = 'by_pixel', split_type: str = 'train') -> pd.DataFrame:
    """
    Loads the dataset from the path
    @param split_type: the type of split to be loaded. either train, or val, or test
    @param classification_type: the type of classification from which to load the dataset
    :return: the dataset
    """
    # Determine the path of the dataset
    name: str = split_type.lower()
    if name == 'train':
        path: Path = TRAINING_DATASET_PATH
    elif name == 'val':
        path: Path = VALIDATION_DATASET_PATH
    elif name == 'test':
        path: Path = TESTING_DATASET_PATH
    else:
        raise ValueError(f'Unknown split type: {split_type}')

    # Determine the type of classification, either by pixel or by patch
    if classification_type == 'by_pixel':
        df = pd.read_csv(Path(path, f'{name}_by_pixel.csv'))
    elif classification_type == 'by_patch':
        # load the pickle file:
        df = pd.read_pickle(Path(path, f'{name}_by_patch.pkl'))
    else:
        raise ValueError(f'Unknown classification type: {classification_type}')

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
    df = load_dataset(classification_type='by_pixel', split_type='train')
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
    df = load_dataset(classification_type='by_pixel', split_type='test')
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
    auc_log = evaluate_model(log_reg)
    print(f'Logistic Regression AUC: {auc_log}')
    auc_knn = evaluate_model(knn)
    print(f'KNN AUC: {auc_knn}')

    # ensure the results directory exists:
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    # save the results:
    results = pd.DataFrame({'model': ['logistic_regression', 'knn'],
                            'val_auc': [auc_log, auc_knn]})
    results.to_csv(Path(RESULTS_PATH, 'pixel_classifier_by_average_rgb.csv'), index=False)


if __name__ == '__main__':
    main()
