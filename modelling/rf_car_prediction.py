# Use the random forest model to classify car pixels, since KNN and Logistic Regression
# did not perform well.

# Libraries:
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from config import TRAINING_DATASET_PATH, TESTING_DATASET_PATH, VALIDATION_DATASET_PATH
from sklearn.metrics import roc_auc_score


def load_car_dataset(classification_type: str = 'by_pixel', split_type: str = 'train',
                     label: str = "sky") -> pd.DataFrame:
    """
    Loads the dataset from the path
    @param label: the class to load
    @param split_type: the type of split to be loaded. either train, or val, or test.
    @param classification_type: the type of classification from which to load the dataset.
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

    # select the correct dataset depending on the class you want to classify.
    subscript = "" if label == "sky" else "_" + label

    # Determine the type of classification, either by pixel or by patch
    if classification_type == 'by_pixel':
        df = pd.read_csv(Path(path, f'{name}_by_pixel{subscript}.csv'))
    elif classification_type == 'by_patch':
        # load the pickle file:
        df = pd.read_pickle(Path(path, f'{name}_by_patch{subscript}.pkl'))
    else:
        raise ValueError(f'Unknown classification type: {classification_type}')

    return df


def main() -> None:
    """
    Main function to segment an image by classifying by patches with a UNet convoluted neural network.
    :return: None, saves the AUC score in the results' folder.
    """
    # load the car train dataset
    train = load_car_dataset(split_type='train', label='car')

    # load the car test dataset
    test = load_car_dataset(split_type='test', label='car')

    # create the random forest model
    model = RandomForestClassifier()

    # fit the model to the training data
    x_train = train[['r', 'g', 'b']]
    y_train = train['class']
    x_train = np.array(x_train).reshape(-1, 3)
    model.fit(x_train, y_train)

    # predict the test data
    x_test = test[['r', 'g', 'b']]
    y_test = test['class']
    x_test = np.array(x_test).reshape(-1, 3)
    y_pred = model.predict(x_test)

    # calculate and print the AUC score
    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC score: {auc}')


if __name__ == '__main__':
    main()