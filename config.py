# File for the global paths of the project
import os

# get the project root path, which is the parent directory of the current file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# get the path of the initial dataset
INITIAL_DATASET_PATH = os.path.join(ROOT_PATH, 'initial_dataset')
# get the path of the training dataset
TRAINING_DATASET_PATH = os.path.join(INITIAL_DATASET_PATH, 'train')
# get the path of the testing dataset
TESTING_DATASET_PATH = os.path.join(INITIAL_DATASET_PATH, 'test')
# Training city:
# We will use the city of Aachen for our training dataset
TRAINING_CITY = "aachen"
# We will use the city of Zürich for our testing dataset
TESTING_CITY = "zurich"
