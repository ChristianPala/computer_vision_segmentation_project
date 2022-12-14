# File to segment an image by classifying each pixel with a convoluted neural network

# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
import os

from keras import Input, Model
from keras.losses import BinaryCrossentropy
# Modelling:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D, \
    UpSampling2D, Concatenate
# Typings
from typing import Union
# Plots
import matplotlib.pyplot as plt
from keras.optimizers import Adam
# Metrics:
from sklearn.metrics import roc_auc_score
# Utility functions
from modelling.pixel_classifier_by_average_rgb import load_dataset

# Global variables:
from config import RESULTS_PATH, SAMPLE_IMAGE_RESULTS_PATH
PATCH_SIZE = 512

# Ensure the directory exists:
Path(SAMPLE_IMAGE_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

# Tensorflow logging level:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_convolutional_model() -> Model:
    """
    Create a convolutional neural network with 2 hidden layers and a sigmoid output layer.
    :return: Sequential: the model.
    """
    pass
