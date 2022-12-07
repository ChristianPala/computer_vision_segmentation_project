# Libraries:
from preprocessing.initial_dataset_creation import main as preprocess_main
from preprocessing.sampler import main as sampler_main
from modelling.pixel_classifier_by_average_rgb import main as pixel_classifier_by_average_main
from modelling.pixel_classifier_by_rgb_features import main as pixel_classifier_by_rgb_features_main


def main() -> None:
    """
    Runs the project pipeline.
    :return: None
    """
    # Preprocessing:
    # ---------------------------
    preprocess_main()
    sampler_main()
    # Modelling:
    # ---------------------------
    # pixel classifier
    pixel_classifier_by_average_main()
    pixel_classifier_by_rgb_features_main()
    # patch classifier
    # TODO: implement patch classifier
    # segmentation
    # ---------------------------
    # TODO: implement segmentation
