# Libraries:
from preprocessing.initial_dataset_creation import main as preprocess_main
from preprocessing.binary_mask_creation_and_visualization import main as binary_mask_main
from preprocessing.sampler import main as sampler_main
from modelling.pixel_classifier_by_average_rgb import main as pixel_classifier_by_average_main
from modelling.pixel_classifier_by_rgb_features import main as pixel_classifier_by_rgb_features_main
from modelling.pixel_classifier_with_convoluted import main as pixel_classifier_with_convoluted_main
from modelling.patch_segmentation_with_feed_forward import main as patch_segmentation_with_feed_forward_main
from modelling.patch_segmentation_with_convoluted import main as patch_segmentation_with_convoluted_main
from modelling.patch_segmentation_with_convoluted_unet import main as patch_segmentation_with_convoluted_unet_main


def main() -> None:
    """
    Runs the project pipeline.
    :return: None
    """
    # Preprocessing:
    # ---------------------------
    preprocess_main()
    binary_mask_main()
    sampler_main()
    # Modelling:
    # ---------------------------
    # pixel classifier
    pixel_classifier_by_average_main()
    pixel_classifier_by_rgb_features_main()
    pixel_classifier_with_convoluted_main()
    # Segmentation
    patch_segmentation_with_feed_forward_main()
    patch_segmentation_with_convoluted_main()
    patch_segmentation_with_convoluted_unet_main()

