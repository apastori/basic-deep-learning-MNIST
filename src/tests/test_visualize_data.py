"""Tests for MNIST Visualizer class."""

from typing import Any

import numpy as np
import pytest

from mnistnn.data_config import dict_config_test, dict_config_train
from mnistnn.mnist_data_loader import MNISTDataLoader
from mnistnn.mnist_visualizer import MNISTVisualizer


class TestMNISTVisualizer:
    """Test suite for MNISTVisualizer class."""

    @pytest.fixture
    def data_loader(self) -> MNISTDataLoader:
        """Initialize MNIST data loader with configuration."""
        loader: MNISTDataLoader = MNISTDataLoader(
            dict_config_train, dict_config_test
        )
        return loader

    @pytest.fixture
    def training_data(self, data_loader: MNISTDataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve training images and labels."""
        train_images, train_labels = data_loader.training_info
        return train_images["images"], train_labels["labels"]

    def test_plot_single_image(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plotting a single image does not raise errors."""
        images_array, labels_array = training_data
        try:
            MNISTVisualizer.plot_single_image(images_array[0], labels_array[0])
        except Exception as e:
            raise AssertionError(f"Plotting single image raised an exception: {e}")

    def test_plot_random_images(self, training_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plotting 10 random images does not raise errors."""
        import random
        import matplotlib.pyplot as plt

        images_array, labels_array = training_data
        random_indices = random.sample(range(len(images_array)), 10)

        try:
            plt.figure(figsize=(10, 5))
            for i, idx in enumerate(random_indices):
                plt.subplot(2, 5, i + 1)
                MNISTVisualizer.plot_single_image(images_array[idx], labels_array[idx])
            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise AssertionError(f"Plotting random images raised an exception: {e}")
