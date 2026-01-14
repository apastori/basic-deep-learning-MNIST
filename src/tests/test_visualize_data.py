"""Tests for MNIST Visualizer class."""

import numpy as np
import pytest
from typing import Tuple

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
        train_images: dict[str, str | int | np.ndarray]
        train_labels: dict[str, str | int | np.ndarray]
        train_images, train_labels = data_loader.training_info
        images_array: np.ndarray = train_images["images"]
        labels_array: np.ndarray = train_labels["labels"]
        return images_array, labels_array

    def test_plot_single_image(self, training_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plotting a single image does not raise errors."""
        images_array: np.ndarray
        labels_array: np.ndarray
        images_array, labels_array = training_data
        try:
            MNISTVisualizer.plot_single_image(images_array[0], labels_array[0])
        except Exception as e:
            raise AssertionError(f"Plotting single image raised an exception: {e}")
        
    def test_plot_random_images_grid(self, training_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plotting a grid of 20 random MNIST images does not raise errors."""
        images_array: np.ndarray
        labels_array: np.ndarray
        images_array, labels_array = training_data
        rng: np.random.Generator = np.random.default_rng(seed=42)
        indices: np.ndarray = rng.choice(len(images_array), size=20, replace=False)

        random_images: np.ndarray = images_array[indices]
        random_labels: np.ndarray = labels_array[indices]

        try:
            MNISTVisualizer.plot_grid(
                random_images,
                random_labels,
                num_images=20,
                title="20 Random MNIST Samples",
            )
        except Exception as e:
            raise AssertionError(
                f"Plotting 20 random MNIST images raised an exception: {e}"
            )


