"""Main package to expose the different components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mnistnn.mnist_data_loader import MNISTDataLoader as MNISTDataLoader
from mnistnn.visualize_mnist import visualize_mnist

if TYPE_CHECKING:
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

def train_mnist(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    visualize: bool = True,
) -> MNISTNeuralNetwork:
    """Load the MNIST dataset and train the neural network."""
    if epochs <= 0:
        raise ValueError('epochs must be greater than zero')
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be greater than zero')

    from mnistnn.data_config import dict_config_test, dict_config_train
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

    data_loader: MNISTDataLoader = MNISTDataLoader(dict_config_train, dict_config_test)
    train_images_info: dict[str, str | int | np.ndarray] = data_loader.training_info[0]
    train_labels_info: dict[str, str | int | np.ndarray] = data_loader.training_info[1]
    test_images_info: dict[str, str | int | np.ndarray] = data_loader.test_info[0]
    test_labels_info: dict[str, str | int | np.ndarray] = data_loader.test_info[1]
    # The image data is stored as a NumPy array of shape (num_images, rows, cols) 3D
    # The label data is stored as a NumPy array of shape (num_labels,) 1D
    train_images: np.ndarray = train_images_info['images']
    train_labels: np.ndarray = train_labels_info['labels']
    test_images: np.ndarray = test_images_info['images']
    test_labels: np.ndarray = test_labels_info['labels']

    if not isinstance(train_images, np.ndarray):
        raise TypeError('Training images must be a NumPy array')
    if not isinstance(train_labels, np.ndarray):
        raise TypeError('Training labels must be a NumPy array')
    if not isinstance(test_images, np.ndarray):
        raise TypeError('Test images must be a NumPy array')
    if not isinstance(test_labels, np.ndarray):
        raise TypeError('Test labels must be a NumPy array')

    # Visualize the MNIST dataset if visualize is True
    visualize_mnist(
        train_images=train_images,
        train_labels=train_labels,
        visualize=visualize
    )

    network: MNISTNeuralNetwork = MNISTNeuralNetwork()
    rng: np.random.Generator = np.random.default_rng()
    train_size: int = len(train_images)
    test_size: int = len(test_images)
    # generated one-hot encoded targets for the 10 classes (digits 0-9) 
    # Matrix 10 X 10
    class_targets: np.ndarray = np.eye(10, dtype=np.float64)
    # Flatten training images if needed (batch_size, 28, 28) -> (batch_size, 784)
    if len(train_images.shape) == 3:
        if train_images.shape[1:] != (28, 28):
            raise ValueError(
                f"Expected images with shape (N, 28, 28), "
                f"got {train_images.shape}"
            )
        train_images = train_images.reshape(train_images.shape[0], -1)
    # Flatten test images if needed (batch_size, 28, 28) -> (batch_size, 784)
    if len(test_images.shape) == 3:
        if test_images.shape[1:] != (28, 28):
            raise ValueError(
                f"Expected images with shape (N, 28, 28), "
                f"got {test_images.shape}"
            )
        test_images = test_images.reshape(test_images.shape[0], -1)
    # Train for the specified number of epochs
    for epoch in range(epochs):
        # Shuffle the training data at the beginning of each epoch
        shuffled_indices: npt.DArray[np.int64] = rng.permutation(train_size)
        total_loss: float = 0.0

        # Train in batches (the loop is done in batches to avoid memory issues)
        for start in range(0, train_size, batch_size):
            batch_indices: npt.NDArray[np.int64] = shuffled_indices[start : start + batch_size]
            # The division by 255.0 is to normalize the pixel values to the range [0, 1] 
            batch_images: np.ndarray = (
                train_images[batch_indices].astype(np.float64) / 255.0
            )
            # Get the correct labels for the current batch
            correct_labels: np.ndarray = train_labels[batch_indices]
            # Get the one-hot encoded 1D targets for the correct labels 
            batch_targets: np.ndarray = class_targets[correct_labels]
            avg_loss: float = network.train_step(
                batch_images, batch_targets, learning_rate
            )
            total_loss += avg_loss * len(batch_indices)
        # Evaluate the model on the test set after each epoch
        correct_predictions: int = 0
        for start in range(0, test_size, batch_size):
            end: int = start + batch_size
            batch_images: np.ndarray = test_images[start:end].astype(np.float64) / 255.0
            predictions: np.ndarray = network.predict(batch_images)
            correct_predictions += int(
                np.count_nonzero(predictions == test_labels[start:end])
            )

        average_loss: float = total_loss / train_size
        test_accuracy: float = correct_predictions / test_size
        print(
            f'Epoch {epoch + 1}/{epochs} - '
            f'loss: {average_loss:.4f} - '
            f'test accuracy: {test_accuracy:.2%}'
        )

    return network


__all__ = ['train_mnist']
