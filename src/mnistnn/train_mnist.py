"""Main package to expose the different components."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mnistnn.log_config import configure_file_logging
from mnistnn.mnist_data_loader import MNISTDataLoader as MNISTDataLoader
from mnistnn.visualize_mnist import visualize_mnist

if TYPE_CHECKING:
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

logger = logging.getLogger(__name__)

def _prepare_images(images: np.ndarray, dataset_name: str) -> np.ndarray:
    """Validate MNIST image dimensions and flatten image batches."""
    if images.ndim == 3:
        if images.shape[1:] != (28, 28):
            raise ValueError(
                f'{dataset_name} images must have shape (N, 28, 28), '
                f'got {images.shape}'
            )
        return images.reshape(images.shape[0], -1)
    if images.ndim == 2 and images.shape[1] == 784:
        return images
    raise ValueError(
        f'{dataset_name} images must have shape (N, 28, 28) or (N, 784), '
        f'got {images.shape}'
    )


def train_mnist(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    visualize: bool = True,
    append_logs: bool = True,
) -> MNISTNeuralNetwork:
    """Load MNIST and train the network, optionally starting a fresh log."""
    if epochs <= 0:
        raise ValueError('epochs must be greater than zero')
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be greater than zero')

    log_path = configure_file_logging(logger, append=append_logs)
    logger.info('Writing MNIST training logs to %s', log_path)
    logger.info(
        'Starting MNIST training (epochs=%d, batch_size=%d, '
        'learning_rate=%g, visualize=%s)',
        epochs,
        batch_size,
        learning_rate,
        visualize,
    )

    from mnistnn.data_config import dict_config_test, dict_config_train
    from mnistnn.mnist_neural_network import MNISTNeuralNetwork

    data_loader: MNISTDataLoader = MNISTDataLoader(dict_config_train, dict_config_test)
    train_images_info: dict[str, str | int | np.ndarray] = data_loader.training_info[0]
    train_labels_info: dict[str, str | int | np.ndarray] = data_loader.training_info[1]
    test_images_info: dict[str, str | int | np.ndarray] = data_loader.test_info[0]
    test_labels_info: dict[str, str | int | np.ndarray] = data_loader.test_info[1]
    # The image data is stored as a NumPy array of shape (num_images, rows, cols) 3D
    # The label data is stored as a NumPy array of shape (num_labels,) 1D
    train_images_value: np.ndarray = train_images_info['images']
    train_labels_value: np.ndarray = train_labels_info['labels']
    test_images_value: np.ndarray = test_images_info['images']
    test_labels_value: np.ndarray = test_labels_info['labels']

    if not isinstance(train_images_value, np.ndarray):
        raise TypeError('Training images must be a NumPy array')
    if not isinstance(train_labels_value, np.ndarray):
        raise TypeError('Training labels must be a NumPy array')
    if not isinstance(test_images_value, np.ndarray):
        raise TypeError('Test images must be a NumPy array')
    if not isinstance(test_labels_value, np.ndarray):
        raise TypeError('Test labels must be a NumPy array')

    train_images: np.ndarray = train_images_value
    train_labels: np.ndarray = train_labels_value
    test_images: np.ndarray = test_images_value
    test_labels: np.ndarray = test_labels_value

    prepared_train_images: np.ndarray = _prepare_images(
        train_images, 'Training'
    )
    prepared_test_images: np.ndarray = _prepare_images(test_images, 'Test')
    train_size: int = len(prepared_train_images)
    test_size: int = len(prepared_test_images)
    if train_size == 0 or test_size == 0:
        raise ValueError('Training and test datasets must not be empty')

    if train_labels.ndim != 1 or len(train_labels) != train_size:
        raise ValueError(
            'Training labels must be a 1D array with one label per image'
        )
    if test_labels.ndim != 1 or len(test_labels) != test_size:
        raise ValueError(
            'Test labels must be a 1D array with one label per image'
        )
    if not np.issubdtype(train_labels.dtype, np.integer):
        raise TypeError('Training labels must contain integer class indices')
    if not np.issubdtype(test_labels.dtype, np.integer):
        raise TypeError('Test labels must contain integer class indices')
    if np.any((train_labels < 0) | (train_labels >= 10)):
        raise ValueError('Training labels must be digits between 0 and 9')
    if np.any((test_labels < 0) | (test_labels >= 10)):
        raise ValueError('Test labels must be digits between 0 and 9')

    logger.info(
        'Loaded MNIST datasets (training: %d images, shape=%s, dtype=%s; '
        'test: %d images, shape=%s, dtype=%s)',
        train_size,
        train_images.shape,
        train_images.dtype,
        test_size,
        test_images.shape,
        test_images.dtype,
    )

    # Visualize the MNIST dataset if visualize is True
    visualize_mnist(
        train_images=train_images,
        train_labels=train_labels,
        visualize=visualize
    )

    network: MNISTNeuralNetwork = MNISTNeuralNetwork()
    rng: np.random.Generator = np.random.default_rng()
    train_images = prepared_train_images
    test_images = prepared_test_images
    # generated one-hot encoded targets for the 10 classes (digits 0-9)
    # Matrix 10 X 10
    class_targets: np.ndarray = np.eye(10, dtype=np.float64)
    # Train for the specified number of epochs
    training_started_at: float = perf_counter()
    for epoch in range(epochs):
        epoch_started_at: float = perf_counter()
        # Shuffle the training data at the beginning of each epoch
        shuffled_indices: npt.NDArray[np.int64] = rng.permutation(train_size)
        total_loss: float = 0.0

        # Train in batches (the loop is done in batches to avoid memory issues)
        for start in range(0, train_size, batch_size):
            batch_indices: npt.NDArray[np.int64] = shuffled_indices[
                start : start + batch_size
            ]
            # The division by 255.0 normalizes pixel values to the range [0, 1]
            batch_images: np.ndarray = (
                train_images[batch_indices].astype(np.float64) / 255.0
            )
            # Get the correct labels for the current batch
            correct_labels: np.ndarray = train_labels[batch_indices]
            # Get the one-hot encoded target vectors for the correct labels
            batch_targets: np.ndarray = class_targets[correct_labels]
            avg_loss: float = network.train_step(
                batch_images, batch_targets, learning_rate
            )
            total_loss += avg_loss * len(batch_indices)
        # Evaluate the model on the test set after each epoch
        correct_predictions: int = 0
        for start in range(0, test_size, batch_size):
            end: int = start + batch_size
            test_batch_images: np.ndarray = (
                test_images[start:end].astype(np.float64) / 255.0
            )
            predictions: np.ndarray = network.predict(test_batch_images)
            correct_predictions += int(
                np.count_nonzero(predictions == test_labels[start:end])
            )

        average_loss: float = total_loss / train_size
        test_accuracy: float = correct_predictions / test_size
        logger.info(
            'Epoch %d/%d - loss: %.4f - test accuracy: %.2f%% - '
            'duration: %.2fs',
            epoch + 1,
            epochs,
            average_loss,
            test_accuracy * 100,
            perf_counter() - epoch_started_at,
        )

    logger.info(
        'MNIST training completed in %.2fs',
        perf_counter() - training_started_at,
    )
    return network


__all__ = ['train_mnist']
