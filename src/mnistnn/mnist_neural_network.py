"""Provide the MNIST neural-network implementation."""

import numpy as np

from mnistnn.functions.CrossEntropyLoss import CrossEntropyLoss
from mnistnn.functions.ReLU import ReLU
from mnistnn.functions.SoftmaxStable import SoftmaxStable
from mnistnn.layer import Layer
from mnistnn.layer_parent import LayerParent
from mnistnn.loss import Loss


class MNISTNeuralNetwork:
    """Neural network for MNIST digit classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden Layer 1: 128 units with ReLU
    - Hidden Layer 2: 64 units with ReLU
    - Output Layer: 10 units with Softmax
    """

    def __init__(self) -> None:
        """Initialize the network layers and loss function."""
        # Initialize layers
        self.layers: list[LayerParent] = [
            Layer(
                input_size=784,
                output_size=128,
                activation=ReLU(),
            ),
            Layer(
                input_size=128,
                output_size=64,
                activation=ReLU(),
            ),
            Layer(input_size=64, output_size=10, activation=SoftmaxStable()),
        ]
        # Loss function
        self.loss_fn: Loss = CrossEntropyLoss()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network."""
        # Flatten input if needed (batch_size, 28, 28) -> (batch_size, 784)
        if len(input_.shape) == 3:
            input_ = input_.reshape(input_.shape[0], -1)

        x: np.ndarray = input_
        for layer in self.layers:
            x: np.ndarray = layer.forward(x)

        return x

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Perform a backward pass through the network."""
        input_grad: np.ndarray = output_grad

        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad)

        return input_grad

    def compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute the cross-entropy loss."""
        return self.loss_fn.forward(predictions, targets)

    def update_parameters(self, learning_rate: float) -> None:
        """Update weights and biases using gradient descent."""
        for layer in self.layers:
            # define weights and bias
            weights: np.ndarray
            bias: np.ndarray
            weights, bias = layer.get_params()
            grad_weights: np.ndarray
            grad_bias: np.ndarray
            grad_weights, grad_bias = layer.get_grads()
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> float:
        """Perform a single training step."""
        # Forward pass
        predictions: np.ndarray = self.forward(x)

        # Compute loss
        loss: float = self.compute_loss(predictions, y)

        # Backward pass
        grad: np.ndarray = self.loss_fn.backward()
        self.backward(grad)

        # Update parameters
        self.update_parameters(learning_rate)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions: np.ndarray = self.forward(x)
        return np.argmax(predictions, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy on a dataset."""
        predictions: np.ndarray = self.predict(x)

        # Convert one-hot encoded labels to class indices if needed
        if len(y.shape) == 2:
            y: np.ndarray = np.argmax(y, axis=1)

        accuracy: float = np.mean(predictions == y)
        return accuracy
