"""Provide a layer combining a linear transformation and activation."""

import numpy as np

from mnistnn.functions.Linear import Linear
from mnistnn.layer_parent import LayerParent
from mnistnn.operation import Operation


class Layer(LayerParent):
    """Layer combining linear transformation and activation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Operation | None = None,
    ) -> None:
        """Initialize the layer with dimensions and an activation."""
        if (input_size <= 0) or (output_size <= 0):
            raise ValueError(
                "Input and output sizes must be positive integers."
            )
        if activation is None:
            raise ValueError("Activation function must be provided.")
        self.linear: Linear = Linear(input_size, output_size)
        self.activation: Operation | None = activation
        # Store the output of the linear layer for backward pass
        self.linear_output: np.ndarray | None = None

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Compute the layer output for the given input."""
        self.linear_output = self.linear.forward(input_)
        if self.activation:
            return self.activation.forward(self.linear_output)
        return self.linear_output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Propagate gradients backward through the layer."""
        if self.activation:
            output_grad = self.activation.backward(output_grad)
        return self.linear.backward(output_grad)

    def get_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the layer weights and bias."""
        return self.linear.weights, self.linear.bias

    def get_grads(self) -> tuple[np.ndarray, np.ndarray]:
        """Return gradients for the layer weights and bias."""
        return self.linear.grad_weights, self.linear.grad_bias
