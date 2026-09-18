"""Provide the softmax activation operation."""

import numpy as np

from mnistnn.operation import Operation  # type: ignore # Corrected module path


class Softmax(Operation):
    """Softmax activation function."""

    """Forward Pass during Prediction"""

    def _output(self) -> np.ndarray:
        # Check if the input is a 2D array
        if self.input_.ndim != 2:
            raise ValueError("Input to Softmax must be a 2D array")
        # Subtract max for numerical stability
        exp_x: np.ndarray = np.exp(
            self.input_ - np.max(self.input_, axis=1, keepdims=True)
        )
        # Output already saved in self.output through forward function in Operation class
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    """Backward Pass during Training"""

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        # Simplified backward pass for use with cross-entropy
        return grad
