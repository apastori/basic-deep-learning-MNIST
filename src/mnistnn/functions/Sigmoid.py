"""Provide the sigmoid activation operation."""

import numpy as np

from mnistnn.operation import Operation  # type: ignore # Corrected module path


class Sigmoid(Operation):
    """Sigmoid activation function."""

    def _output(self) -> np.ndarray:
        """Forward Pass during Prediction."""
        ## The output value of this function is stored in output through forward function in Operation class
        return 1 / (1 + np.exp( self.input_ * -1))

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Backward Pass during Training."""
        sigmoid_derivative = self.output * (1 - self.output)
        return grad * sigmoid_derivative
