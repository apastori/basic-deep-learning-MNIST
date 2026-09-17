import numpy as np
from mnistnn.operation import Operation  # type: ignore # Corrected module path

class Sigmoid(Operation):
    """Sigmoid activation function"""

    def _output(self, x: np.ndarray) -> np.ndarray:
        """Forward Pass during Prediction"""
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Backward Pass during Training"""
        sigmoid_derivative = self.output * (1 - self.output)
        return grad * sigmoid_derivative