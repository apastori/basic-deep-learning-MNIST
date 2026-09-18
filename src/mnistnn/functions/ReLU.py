"""Provide the rectified linear unit activation operation."""

import numpy as np

from mnistnn.operation import Operation  # type: ignore # Corrected module path


class ReLU(Operation):
    """ReLU activation function."""

    """Forward Pass during Prediction"""
    def _output(self) -> np.ndarray:
        ## value saved in Output through forward function in Operation class
        return np.maximum(0, self.input_)

    """Backward Pass during Training"""
    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
         # Create a mask where ReLU was active
         # this produces a boolean array where True indicates the input was greater than 0
        relu_mask: np.ndarray = self.input_ > 0

        # 2. Convert boolean mask to numeric form
        relu_derivative: np.ndarray = relu_mask.astype(grad.dtype)

        # 3. Apply chain rule
        grad_input: np.ndarray = grad * relu_derivative

        return grad_input
