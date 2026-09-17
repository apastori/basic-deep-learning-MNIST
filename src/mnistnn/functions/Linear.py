import numpy as np
from mnistnn.operation import Operation  # type: ignore # Corrected module path

class Linear(Operation):
    
    """Linear transformation (fully connected layer)"""
    def __init__(self, input_size, output_size) -> None:
        # The Initialization for Linear Function
        self.weights: np.ndarray = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias: np.ndarray = np.zeros((1, output_size))

    """Forward Pass during Prediction"""    
    def _output(self, x: np.ndarray) -> np.ndarray:
        ## output value of this function is stored in output through forward function in Operation class
        return np.dot(x, self.weights) + self.bias
    
    """Backward Pass during Training"""
    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        self.grad_weights: np.ndarray = np.dot(self.x.T, grad)
        self.grad_bias: np.ndarray = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)