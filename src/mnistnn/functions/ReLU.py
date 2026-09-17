from mnistnn.operation import Operation  # type: ignore # Corrected module path
import numpy as np

class ReLU(Operation):

    """ReLU activation function"""
    
    """Forward Pass during Prediction"""
    def _output(self, x: np.ndarray) -> np.ndarray:
        ## value saved in Output through forward function in Operation class
        return np.maximum(0, x)
    
    """Backward Pass during Training"""
    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
         # 1. Create a mask where ReLU was active
        relu_mask: np.ndarray = self.x > 0

        # 2. Convert boolean mask to numeric form
        relu_derivative: np.ndarray = relu_mask.astype(grad.dtype)

        # 3. Apply chain rule
        grad_input: np.ndarray = grad * relu_derivative

        return grad_input