import numpy as np
from mnistnn.operation_param import ParamOperation

class WeightMultiply(ParamOperation):
    
    """Weight multiplication operation for a neural network."""
    def __init__(self, weights: np.ndarray):
        ## Initialize Operation with self.param = weights.
        super().__init__(weights)

    def _output(self) -> np.ndarray:
        """Compute output."""
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Compute input gradient."""
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Compute parameter gradient."""
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)