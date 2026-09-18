import numpy as np

from mnistnn.operation_param import ParamOperation


class BiasAdd(ParamOperation):
    """Compute bias addition."""

    def __init__(self, bias: np.ndarray) -> None:
        """Initialize Operation with self.param = bias.
        Check appropriate shape.
        """
        assert bias.shape[0] == 1
        super().__init__(bias)

    def _output(self) -> np.ndarray:
        """Compute Output."""
        return self.input_ + self.param

    def _input_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute input gradient."""
        return np.ones_like(self.input_) * grad

    def _param_grad(self, grad: np.ndarray) -> np.ndarray:
        """Compute parameter gradient."""
        param_grad = np.ones_like(self.param) * grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
