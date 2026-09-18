from abc import ABC, abstractmethod

import numpy as np


class LayerParent(ABC):
    """Abstract base class for layers."""

    @abstractmethod
    def forward(self, input_: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _param_grads(self) -> np.ndarray:
        raise NotImplementedError
