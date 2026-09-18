"""Provide a base implementation for sequential neural-network layers."""

from abc import abstractmethod

import numpy as np

from mnistnn.layer_parent import LayerParent
from mnistnn.operation import Operation
from mnistnn.operation_param import ParamOperation
from mnistnn.utils.shape import assert_same_shape


class GenericLayer(LayerParent):
    """A "layer" of neurons in a neural network."""

    def __init__(self, neurons: int):
        """Initialize the layer with the specified number of neurons.

        The number of "neurons" roughly corresponds to the "breadth" of the
        layer.
        """
        self.neurons: int = neurons
        self.first: bool = True
        self.params: list[np.ndarray] = []
        self.param_grads: list[np.ndarray] = []
        self.operations: list[Operation] = []

    @abstractmethod
    def _setup_layer(self, num_in: int) -> None:
        """Set up the layer for the specified input size."""
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Pass input forward through a series of operations."""
        if self.first:
            self._setup_layer(input_)
            self.first = False
            self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
            self.output = input_
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Pass output_grad backward through a series of operations.

        Checks appropriate shapes.
        """
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
            input_grad = output_grad
        self._param_grads()
        return input_grad
    def _param_grads(self) -> np.ndarray:
        """Extract the _param_grads from the layer's operations."""
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)
        return self.param_grads

    def _params(self) -> np.ndarray:
        """Extract the _params from the layer's operations."""
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
        return self.params
