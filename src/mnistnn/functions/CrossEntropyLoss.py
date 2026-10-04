"""Provide cross-entropy loss for multi-class classification."""

import numpy as np

from mnistnn.loss import Loss


class CrossEntropyLoss(Loss):
    """Calculate cross-entropy loss for multi-class classification."""

    def _output(self, predictions, targets) -> float:
        """Calculate loss from softmax predictions and one-hot targets.

        Predictions have shape ``(batch_size, num_classes)``. Targets are
        one-hot encoded labels with the same shape.
        """
        self.predictions = predictions
        self.targets = targets

        # Clip predictions to avoid log(0)
        clipped_preds = np.clip(predictions, 1e-10, 1 - 1e-10)

        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(clipped_preds)) / targets.shape[0]
        return loss

    def _input_grad(self) -> np.ndarray:
        # Gradient of cross-entropy with softmax
        return (self.predictions - self.targets) / self.targets.shape[0]
