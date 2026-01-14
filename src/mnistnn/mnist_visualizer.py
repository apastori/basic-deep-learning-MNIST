import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class MNISTVisualizer:
    """Utility class for visualizing MNIST data."""

    @staticmethod
    def plot_single_image(
        image: np.ndarray, label: int, title: str = "MNIST Image"
    ) -> None:
        """Plot a single MNIST image with its label."""
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.title(f"{title} - Label: {label}")
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_grid(
        images: np.ndarray,
        labels: np.ndarray,
        num_images: int = 9,
        title: str = "MNIST Images Grid",
    ) -> None:
        """Plot multiple MNIST images in a grid."""
        num_images: int = min(num_images, len(images))
        grid_size: int = int(np.ceil(np.sqrt(num_images)))
        fig: matplotlib.figure.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes: np.ndarray = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_images:
                ax.imshow(images[i], cmap="gray")
                ax.set_title(f"Label: {labels[i]}")
                ax.axis("off")
            else:
                ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_image_statistics(
        images: np.ndarray, title: str = "MNIST Image Statistics"
    ) -> None:
        """Plot pixel intensity distribution."""
        pixels: np.ndarray = images.flatten()
        plt.figure(figsize=(10, 5))
        plt.hist(pixels, bins=256, color="blue", alpha=0.7)
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()