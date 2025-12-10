import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SegmentationParams:
    """
    Parameters for the HSV thresholding and morphological operations.
    These correspond to the Tkinter-controlled fields in your original app.
    """
    lower_hue: int = 0
    lower_saturation: int = 40
    lower_value: int = 50
    upper_hue: int = 80
    upper_saturation: int = 255
    upper_value: int = 255
    morph_iterations: int = 2
    kernel_size: int = 3
    dilate_iterations: int = 3
    dist_transform_threshold: float = 0.7


def create_mask_debug(
    image: np.ndarray,
    params: SegmentationParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full segmentation pipeline and return all intermediate results:

    Returns:
        mask            : initial HSV mask
        mask_cleaned    : after morphological ops
        dist_transform  : distance transform
        sure_fg         : sure foreground
        sure_bg         : sure background
        unknown         : unknown region
        markers         : watershed markers (with -1 at boundaries)
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range and create a binary mask
    lower_green = np.array(
        [params.lower_hue, params.lower_saturation, params.lower_value],
        dtype=np.uint8,
    )
    upper_green = np.array(
        [params.upper_hue, params.upper_saturation, params.upper_value],
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = np.ones((params.kernel_size, params.kernel_size), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iterations)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=params.morph_iterations)

    # Distance transform
    dist_transform = cv2.distanceTransform(mask_cleaned, cv2.DIST_L2, 5)

    # Sure foreground
    _, sure_fg = cv2.threshold(
        dist_transform,
        params.dist_transform_threshold * dist_transform.max(),
        255,
        0,
    )
    sure_fg = np.uint8(sure_fg)

    # Sure background
    sure_bg = cv2.dilate(mask_cleaned, kernel, iterations=params.dilate_iterations)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components on sure_fg (marker labeling)
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one so sure background is 1 instead of 0
    markers = markers + 1

    # Mark unknown regions as 0
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)
    # Boundaries in markers are -1

    return mask, mask_cleaned, dist_transform, sure_fg, sure_bg, unknown, markers


def create_debug_figure(
    image: np.ndarray,
    filename: str,
    params: SegmentationParams,
) -> plt.Figure:
    """
    Create a 2x4 debug figure showing:
    - Original image
    - Initial mask
    - Mask after morph operations
    - Distance transform
    - Sure foreground
    - Sure background
    - Unknown region
    - Markers (watershed)

    Returns:
        Matplotlib Figure (for use in Streamlit via st.pyplot(fig)).
    """
    (
        mask,
        mask_cleaned,
        dist_transform,
        sure_fg,
        sure_bg,
        unknown,
        markers,
    ) = create_mask_debug(image, params)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # Original image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"Original Image\n{filename}")
    axs[0, 0].axis("off")

    # Initial mask
    axs[0, 1].imshow(mask, cmap="gray")
    axs[0, 1].set_title("Initial Mask")
    axs[0, 1].axis("off")

    # Mask after morph ops
    axs[0, 2].imshow(mask_cleaned, cmap="gray")
    axs[0, 2].set_title("Mask after Morphological Operations")
    axs[0, 2].axis("off")

    # Distance transform
    axs[0, 3].imshow(dist_transform, cmap="gray")
    axs[0, 3].set_title("Distance Transform")
    axs[0, 3].axis("off")

    # Sure foreground
    axs[1, 0].imshow(sure_fg, cmap="gray")
    axs[1, 0].set_title("Sure Foreground")
    axs[1, 0].axis("off")

    # Sure background
    axs[1, 1].imshow(sure_bg, cmap="gray")
    axs[1, 1].set_title("Sure Background")
    axs[1, 1].axis("off")

    # Unknown region
    axs[1, 2].imshow(unknown, cmap="gray")
    axs[1, 2].set_title("Unknown Region")
    axs[1, 2].axis("off")

    # Markers (with -1 -> 0 for display)
    markers_copy = markers.copy()
    markers_copy[markers_copy == -1] = 0
    axs[1, 3].imshow(markers_copy, cmap="nipy_spectral")
    axs[1, 3].set_title("Markers (Watershed)")
    axs[1, 3].axis("off")

    fig.suptitle("Leaf Segmentation Debugger", fontsize=16)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def save_params_to_json(params: SegmentationParams, filepath: str) -> None:
    """
    Save parameter configuration to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(asdict(params), f, indent=2)


def load_params_from_json(filepath: str) -> SegmentationParams:
    """
    Load segmentation parameters from a JSON config file.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return SegmentationParams(**data)
