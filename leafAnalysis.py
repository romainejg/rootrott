import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from openpyxl import Workbook


# Type alias for readability: (x, y, w, h)
BBox = Tuple[int, int, int, int]


def create_mask(image: np.ndarray, mask_type: int = 0) -> np.ndarray:
    """
    Create a binary mask for the leaf area using HSV and watershed.
    `mask_type` is currently unused but kept for API compatibility.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color and create a binary mask
    lower_green = np.array([0, 40, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold to obtain the sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background area
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that the sure background is not 0, but 1
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    # Create the mask based on watershed result
    final_mask = np.zeros_like(image[:, :, 0])
    final_mask[markers > 1] = 255

    return final_mask


def measure_objects(mask: np.ndarray, min_area: int = 50000) -> Tuple[List[np.ndarray], List[BBox]]:
    """
    Find contours and bounding boxes for objects in the mask.
    Only keeps objects with area > min_area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    measurements: List[BBox] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            measurements.append((x, y, w, h))

    return contours, measurements


def calculate_pixels_per_cm(image: np.ndarray) -> Optional[float]:
    """
    Estimate pixels per cm² using square-like objects in the image.
    Returns pixels_per_cm² or None if it cannot be determined.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    # Need enough squares to get a reliable estimate
    if len(squares) < 20:
        return None

    areas = [w * h for _, _, w, h in squares]
    median_area = np.median(areas)

    # Choose squares closest to the median area
    squares = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    squares = squares[:20]

    average_area = np.mean([w * h for _, _, w, h in squares])
    pixels_per_cm2 = average_area
    return float(pixels_per_cm2)


def analyze_image(image: np.ndarray, pixels_per_cm2: Optional[float] = None) -> Tuple[np.ndarray, List[BBox], float]:
    """
    Analyze a single image:
    - create mask
    - measure objects
    - compute or use provided pixels_per_cm²

    Returns:
        mask (np.ndarray): binary mask
        measurements (List[BBox]): list of bounding boxes
        pixels_per_cm2 (float): scale used (raises ValueError if cannot estimate)
    """
    if pixels_per_cm2 is None:
        pixels_per_cm2 = calculate_pixels_per_cm(image)
        if pixels_per_cm2 is None:
            raise ValueError("Could not determine pixels per cm² for this image.")

    mask = create_mask(image)
    _, measurements = measure_objects(mask)

    return mask, measurements, pixels_per_cm2


def analyze_folder(folder_path: str, output_filename: str = "measurements.xlsx") -> str:
    """
    Process all images in a folder, measure objects, and save results to an Excel file.

    The logic is adapted from your original `process_images` method, but:
    - No Tkinter / messagebox
    - Returns the path to the saved workbook instead of showing a popup

    Returns:
        Path to the saved Excel file.
    """
    workbook_path = os.path.join(folder_path, output_filename)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Filename", "Object", "Width (cm)", "Height (cm)"])

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: could not read image {image_path}")
            continue

        pixels_per_cm2 = calculate_pixels_per_cm(image)
        if pixels_per_cm2 is None:
            print(f"Could not determine pixels per cm² for {filename}")
            continue

        mask = create_mask(image)
        _, measurements = measure_objects(mask)

        # Original logic: filter by x-location clustering
        x_locations = [x + w // 2 for x, y, w, h in measurements]
        if not x_locations:
            continue

        median_x = np.median(x_locations)

        valid_measurements: List[BBox] = []
        for x, y, w, h in measurements:
            center_x = x + w // 2
            close_count = sum(
                1 for xx in x_locations if abs(center_x - xx) < 10 * np.sqrt(pixels_per_cm2)
            )
            if close_count >= 3:
                valid_measurements.append((x, y, w, h))

        for i, (x, y, w, h) in enumerate(valid_measurements):
            width_cm = round(w / np.sqrt(pixels_per_cm2), 1)
            height_cm = round(h / np.sqrt(pixels_per_cm2), 1)
            sheet.append([filename, i + 1, width_cm, height_cm])

    workbook.save(workbook_path)
    return workbook_path
