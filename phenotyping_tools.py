# phenotyping.py

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------
# Grid detection helper
# ------------------------------

@dataclass
class GridDetectionResult:
    px_per_cm: Optional[float]
    vertical_centers: Optional[np.ndarray]
    horizontal_centers: Optional[np.ndarray]
    crop_offset: Tuple[int, int]  # (offset_y, offset_x)


def _line_centers_from_binary(binary: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Given a binary image that contains mostly vertical or mostly horizontal lines,
    find line centers along the specified axis.

    axis=0 → we sum along rows to find vertical lines
    axis=1 → we sum along columns to find horizontal lines
    """
    proj = binary.sum(axis=axis).astype(np.float32)
    if proj.max() <= 0:
        return np.array([])

    # Threshold at 50% of max projection to find strong lines
    thresh = 0.5 * proj.max()
    mask = proj > thresh

    centers = []
    in_run = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            end = i - 1
            centers.append((start + end) / 2.0)
            in_run = False

    if in_run:
        centers.append((start + len(mask) - 1) / 2.0)

    centers = np.array(centers, dtype=np.float32)

    # Clean up: remove duplicate / very close lines based on median gap
    if len(centers) > 1:
        diffs = np.diff(centers)
        median_gap = np.median(diffs)
        if median_gap > 0:
            cleaned = [centers[0]]
            for c in centers[1:]:
                if c - cleaned[-1] > 0.5 * median_gap:
                    cleaned.append(c)
            centers = np.array(cleaned, dtype=np.float32)

    return centers


def detect_grid_spacing_px(image_bgr: np.ndarray) -> GridDetectionResult:
    """
    Automatically detect the pixel spacing between grid lines (1 cm squares).

    Returns a GridDetectionResult with px_per_cm and the line centers
    (in crop coordinates). If detection fails, px_per_cm is None.
    """
    h, w = image_bgr.shape[:2]

    # Crop to the central region where the grid almost surely lives.
    # This avoids edges / table / leaves confusing the detector.
    top = int(h * 0.05)
    bottom = int(h * 0.85)
    left = int(w * 0.05)
    right = int(w * 0.95)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[top:bottom, left:right]
    ch, cw = crop.shape

    # Adaptive threshold to pick up thin grid lines regardless of lighting
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )

    # Morphological extraction of mostly vertical & mostly horizontal lines
    kernel_len = max(10, ch // 60)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    vertical = cv2.erode(thr, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    horizontal = cv2.erode(thr, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    v_centers = _line_centers_from_binary(vertical, axis=0)
    h_centers = _line_centers_from_binary(horizontal, axis=1)

    px_per_cm = None
    if len(v_centers) >= 4:
        px_per_cm = float(np.median(np.diff(v_centers)))
    if len(h_centers) >= 4:
        if px_per_cm is None:
            px_per_cm = float(np.median(np.diff(h_centers)))
        else:
            px_per_cm = float(
                0.5 * (px_per_cm + np.median(np.diff(h_centers)))
            )

    return GridDetectionResult(
        px_per_cm=px_per_cm,
        vertical_centers=v_centers if len(v_centers) > 0 else None,
        horizontal_centers=h_centers if len(h_centers) > 0 else None,
        crop_offset=(top, left),
    )


def overlay_grid_on_image(
    image_bgr: np.ndarray,
    grid_res: GridDetectionResult,
    step: int = 2,
) -> np.ndarray:
    """
    Return an RGB image with detected grid lines overlayed so you can visually
    confirm the spacing.

    `step` controls how many lines we show (every 2nd, every 3rd, etc.).
    """
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]
    top, left = grid_res.crop_offset

    if grid_res.vertical_centers is not None:
        for i, cx in enumerate(grid_res.vertical_centers):
            if i % step != 0:
                continue
            x = int(cx + left)
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 0, 255), 1)

    if grid_res.horizontal_centers is not None:
        for i, cy in enumerate(grid_res.horizontal_centers):
            if i % step != 0:
                continue
            y = int(cy + top)
            cv2.line(overlay, (0, y), (w - 1, y), (0, 255, 0), 1)

    return overlay


# ------------------------------
# Leaf segmentation
# ------------------------------

@dataclass
class LeafMeasurement:
    leaf_id: int
    area_px: int
    height_px: int
    area_cm2: Optional[float]
    height_cm: Optional[float]
    area_height_ratio: Optional[float]


def segment_leaves(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[LeafMeasurement]]:
    """
    Segment leaves using HSV colour thresholding and watershed.
    Returns:
      - binary mask (uint8 0/255)
      - list of LeafMeasurement objects with pixel-level metrics filled in.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    # --- 1) HSV-based mask for green-ish + lettuce tissue ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # A fairly generous green / yellow-green band
    lower = np.array([20, 20, 10])
    upper = np.array([95, 255, 255])
    raw_mask = cv2.inRange(hsv, lower, upper)

    # Some stems & pale tissue: also include high value but lower saturation
    lower2 = np.array([0, 0, 120])
    upper2 = np.array([120, 80, 255])
    pale_mask = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(raw_mask, pale_mask)

    # --- 2) Morphological cleaning ---
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )
    min_area_px = 500  # tiny speck filter
    keep_mask = np.zeros_like(mask_clean)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            keep_mask[labels == i] = 255
    mask_clean = keep_mask

    # --- 3) Watershed to separate overlapping leaves ---
    # Background
    sure_bg = cv2.dilate(mask_clean, kernel, iterations=3)

    # Distance transform for foreground peaks
    dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = img.copy()
    markers_ws = cv2.watershed(ws_img, markers)

    # --- 4) Extract leaf components ---
    leaf_measurements: List[LeafMeasurement] = []
    final_mask = np.zeros((h, w), dtype=np.uint8)

    next_id = 1
    for label_val in np.unique(markers_ws):
        if label_val <= 1:
            continue  # 0 / 1 are background-related in this scheme

        component_mask = (markers_ws == label_val).astype(np.uint8)
        area_px = int(component_mask.sum())
        if area_px < min_area_px:
            continue

        ys, xs = np.where(component_mask > 0)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        height_px = int(y2 - y1 + 1)

        # Add to final mask
        final_mask[component_mask > 0] = 255

        leaf_measurements.append(
            LeafMeasurement(
                leaf_id=next_id,
                area_px=area_px,
                height_px=height_px,
                area_cm2=None,
                height_cm=None,
                area_height_ratio=None,
            )
        )
        next_id += 1

    return final_mask, leaf_measurements


def draw_leaf_boxes(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    leaf_measurements: List[LeafMeasurement],
) -> np.ndarray:
    """
    Draw bounding boxes and leaf IDs on an RGB copy of the original image.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # We reconstruct component masks by re-labelling connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    for leaf in leaf_measurements:
        # find closest connected component to this leaf by area
        best_idx = None
        best_diff = None
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            diff = abs(area - leaf.area_px)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is None:
            continue

        x = stats[best_idx, cv2.CC_STAT_LEFT]
        y = stats[best_idx, cv2.CC_STAT_TOP]
        w_box = stats[best_idx, cv2.CC_STAT_WIDTH]
        h_box = stats[best_idx, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(out, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cx, cy = int(centroids[best_idx][0]), int(centroids[best_idx][1])
        cv2.putText(
            out,
            f"{leaf.leaf_id}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# ------------------------------
# Streamlit UI
# ------------------------------

class PhenotypingUI:
    """
    Rootweiler leaf phenotyping.

    - Auto-detect grid spacing (1 cm squares)
    - Segment leaves
    - Report leaf area (cm²), height (cm), area:height ratio
    """

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a **top-down image of lettuce leaves on the grid board**.
            Rootweiler will:

            - auto-detect the **grid spacing** (1 cm squares)  
            - segment individual leaves  
            - report **leaf area (cm²)**, **height (cm)** and **area : height**  
            """
        )

        uploaded = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read image. Please check the file.")
            return

        # 1. Grid detection
        grid_res = detect_grid_spacing_px(img_bgr)

        if grid_res.px_per_cm is None:
            st.warning(
                "Could not automatically detect grid spacing. "
                "Area will be reported in pixel units only."
            )
        else:
            st.success(
                f"Detected grid spacing: **~{grid_res.px_per_cm:.1f} pixels per 1 cm**"
            )

        overlay_img = overlay_grid_on_image(img_bgr, grid_res, step=3)

        # 2. Leaf segmentation
        binary_mask, leaf_measurements = segment_leaves(img_bgr)

        # If we have px_per_cm, convert metrics
        px_per_cm = grid_res.px_per_cm
        if px_per_cm is not None and px_per_cm > 0:
            for leaf in leaf_measurements:
                leaf.area_cm2 = leaf.area_px / (px_per_cm ** 2)
                leaf.height_cm = leaf.height_px / px_per_cm
                if leaf.height_cm > 0:
                    leaf.area_height_ratio = leaf.area_cm2 / leaf.height_cm
        else:
            for leaf in leaf_measurements:
                leaf.area_cm2 = None
                leaf.height_cm = None
                leaf.area_height_ratio = None

        # 3. Visual overview
        st.markdown("### Segmentation overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original + grid overlay**")
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                use_column_width=True,
            )

        leaf_label_img = draw_leaf_boxes(img_bgr, binary_mask, leaf_measurements)
        with col2:
            st.markdown("**Leaf labels**")
            st.image(
                cv2.cvtColor(leaf_label_img, cv2.COLOR_BGR2RGB),
                use_column_width=True,
            )

        with col3:
            st.markdown("**Binary mask**")
            st.image(binary_mask, use_column_width=True, clamp=True)

        # 4. Table of measurements
        st.markdown("### Leaf measurements")

        if not leaf_measurements:
            st.warning("No leaves detected – check the image or segmentation.")
            return

        rows = []
        for leaf in leaf_measurements:
            rows.append(
                {
                    "Leaf ID": leaf.leaf_id,
                    "Area (cm²)" if leaf.area_cm2 is not None else "Area (px²)": (
                        round(leaf.area_cm2, 2)
                        if leaf.area_cm2 is not None
                        else leaf.area_px
                    ),
                    "Height (cm)" if leaf.height_cm is not None else "Height (px)": (
                        round(leaf.height_cm, 2)
                        if leaf.height_cm is not None
                        else leaf.height_px
                    ),
                    "Area : height"
                    + (" (cm²/cm)" if leaf.area_height_ratio is not None else " (px)"): (
                        round(leaf.area_height_ratio, 2)
                        if leaf.area_height_ratio is not None
                        else None
                    ),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # 5. Summary stats
        st.markdown("### Summary")

        leaf_count = len(leaf_measurements)
        st.write(f"- Leaf count: **{leaf_count}**")

        if px_per_cm is not None:
            areas = np.array([leaf.area_cm2 for leaf in leaf_measurements])
            areas = areas[~np.isnan(areas)]
            if len(areas) > 0:
                st.write(
                    f"- Mean leaf area: **{areas.mean():.2f} cm²** "
                    f"(± {areas.std(ddof=1):.2f} cm²)"
                )
        else:
            areas_px = np.array([leaf.area_px for leaf in leaf_measurements])
            st.write(
                f"- Mean leaf area: **{areas_px.mean():.0f} px²** "
                f"(± {areas_px.std(ddof=1):.0f} px²) – "
                "grid not detected, so still in pixels."
            )

