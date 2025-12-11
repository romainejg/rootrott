import io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


# ---------------------------------
# Grid spacing (1 cm squares) helper
# ---------------------------------

def _estimate_grid_spacing_px(gray: np.ndarray) -> Optional[float]:
    """
    Estimate grid spacing (pixels per 1 cm square) using Hough lines.
    We look for long, nearly horizontal and vertical lines, then take
    the median spacing between parallel lines.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=120,  # long-ish lines only
        maxLineGap=10,
    )

    if lines is None or len(lines) < 10:
        return None

    vertical_positions = []
    horizontal_positions = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)

        if length < 80:
            continue

        # Vertical-ish
        if abs(dx) < abs(dy) * 0.3:
            vertical_positions.extend([x1, x2])
        # Horizontal-ish
        elif abs(dy) < abs(dx) * 0.3:
            horizontal_positions.extend([y1, y2])

    spacings: List[float] = []

    for positions in (vertical_positions, horizontal_positions):
        if len(positions) < 6:
            continue
        positions = sorted(positions)
        diffs = np.diff(positions)
        # Filter out tiny and huge gaps
        diffs = [d for d in diffs if 10 < d < 200]
        if diffs:
            spacings.extend(diffs)

    if not spacings:
        return None

    return float(np.median(spacings))


# ---------------------------
# Leaf segmentation (watershed)
# ---------------------------

@dataclass
class LeafMeasurement:
    leaf_id: int
    area_px: int
    area_cm2: float
    height_px: int
    height_cm: float
    area_per_height: float


def _segment_leaves_watershed(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment leaves using color threshold + watershed.
    Returns:
      - markers labeled image (int32)
      - cleaned binary mask (uint8 0/255)
    """
    # HSV threshold for green foliage
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # These ranges work nicely for typical lettuce images on white boards.
    lower = np.array([25, 30, 40])   # H, S, V
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, 2)

    # Remove small noise
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_mask = np.zeros_like(mask_clean)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # area threshold
            cv2.drawContours(big_mask, [cnt], -1, 255, thickness=-1)
    mask_clean = big_mask

    # Distance transform
    dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Foreground markers from peaks
    _, sure_fg = cv2.threshold(dist_norm, 0.35, 1.0, cv2.THRESH_BINARY)
    sure_fg_uint8 = (sure_fg * 255).astype("uint8")

    # Background and unknown
    sure_bg = cv2.dilate(mask_clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg_uint8)

    # Connected components for markers
    _, markers = cv2.connectedComponents(sure_fg_uint8)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed on original color image
    ws_input = bgr.copy()
    markers = cv2.watershed(ws_input, markers)

    # markers: -1 = boundary, 0 = background, 2..N = leaves
    return markers, mask_clean


def _measure_leaves(markers: np.ndarray, px_per_cm: float) -> Tuple[List[LeafMeasurement], np.ndarray]:
    """
    Measure each leaf based on watershed markers.
    Returns list of LeafMeasurement + BGR overlay image with boxes & labels.
    """
    overlay = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    overlay[:] = (0, 0, 0)

    measurements: List[LeafMeasurement] = []
    leaf_ids = [lab for lab in np.unique(markers) if lab > 1]  # ignore boundary (-1), bg (0), "1"

    for i, lab in enumerate(sorted(leaf_ids), start=1):
        ys, xs = np.where(markers == lab)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        height_px = y_max - y_min + 1
        area_px = len(xs)

        if px_per_cm <= 0:
            area_cm2 = float("nan")
            height_cm = float("nan")
        else:
            area_cm2 = area_px / (px_per_cm ** 2)
            height_cm = height_px / px_per_cm

        area_per_height = area_cm2 / height_cm if (height_cm and height_cm > 0) else float("nan")

        # Draw box + id on overlay (white leaves, green boxes)
        overlay[markers == lab] = (255, 255, 255)
        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            str(i),
            (x_min + 3, y_min + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        measurements.append(
            LeafMeasurement(
                leaf_id=i,
                area_px=area_px,
                area_cm2=area_cm2,
                height_px=height_px,
                height_cm=height_cm,
                area_per_height=area_per_height,
            )
        )

    return measurements, overlay


# -----------------------------
# Streamlit UI: Phenotyping tab
# -----------------------------

class PhenotypingUI:
    """Leaf phenotyping tool with watershed segmentation and automatic grid calibration."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload an image on a **1 cm² grid board**. Rootweiler will:

            - Detect the grid spacing → **pixels per centimetre**  
            - Segment individual leaves (even when they touch) using **watershed**  
            - Measure each leaf:
              - Area (cm²)
              - Height (cm)
              - Area : height ratio  
            - Summarise average size & variability
            """
        )

        uploaded = st.file_uploader(
            "Upload bench image",
            type=["jpg", "jpeg", "png"],
            help="Photo of leaves on a 1 cm grid board.",
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read via PIL then convert to OpenCV BGR
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Grayscale for grid detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Auto grid calibration
        px_per_cm_auto = _estimate_grid_spacing_px(gray)

        with st.expander("Grid calibration (1 cm squares)", expanded=True):
            if px_per_cm_auto is not None:
                st.success(
                    f"Auto-detected grid spacing: **~{px_per_cm_auto:.1f} px per 1 cm**"
                )
                px_per_cm = st.number_input(
                    "Pixels per centimetre (edit if needed)",
                    min_value=1.0,
                    max_value=500.0,
                    value=float(round(px_per_cm_auto, 1)),
                    step=0.5,
                )
            else:
                st.warning(
                    "Could not reliably detect grid spacing. "
                    "Please enter pixels per centimetre manually."
                )
                px_per_cm = st.number_input(
                    "Pixels per centimetre",
                    min_value=1.0,
                    max_value=500.0,
                    value=60.0,
                    step=0.5,
                )

        if st.button("Run phenotyping", type="primary"):
            cls._run_phenotyping(img_rgb, img_bgr, px_per_cm)

    @classmethod
    def _run_phenotyping(cls, img_rgb: np.ndarray, img_bgr: np.ndarray, px_per_cm: float):
        # Watershed segmentation
        markers, mask_clean = _segment_leaves_watershed(img_bgr)

        # Measurements
        measurements, overlay = _measure_leaves(markers, px_per_cm)

        if not measurements:
            st.error(
                "No leaves were detected. Try adjusting lighting, background, or the image."
            )
            return

        # Convert images for display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB)

        st.markdown("### Segmentation overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original image**")
            st.image(img_rgb, use_column_width=True)
        with c2:
            st.markdown("**Leaf labels**")
            st.image(overlay_rgb, use_column_width=True)
        with c3:
            st.markdown("**Binary mask**")
            st.image(mask_rgb, use_column_width=True)

        # Build table
        df = pd.DataFrame(
            [
                {
                    "Leaf ID": m.leaf_id,
                    "Area (px)": m.area_px,
                    "Area (cm²)": round(m.area_cm2, 2) if np.isfinite(m.area_cm2) else np.nan,
                    "Height (px)": m.height_px,
                    "Height (cm)": round(m.height_cm, 2) if np.isfinite(m.height_cm) else np.nan,
                    "Area : height (cm)": round(m.area_per_height, 2)
                    if np.isfinite(m.area_per_height)
                    else np.nan,
                }
                for m in measurements
            ]
        )

        st.markdown("### Leaf measurements")
        st.dataframe(df, use_container_width=True)

        # Summary stats
        st.markdown("### Summary")
        areas = df["Area (cm²)"].to_numpy(dtype=float)
        heights = df["Height (cm)"].to_numpy(dtype=float)

        def _safe_stats(arr: np.ndarray) -> Tuple[float, float]:
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else (float(np.mean(arr)), 0.0)

        mean_area, sd_area = _safe_stats(areas)
        mean_height, sd_height = _safe_stats(heights)

        st.write(f"- **Leaf count:** {len(df)}")
        if np.isfinite(mean_area):
            st.write(
                f"- **Average leaf area:** {mean_area:.2f} cm² "
                f"(SD: {sd_area:.2f} cm²)"
            )
        if np.isfinite(mean_height):
            st.write(
                f"- **Average leaf height:** {mean_height:.2f} cm "
                f"(SD: {sd_height:.2f} cm)"
            )

        st.caption(
            "Each square on the board is treated as 1 cm². "
            "Leaf height is measured along the vertical axis of the board, from tip to base of the segmented region."
        )
