# phenotyping_tools.py

import numpy as np
import cv2
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image


@dataclass
class LeafMeasurement:
    leaf_id: int
    area_px: float
    height_px: float
    area_cm2: float
    height_cm: float
    area_to_height_cm: float


# -------------------------
# Grid calibration helpers
# -------------------------

def _detect_grid_spacing_px(gray: np.ndarray) -> Optional[float]:
    """
    Try to estimate pixels-per-1-cm-grid spacing from the background.

    Returns: average spacing in pixels between vertical/horizontal grid lines,
             or None if detection fails.
    """
    # Emphasize dark lines on light background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(th, 50, 150, apertureSize=3)

    # Probabilistic Hough for line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=40,
        maxLineGap=5,
    )

    if lines is None or len(lines) < 5:
        return None

    vertical_x = []
    horizontal_y = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx < 5 and dy > 20:  # near-vertical
            vertical_x.append((x1 + x2) / 2)
        elif dy < 5 and dx > 20:  # near-horizontal
            horizontal_y.append((y1 + y2) / 2)

    def _spacing(vals: List[float]) -> Optional[float]:
        if len(vals) < 4:
            return None
        vals = np.array(sorted(vals))
        # cluster close lines (same physical grid line)
        diffs = np.diff(vals)
        # heuristic: ignore outliers > 2× median
        med = np.median(diffs)
        diffs = diffs[(diffs > 0.5 * med) & (diffs < 2.0 * med)]
        if len(diffs) == 0:
            return None
        return float(np.median(diffs))

    v_space = _spacing(vertical_x)
    h_space = _spacing(horizontal_y)

    candidates = [s for s in [v_space, h_space] if s is not None]
    if not candidates:
        return None

    return float(np.mean(candidates))


# -------------------------
# Leaf segmentation helpers
# -------------------------

def _segment_leaves_bgr(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment lettuce leaves from grid background.

    Returns:
        mask (uint8): binary mask of leaves
        markers (int32): watershed marker image (for debugging)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Wide-ish green range; user can still tweak upstream if needed
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform for watershed
    dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    # Threshold for sure-foreground
    _, sure_fg = cv2.threshold(
        dist,
        0.5 * dist.max(),  # 50% of max distance
        255,
        0,
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background
    sure_bg = cv2.dilate(mask_clean, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components on sure foreground
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 0 reserved for unknown in watershed
    markers[unknown == 255] = 0

    # Watershed modifies markers in place
    bgr_copy = bgr.copy()
    cv2.watershed(bgr_copy, markers)

    # Final leaf mask: regions with marker > 1
    leaf_mask = np.zeros_like(mask_clean)
    leaf_mask[markers > 1] = 255

    return leaf_mask, markers


def _measure_leaves(
    mask: np.ndarray,
    px_per_cm: float,
    min_area_px: int = 2000,
) -> List[LeafMeasurement]:
    """
    Measure each leaf region in a binary mask.

    Uses contour area and bounding box 'height'.

    Args:
        mask: uint8 0/255
        px_per_cm: calibration factor
        min_area_px: tiny blobs below this are ignored
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves: List[LeafMeasurement] = []

    for leaf_id, cnt in enumerate(contours, start=1):
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        height_px = max(w, h)  # approximate "leaf height"

        # convert to cm/ cm²
        if px_per_cm is not None and px_per_cm > 0:
            area_cm2 = area_px / (px_per_cm ** 2)
            height_cm = height_px / px_per_cm
            area_to_height_cm = area_cm2 / height_cm if height_cm > 0 else 0.0
        else:
            # fallback: report px units only
            area_cm2 = float("nan")
            height_cm = float("nan")
            area_to_height_cm = float("nan")

        leaves.append(
            LeafMeasurement(
                leaf_id=leaf_id,
                area_px=area_px,
                height_px=height_px,
                area_cm2=area_cm2,
                height_cm=height_cm,
                area_to_height_cm=area_to_height_cm,
            )
        )

    return leaves


# -------------------------
# Streamlit UI
# -------------------------

class PhenotypingUI:
    """Image-based phenotyping for grid images (lettuce leaves)."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a **top-down photo on the centimetre grid**.  
            Rootweiler will try to:

            - Count individual leaves (even if touching / overlapping)
            - Measure **leaf area (cm²)** and an approximate **leaf height (cm)**
            - Compute **area : height** ratio for each leaf
            - Summarize mean + deviation of leaf area
            """
        )

        uploaded = st.file_uploader(
            "Upload grid image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload a sample photo to begin.")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        st.markdown("#### Original image")
        st.image(pil_img, use_container_width=True)

        # --- Grid calibration -------------------------------------------------
        with st.expander("Grid calibration (1 cm squares)", expanded=True):
            auto_spacing = _detect_grid_spacing_px(gray)

            if auto_spacing is not None:
                st.success(
                    f"Auto-detected grid spacing: ~**{auto_spacing:.1f} px per 1 cm**"
                )
                default_px_cm = auto_spacing
            else:
                st.warning(
                    "Could not confidently detect grid spacing. "
                    "Use the slider below to calibrate manually."
                )
                # crude guess: width / 30 (for ~30 cm board)
                default_px_cm = bgr.shape[1] / 30.0

            px_per_cm = st.slider(
                "Pixels per centimetre (adjust if needed)",
                min_value=5.0,
                max_value=80.0,
                value=float(default_px_cm),
                step=0.5,
                help="Drag until 1 cm on the grid visually matches on-screen.",
            )

        # --- Segmentation + measurement --------------------------------------
        mask, markers = _segment_leaves_bgr(bgr)
        leaves = _measure_leaves(mask, px_per_cm=px_per_cm)

        if not leaves:
            st.error("No leaf regions detected. Try checking lighting, focus, or grid visibility.")
            return

        # Build overlay for preview
        overlay = bgr.copy()
        vis = bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours, start=1):
            area_px = cv2.contourArea(cnt)
            if area_px < 2000:
                continue
            color = (0, 255, 0)
            cv2.drawContours(overlay, [cnt], -1, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(
                overlay,
                str(i),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Blend overlay with original
        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Segmented leaves")
            st.image(vis_rgb, use_container_width=True)
        with col_right:
            st.markdown("#### Binary mask")
            st.image(mask, clamp=True, use_container_width=True)

        # --- Build table + summary -------------------------------------------
        df = pd.DataFrame(
            [
                {
                    "Leaf ID": lm.leaf_id,
                    "Area (cm²)": lm.area_cm2,
                    "Height (cm)": lm.height_cm,
                    "Area : height (cm)": lm.area_to_height_cm,
                }
                for lm in leaves
            ]
        ).sort_values("Leaf ID")

        st.markdown("### Leaf measurements")
        st.dataframe(df.style.format(
            {
                "Area (cm²)": "{:.2f}",
                "Height (cm)": "{:.2f}",
                "Area : height (cm)": "{:.2f}",
            }
        ), use_container_width=True)

        areas = df["Area (cm²)"].to_numpy()
        mean_area = float(np.nanmean(areas))
        std_area = float(np.nanstd(areas))

        st.markdown("### Summary")
        st.write(f"- Leaf count: **{len(df)}**")
        st.write(f"- Average leaf area: **{mean_area:.2f} cm²**")
        st.write(f"- Leaf area standard deviation: **{std_area:.2f} cm²**")

        st.caption(
            "Leaf height is approximated from the longer side of the bounding box around each leaf. "
            "Area : height (cm) can help compare compact vs. elongated leaf shapes."
        )
