# phenotyping_tools.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------
# Basic types / dataclasses
# ------------------------------

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class GridDetectionResult:
    px_per_cm: Optional[float]
    squares: List[BBox]


@dataclass
class LeafMeasurement:
    leaf_id: int
    area_px: int
    width_px: int
    height_px: int

    area_cm2: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    area_height_ratio: Optional[float] = None


# ------------------------------
# Grid detection from squares
# ------------------------------

def _find_grid_squares(image_bgr: np.ndarray) -> List[BBox]:
    """
    Identify many near-square contours that likely correspond to the 1 cm grid cells.
    Returns a list of bounding boxes.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strong edges on black grid lines
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 300:
            # Ignore tiny specks
            continue

        side_min = min(w, h)
        side_max = max(w, h)
        if side_min == 0:
            continue

        aspect = side_max / side_min
        if aspect > 1.15:
            # Require near-square
            continue

        squares.append((x, y, w, h))

    return squares


def detect_grid_from_squares(image_bgr: np.ndarray) -> GridDetectionResult:
    """
    Robustly estimate pixels per centimetre using the many square grid cells.

    Assumes each detected square is ~1 cm x 1 cm.
    Uses median side length and robust filtering.

    Returns:
        GridDetectionResult(px_per_cm, squares_used)
    """
    squares = _find_grid_squares(image_bgr)

    if len(squares) < 20:
        return GridDetectionResult(px_per_cm=None, squares=[])

    side_lengths = np.array([(w + h) / 2.0 for (_, _, w, h) in squares], dtype=np.float32)

    median_side = float(np.median(side_lengths))
    abs_dev = np.abs(side_lengths - median_side)
    mad = float(np.median(abs_dev)) if np.any(abs_dev) else 0.0

    if mad == 0:
        px_per_cm = median_side
        used = squares
    else:
        # Robust z-score; keep squares within ~2.5 robust SD of the median
        z = abs_dev / (1.4826 * mad)
        keep = z < 2.5
        used = [sq for sq, k in zip(squares, keep) if k]

        if len(used) < 10:
            # Fallback if we were too aggressive
            used = squares

        side_lengths_used = np.array([(w + h) / 2.0 for (_, _, w, h) in used], dtype=np.float32)
        px_per_cm = float(np.median(side_lengths_used))

    return GridDetectionResult(px_per_cm=px_per_cm, squares=used)


def overlay_grid_squares(image_bgr: np.ndarray, grid: GridDetectionResult) -> np.ndarray:
    """
    Draw the squares used for calibration (red boxes) onto the original image.
    """
    overlay = image_bgr.copy()

    if not grid.squares:
        return overlay

    # Downsample if there are many squares to avoid a red mess
    step = max(1, len(grid.squares) // 120)

    for i, (x, y, w, h) in enumerate(grid.squares):
        if i % step != 0:
            continue
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return overlay


# ------------------------------
# Leaf segmentation
# ------------------------------

def _postprocess_leaf_mask(leaf_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    Take a binary leaf mask (0/255) and:
      - clean noise
      - split touching leaves with watershed on distance transform
      - return final mask, label image, and bounding boxes
    """
    leaf_mask = (leaf_mask > 0).astype("uint8") * 255

    # Clean up small specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    cleaned = np.zeros_like(leaf_mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > 500:
            cleaned[labels == lbl] = 255

    if cv2.countNonZero(cleaned) == 0:
        return cleaned, labels, []

    # Distance transform for splitting
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Lower threshold → more peaks → more separation of overlaps
    _, sure_fg = cv2.threshold(dist_norm, 0.25, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs 3-channel image
    color_for_ws = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(color_for_ws, markers)

    final_mask = np.zeros_like(cleaned)
    final_mask[markers_ws > 1] = 255

    # Label individual leaves
    num_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(
        final_mask, connectivity=8
    )

    boxes: List[BBox] = []
    for lbl in range(1, num_final):
        x = stats_final[lbl, cv2.CC_STAT_LEFT]
        y = stats_final[lbl, cv2.CC_STAT_TOP]
        w = stats_final[lbl, cv2.CC_STAT_WIDTH]
        h = stats_final[lbl, cv2.CC_STAT_HEIGHT]
        area = stats_final[lbl, cv2.CC_STAT_AREA]

        if area < 500:
            continue

        boxes.append((x, y, w, h))

    return final_mask, labels_final, boxes


def _segment_leaves_classical(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    Classical HSV-based segmentation + watershed, improved to split overlaps.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # A bit narrower and shifted towards “lettuce green”
    # You can tweak these if you change crops.
    lower_green = np.array([30, 25, 40])
    upper_green = np.array([85, 255, 255])

    base_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return _postprocess_leaf_mask(mask)


# -------- Optional: learned model backend -------- #

def _segment_leaves_deep(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, List[BBox]]]:
    """
    OPTIONAL: use a trained model for leaf foreground detection.

    This function looks for a file `leaf_model.pth` in the working directory and,
    if PyTorch is available, loads it as a **binary segmentation** UNet-style model
    that returns a single-channel 0-1 mask.

    If anything fails (no torch, no weights, etc.), it returns None and the
    calling code will fall back to the classical pipeline.

    NOTE:
      - You need to provide the trained weights yourself.
      - The simple UNet architecture below must match how you trained it.
    """
    import os

    weights_path = "leaf_model.pth"
    if not os.path.exists(weights_path):
        return None

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        # Torch not installed in this environment
        return None

    # ---- minimal UNet backbone (2-downsample) ---- #
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1 = DoubleConv(3, 32)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(32, 64)
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = DoubleConv(64, 128)

            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv2 = DoubleConv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv1 = DoubleConv(64, 32)

            self.out_conv = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            d1 = self.down1(x)
            p1 = self.pool1(d1)
            d2 = self.down2(p1)
            p2 = self.pool2(d2)

            b = self.bottleneck(p2)

            u2 = self.up2(b)
            u2 = torch.cat([u2, d2], dim=1)
            c2 = self.conv2(u2)

            u1 = self.up1(c2)
            u1 = torch.cat([u1, d1], dim=1)
            c1 = self.conv1(u1)

            out = self.out_conv(c1)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        # weights not compatible / corrupted etc.
        return None

    model.eval()

    # Prepare image for model (HWC BGR → CHW RGB, 0-1 float)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs[0, 0].cpu().numpy() > 0.5).astype("uint8") * 255

    return _postprocess_leaf_mask(mask)


def segment_leaves(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    High-level segmentation entry point:

      1. Try a **trained deep model** (if leaf_model.pth + torch are available).
      2. If that fails, fall back to the improved classical HSV + watershed method.

    This keeps the rest of your phenotyping code unchanged.
    """
    # First, try the learned model
    try:
        deep_result = _segment_leaves_deep(image_bgr)
    except Exception:
        deep_result = None

    if deep_result is not None:
        return deep_result

    # Fallback: classical pipeline
    return _segment_leaves_classical(image_bgr)

# ------------------------------
# Measurement + summary helpers
# ------------------------------

def make_leaf_measurements(
    mask: np.ndarray,
    boxes: List[BBox],
    px_per_cm: Optional[float],
) -> List[LeafMeasurement]:
    """
    Build LeafMeasurement objects from bounding boxes + mask.
    If px_per_cm is None, cm fields will be left as None.
    """
    leaves: List[LeafMeasurement] = []

    for i, (x, y, w, h) in enumerate(sorted(boxes, key=lambda b: b[0])):
        # Area in pixels: count non-zero pixels within bounding box
        roi = mask[y : y + h, x : x + w]
        area_px = int(cv2.countNonZero(roi))

        leaf = LeafMeasurement(
            leaf_id=i + 1,
            area_px=area_px,
            width_px=w,
            height_px=h,
        )

        if px_per_cm is not None and px_per_cm > 0:
            leaf.width_cm = w / px_per_cm
            leaf.height_cm = h / px_per_cm
            leaf.area_cm2 = area_px / (px_per_cm ** 2)

            if leaf.height_cm > 0:
                leaf.area_height_ratio = leaf.area_cm2 / leaf.height_cm

        leaves.append(leaf)

    return leaves


def summarize_leaves(leaves: List[LeafMeasurement], use_cm: bool = True) -> dict:
    """
    Compute summary stats across leaves.
    """
    if not leaves:
        return {}

    if use_cm and leaves[0].area_cm2 is not None:
        areas = np.array([leaf.area_cm2 for leaf in leaves], dtype=float)
        heights = np.array([leaf.height_cm for leaf in leaves], dtype=float)
        unit_area = "cm²"
        unit_height = "cm"
    else:
        areas = np.array([leaf.area_px for leaf in leaves], dtype=float)
        heights = np.array([leaf.height_px for leaf in leaves], dtype=float)
        unit_area = "px²"
        unit_height = "px"

    return {
        "count": len(leaves),
        "mean_area": float(np.mean(areas)),
        "std_area": float(np.std(areas)),
        "mean_height": float(np.mean(heights)),
        "std_height": float(np.std(heights)),
        "unit_area": unit_area,
        "unit_height": unit_height,
    }


# ------------------------------
# Streamlit UI
# ------------------------------

class PhenotypingUI:
    """Rootweiler leaf phenotyping tool."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload an image of lettuce leaves on the **grid board**.  
            Rootweiler will:
            - Detect the **1 cm grid** and convert pixels → cm²  
            - Segment individual leaves (even when they touch)  
            - Report **leaf count, area, height, and area:height ratio**  
            - Summarise average leaf size and variation  
            """
        )

        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload a board image with leaves to begin.")
            return

        # Read image into OpenCV (BGR)
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read image.")
            return

        # --------- Grid calibration ---------
        st.markdown("### Grid calibration (1 cm squares)")

        grid = detect_grid_from_squares(img_bgr)

        if grid.px_per_cm is None:
            st.warning(
                "Could not automatically detect grid spacing from squares. "
                "Measurements will stay in pixels."
            )
        else:
            st.success(
                f"Detected grid spacing: **~{grid.px_per_cm:.1f} pixels per 1 cm** "
                "(from many grid squares)."
            )

        overlay = overlay_grid_squares(img_bgr, grid)
        st.image(
            bgr_to_rgb(overlay),
            caption="Original image with calibration squares (red) used to estimate 1 cm cells",
            use_column_width=True,
        )

        px_per_cm = grid.px_per_cm

        # --------- Leaf segmentation ---------
        st.markdown("### Segmentation overview")

        final_mask, labels_final, boxes = segment_leaves(img_bgr)

        # Prepare visuals
        img_rgb = bgr_to_rgb(img_bgr)
        h, w = img_rgb.shape[:2]

        # Original + leaf boxes overlay
        vis_boxes = img_rgb.copy()
        for i, (x, y, bw, bh) in enumerate(sorted(boxes, key=lambda b: b[0])):
            cv2.rectangle(vis_boxes, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                vis_boxes,
                str(i + 1),
                (x + 3, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original image**")
            st.image(img_rgb, use_column_width=True)
        with col2:
            st.markdown("**Leaf labels**")
            st.image(vis_boxes, use_column_width=True)
        with col3:
            st.markdown("**Binary mask**")
            st.image(final_mask, clamp=True, use_column_width=True)

        # --------- Measurements ---------
        leaves = make_leaf_measurements(final_mask, boxes, px_per_cm)

        if not leaves:
            st.warning("No leaves detected. Check that the board and leaves are clearly visible.")
            return

        use_cm = px_per_cm is not None and px_per_cm > 0

        summary = summarize_leaves(leaves, use_cm=use_cm)

        st.markdown("### Leaf measurements")

        # Summary text
        if summary:
            unit_area = summary["unit_area"]
            unit_height = summary["unit_height"]

            st.write(
                f"- Leaf count: **{summary['count']}**\n"
                f"- Mean leaf area: **{summary['mean_area']:.2f} {unit_area}** "
                f"(± {summary['std_area']:.2f})\n"
                f"- Mean leaf height: **{summary['mean_height']:.2f} {unit_height}** "
                f"(± {summary['std_height']:.2f})"
            )

        # Detailed table
        rows = []
        for leaf in leaves:
            if use_cm:
                rows.append(
                    {
                        "Leaf ID": leaf.leaf_id,
                        "Area (cm²)": round(leaf.area_cm2, 2) if leaf.area_cm2 is not None else None,
                        "Height (cm)": round(leaf.height_cm, 2) if leaf.height_cm is not None else None,
                        "Width (cm)": round(leaf.width_cm, 2) if leaf.width_cm is not None else None,
                        "Area / height": round(leaf.area_height_ratio, 2)
                        if leaf.area_height_ratio is not None
                        else None,
                    }
                )
            else:
                rows.append(
                    {
                        "Leaf ID": leaf.leaf_id,
                        "Area (px²)": leaf.area_px,
                        "Height (px)": leaf.height_px,
                        "Width (px)": leaf.width_px,
                    }
                )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.caption(
            "Area is calculated from the leaf mask in the bounding box. "
            "Height and width come from the bounding box in the calibrated image. "
            "Area:height gives a quick sense of leaf 'fullness' vs. elongation."
        )
