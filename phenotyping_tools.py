# phenotyping.py

import io
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import streamlit as st

# OpenCV is only used inside functions so importing it here is okay
import cv2

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------
# Roboflow integration helpers
# -----------------------------


def _call_roboflow_workflow(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Try to call the Roboflow workflow using the inference SDK.
    Handles different SDK expectations for the `images` argument.

    Returns the raw workflow result dict, or None if anything fails.
    """
    api_key = st.secrets.get("ROBOFLOW_API_KEY", None)
    if api_key is None:
        st.info("No ROBOFLOW_API_KEY found in Streamlit secrets – using color-based segmentation instead.")
        return None

    if InferenceHTTPClient is None:
        st.info("inference-sdk is not installed. Run `pip install inference-sdk` to enable Roboflow.")
        return None

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Write bytes to a temporary file (most SDK versions like file paths)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # We try two shapes for `images`: dict and list-of-dict.
    # This avoids the `'list' object has no attribute 'items'` issues
    # if the installed SDK version changed its expectations.
    attempts = [
        {"image": tmp_path},         # dict form
        [{"image": tmp_path}],       # list-of-dict form
    ]

    last_error = None
    for images_payload in attempts:
        try:
            result = client.run_workflow(
                workspace_name="rootweiler",
                workflow_id="find-leaves-3",
                images=images_payload,
            )
            # If we get here, call succeeded
            os.unlink(tmp_path)
            return result
        except Exception as e:
            last_error = e

    os.unlink(tmp_path)
    st.warning(
        "Roboflow workflow failed or is not configured correctly. "
        "Falling back to color-based segmentation.\n\n"
        f"Details: {last_error}"
    )
    return None


def _extract_mask_from_workflow_result(
    result: Dict[str, Any], image_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Try to turn the Roboflow workflow result into a binary leaf mask.

    Assumes the workflow's final step returns either:
      - a list of bounding boxes in `result['predictions']`, or
      - a dict containing `predictions` as a key, or
      - polygon segmentations under `points` or `segments`.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Different workflow configs can wrap predictions differently,
    # so we try a few common layouts.
    preds = None

    if isinstance(result, dict):
        if "predictions" in result and isinstance(result["predictions"], list):
            preds = result["predictions"]
        else:
            # Sometimes workflows return a dict of steps; try to find a step with `predictions`
            for v in result.values():
                if isinstance(v, dict) and isinstance(v.get("predictions"), list):
                    preds = v["predictions"]
                    break

    if not preds:
        return None

    for p in preds:
        # 1) Polygon segmentation (if available)
        if "points" in p and isinstance(p["points"], list) and len(p["points"]) >= 3:
            pts = np.array([[pt["x"], pt["y"]] for pt in p["points"]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        # 2) Or bounding box (x, y, width, height, often center-based)
        elif all(k in p for k in ("x", "y", "width", "height")):
            cx = float(p["x"])
            cy = float(p["y"])
            bw = float(p["width"])
            bh = float(p["height"])
            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(w - 1, cx + bw / 2))
            y2 = int(min(h - 1, cy + bh / 2))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if mask.sum() == 0:
        return None

    return mask


# -----------------------------
# Fallback: color-based mask
# -----------------------------


def _color_based_leaf_mask_bgr(bgr_image: np.ndarray) -> np.ndarray:
    """
    Simple HSV-based segmentation to find green leaf areas in a BGR OpenCV image.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # Tuned fairly loosely for green; adjust if needed
    lower_green = np.array([20, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up with morphology
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Optional watershed-based refinement
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr_image.copy(), markers)

    final_mask = np.zeros_like(mask)
    final_mask[markers > 1] = 255
    return final_mask


# -----------------------------
# Grid square / scale detection
# -----------------------------


def _find_grid_square(bgr_image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
    """
    Detect grid squares in the background and estimate pixels per cm².

    Returns:
        (square_bbox, pixels_per_cm2)
        - square_bbox: (x, y, w, h) of the representative square used for scaling
        - pixels_per_cm2: estimated area in pixels² corresponding to 1 cm²
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Rough "square-ish" check
        if abs(w - h) < 10 and w * h > 500:
            squares.append((x, y, w, h))

    if len(squares) < 10:
        return None, None

    areas = [w * h for _, _, w, h in squares]
    median_area = float(np.median(areas))

    # Choose squares closest to the median
    squares_sorted = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    best_squares = squares_sorted[:20]
    avg_area = float(np.mean([w * h for _, _, w, h in best_squares]))

    # Use the most "typical" square for overlay
    representative_square = best_squares[0]
    pixels_per_cm2 = avg_area   # each cm² ≈ this many pixels
    return representative_square, pixels_per_cm2


# -----------------------------
# Leaf metrics
# -----------------------------


def _measure_leaves(mask: np.ndarray, pixels_per_cm2: Optional[float]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Measure leaf areas and bounding boxes from a binary mask.

    Returns:
        per_leaf: list of dicts with per-leaf metrics
        summary: dict with aggregate metrics
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], {}

    # Find "plant height" (vertical extent of all leaves combined)
    all_y = []
    all_y2 = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        all_y.append(y)
        all_y2.append(y + h)
    plant_height_px = max(all_y2) - min(all_y) if all_y and all_y2 else 0

    per_leaf: List[Dict[str, float]] = []

    for idx, c in enumerate(contours, start=1):
        area_px = float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)

        if pixels_per_cm2 and pixels_per_cm2 > 0:
            area_cm2 = area_px / pixels_per_cm2
            leaf_height_cm = h / np.sqrt(pixels_per_cm2)
            leaf_width_cm = w / np.sqrt(pixels_per_cm2)
            plant_height_cm = plant_height_px / np.sqrt(pixels_per_cm2) if plant_height_px > 0 else 0.0
        else:
            area_cm2 = float("nan")
            leaf_height_cm = float("nan")
            leaf_width_cm = float("nan")
            plant_height_cm = float("nan")

        if plant_height_cm and plant_height_cm > 0:
            leaf_to_height = leaf_height_cm / plant_height_cm
        else:
            leaf_to_height = float("nan")

        per_leaf.append(
            {
                "leaf_id": idx,
                "area_cm2": area_cm2,
                "height_cm": leaf_height_cm,
                "width_cm": leaf_width_cm,
                "leaf_to_plant_height_ratio": leaf_to_height,
            }
        )

    # Summary stats
    areas = [l["area_cm2"] for l in per_leaf if np.isfinite(l["area_cm2"])]
    if areas:
        avg_area = float(np.mean(areas))
        std_area = float(np.std(areas))
    else:
        avg_area = float("nan")
        std_area = float("nan")

    summary = {
        "n_leaves": len(per_leaf),
        "avg_leaf_area_cm2": avg_area,
        "std_leaf_area_cm2": std_area,
        "plant_height_cm": plant_height_cm if pixels_per_cm2 else float("nan"),
    }

    return per_leaf, summary


def _draw_overlay(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    square_bbox: Optional[BBox],
) -> np.ndarray:
    """
    Draw the grid square used for scale + leaf contours/bounding boxes.
    Returns a BGR image with overlays.
    """
    overlay = bgr_image.copy()

    # Draw grid square (if found)
    if square_bbox is not None:
        x, y, w, h = square_bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            "1 cm^2 (estimated)",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Leaf contours and bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(
            overlay,
            str(idx),
            (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


# -----------------------------
# Streamlit UI
# -----------------------------


class PhenotypingUI:
    """Streamlit UI wrapper for the leaf phenotyping tool."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a **top-down image** of a small lettuce plant on a grid background.  
            Rootweiler will:

            - Detect leaves (using your Roboflow workflow where available)  
            - Estimate **leaf area** in cm² using the grid as scale  
            - Report per-leaf size and **leaf-to-plant-height** ratios  
            - Show an overlay with the grid square used for scaling
            """
        )

        uploaded = st.file_uploader(
            "Upload a single plant image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded is None:
            st.info("Upload a JPEG or PNG image to begin.")
            return

        # Read bytes & open with PIL
        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.markdown("#### Original image")
        st.image(pil_img, use_container_width=True)

        if st.button("Run phenotyping", type="primary"):
            cls._run_phenotyping(pil_img, image_bytes)

    @classmethod
    def _run_phenotyping(cls, pil_img: Image.Image, image_bytes: bytes):
        # Convert PIL image to OpenCV BGR
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # --- Step 1: grid square detection / scale ---
        square_bbox, pixels_per_cm2 = _find_grid_square(bgr)

        if pixels_per_cm2 is None:
            st.warning(
                "Could not reliably detect grid squares for scale. "
                "Leaf areas will be reported in pixels² instead of cm²."
            )

        # --- Step 2: Roboflow segmentation or fallback ---
        workflow_result = _call_roboflow_workflow(image_bytes)
        if workflow_result is not None:
            mask = _extract_mask_from_workflow_result(workflow_result, bgr.shape)
            if mask is None or mask.sum() == 0:
                st.warning(
                    "Roboflow workflow returned no usable leaf mask. "
                    "Falling back to color-based segmentation."
                )
                mask = _color_based_leaf_mask_bgr(bgr)
        else:
            # Roboflow not available / failed
            mask = _color_based_leaf_mask_bgr(bgr)

        if mask.sum() == 0:
            st.error("No leaf area detected in this image.")
            return

        # --- Step 3: measure leaves ---
        per_leaf, summary = _measure_leaves(mask, pixels_per_cm2)

        if not per_leaf:
            st.error("No distinct leaves found after segmentation.")
            return

        # --- Step 4: overlay image ---
        overlay_bgr = _draw_overlay(bgr, mask, square_bbox)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("#### Segmentation overlay")
        st.image(overlay_rgb, caption="Detected leaves and scale square", use_container_width=True)

        # --- Metrics ---
        st.markdown("#### Per-leaf metrics")
        df_leaves = pd.DataFrame(per_leaf)
        st.dataframe(df_leaves.style.format(
            {
                "area_cm2": "{:.2f}",
                "height_cm": "{:.2f}",
                "width_cm": "{:.2f}",
                "leaf_to_plant_height_ratio": "{:.2f}",
            }
        ), use_container_width=True)

        st.markdown("#### Summary")
        if summary:
            n = summary.get("n_leaves", 0)
            avg_area = summary.get("avg_leaf_area_cm2", float("nan"))
            std_area = summary.get("std_leaf_area_cm2", float("nan"))
            plant_height_cm = summary.get("plant_height_cm", float("nan"))

            st.write(f"- Leaf count (objects): **{n}**")
            if pixels_per_cm2:
                st.write(f"- Average leaf area: **{avg_area:.2f} cm²**")
                st.write(f"- Leaf area standard deviation: **{std_area:.2f} cm²**")
                st.write(f"- Estimated plant height: **{plant_height_cm:.2f} cm**")
            else:
                st.write(
                    "Scale not available; areas are approximate and not in cm². "
                    "Try improving the grid visibility."
                )

            st.caption(
                "Leaf-to-plant-height ratio helps compare leaf size relative to plant size, "
                "useful for tracking compactness vs. stretch between treatments."
            )