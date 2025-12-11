# phenotyping_tools.py

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
@dataclass
class LeafMeasurement:
    id: int
    area_cm2: float
    height_cm: float
    area_height_ratio: float


# ---------------------------------------------------------------------
# Grid calibration – using your older "square detection" style
# ---------------------------------------------------------------------
def _detect_grid_squares(image_bgr: np.ndarray) -> Tuple[Optional[float], List[Tuple[int, int, int, int]]]:
    """
    Detect square-ish contours and estimate pixel area for a 1 cm² grid square.

    Returns:
        pixels_per_cm2 (float | None), list of chosen square bounding boxes (for overlay).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to help with Canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding might also work well, but Canny is often used for contours
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Check if the contour has 4 corners and is relatively large
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's roughly square (aspect ratio close to 1)
            aspect_ratio = float(w)/h
            if 0.8 < aspect_ratio < 1.2:
                 squares.append((x, y, w, h))

    if len(squares) < 5: # Lowering the required count for robustness
        return None, []

    # Use median area to filter consistent squares
    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    # Keep squares whose area is closest to median, filtering outliers
    # Use squares within 30% of the median area
    consistent_squares = [
        s for s, area in zip(squares, areas) if 0.7 * median_area < area < 1.3 * median_area
    ]

    if not consistent_squares:
        return None, []
    
    # Use up to 20 best squares for the final average
    squares_sorted = sorted(consistent_squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    chosen = squares_sorted[:20]

    if not chosen:
        return None, []

    avg_area = float(np.mean([w * h for (_, _, w, h) in chosen]))
    pixels_per_cm2 = avg_area

    return pixels_per_cm2, chosen


def _overlay_grid_boxes(image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy of the image with green rectangles drawn on detected grid squares."""
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        # Draw a green rectangle with thickness 2
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------
# Roboflow workflow wrapper
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation workflow...")
def _run_roboflow_workflow(image_bytes: bytes) -> Optional[Dict[str, object]]:
    """
    Call the Roboflow workflow 'leafy' and return a dict with:
      - 'predictions': list of polygon predictions
    Returns None if anything fails.
    """
    api_key = None
    # Preferred: secrets
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        # st.info is removed here to avoid multiple messages on reruns
        return None

    try:
        # Use an environment variable for the base URL if needed, but the serverless endpoint is correct
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
    except Exception:
        # st.info is removed here
        return None

    # Write bytes to a temporary file; run_workflow expects a file path
    tmp_path = "tmp_phenotype_image.jpg"
    try:
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)

        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="leafy",
            images={"image": tmp_path},
        )
    except Exception as e:
        # st.info is removed here
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Process the result structure
    obj = None
    if isinstance(result, list) and len(result) > 0:
        obj = result[0]
    elif isinstance(result, dict):
        obj = result
    
    if not isinstance(obj, dict):
        # st.info is removed here
        return None

    # Your workflow JSON uses "output2": { "image": {...}, "predictions": [...] }
    output2 = obj.get("output2")
    if not isinstance(output2, dict):
        # st.info is removed here
        return None

    preds = output2.get("predictions")
    if not isinstance(preds, list) or len(preds) == 0:
        # st.info is removed here
        return None

    return {"predictions": preds}


def _mask_from_roboflow_predictions(
    image_shape: Tuple[int, int, int], predictions: List[dict]
) -> np.ndarray:
    """
    Build a binary mask from Roboflow instance segmentation polygons.
    All leaves are set to 255 in a single-channel mask.
    """
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list):
            continue
        # Ensure all points are integers and convert to a numpy array for cv2.fillPoly
        poly = np.array([[int(p["x"]), int(p["y"])] for p in pts if "x" in p and "y" in p], dtype=np.int32)
        if poly.shape[0] < 3:
            continue
        # Fill the polygon with white (255)
        cv2.fillPoly(mask, [poly], 255)

    return mask


# ---------------------------------------------------------------------
# Fallback color-based segmentation (simple HSV threshold)
# ---------------------------------------------------------------------
def _color_based_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Simple green-ish segmentation as a fallback when Roboflow is not available."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # A standard range for green in HSV: H[30-90], S[40-255], V[40-255]
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8) # Slightly larger kernel for better smoothing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove small specks

    return mask


# ---------------------------------------------------------------------
# Leaf measurement helpers
# ---------------------------------------------------------------------
def _measure_leaves(mask: np.ndarray, pixels_per_cm2: float) -> List[LeafMeasurement]:
    """
    Use connected components on the binary mask to measure each leaf.
    """
    if pixels_per_cm2 <= 0:
        raise ValueError("pixels_per_cm2 must be positive")

    pixels_per_cm = float(np.sqrt(pixels_per_cm2))

    # Use connectedComponentsWithStats to get better bounding boxes and area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    measurements: List[LeafMeasurement] = []
    
    # Minimum area threshold in pixels (e.g., 500 pixels)
    MIN_AREA_PX = 500

    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        
        if area_px < MIN_AREA_PX: 
            continue

        # Extract bounding box stats: x, y, width, height
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        height_px = h # Using the height of the bounding box

        # Conversion to cm
        area_cm2 = area_px / pixels_per_cm2
        height_cm = height_px / pixels_per_cm
        
        # Calculate ratio, guard against division by zero
        ratio = area_cm2 / height_cm if height_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                id=len(measurements) + 1,
                area_cm2=round(area_cm2, 2),
                height_cm=round(height_cm, 2),
                area_height_ratio=round(ratio, 2),
            )
        )

    return measurements


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
class PhenotypingUI:
    """Leaf phenotyping tool using Roboflow (preferred) + grid calibration."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")
        
        # Adding a visual aid description
        st.markdown(
            """
            Upload a **phenotyping photo** taken on the 1&nbsp;cm grid board.  
            

            Rootweiler will:

            1. Detect the grid and convert pixels to **cm²** (Grid Calibration)
            2. Segment each leaf (via Roboflow or color-based fallback)
            3. Measure leaf **area**, **height**, and **area : height** ratio
            """
        )

        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read and prepare image formats
        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # --- Grid calibration ---
        with st.spinner("Step 1/3: Calibrating grid for scale..."):
            pixels_per_cm2, grid_boxes = _detect_grid_squares(img_bgr)
            
        if pixels_per_cm2 is None:
            st.error(
                "❌ **Calibration Failed:** Could not detect enough 1 cm squares on the board. "
                "Check that the grid is in view and in focus."
            )
            return

        pixels_per_cm = float(np.sqrt(pixels_per_cm2))

        st.success(
            f"✅ **Calibration Success:** ~{pixels_per_cm:.1f} pixels per cm "
            f"→ ~{pixels_per_cm2:.0f} pixels per cm²"
        )

        # Show overlay of squares used for calibration
        with st.expander("Show grid squares used for calibration", expanded=False):
            overlay = _overlay_grid_boxes(img_bgr, grid_boxes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Detected 1 cm² grid squares", use_container_width=True)

        # --- Segmentation: Roboflow first, fallback to color ---
        with st.spinner("Step 2/3: Segmenting leaves..."):
            rf_result = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.caption("Leaf segmentation method: Roboflow instance segmentation workflow (preferred).")
        else:
            mask = _color_based_mask(img_bgr)
            st.caption("Leaf segmentation method: Simple color-based fallback (No Roboflow API key found).")

        # Ensure binary mask
        mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)

        # --- Measurements ---
        with st.spinner("Step 3/3: Measuring individual leaves..."):
            try:
                leaf_measurements = _measure_leaves(mask_bin, pixels_per_cm2)
            except ValueError as e:
                st.error(f"❌ Measurement Error: {str(e)}")
                return
            
        if not leaf_measurements:
            st.error("❌ **Segmentation Failed:** No leaves detected in the mask. Check the image and segmentation.")
            return

        st.success(f"✅ Found and measured **{len(leaf_measurements)}** leaves.")

        # --- Visuals: original, mask ---
        st.markdown("### Segmentation Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original Image", use_container_width=True)
        with c2:
            st.image(mask_bin, caption="Binary Leaf Mask ", use_container_width=True)

        # --- Table + summary ---
        st.markdown("### Leaf Measurements")

        # --- DataFrame Creation (Completed Section) ---
        df = pd.DataFrame(
            [
                {
                    "ID": m.id,
                    "Area (cm²)": m.area_cm2,
                    "Height (cm)": m.height_cm,
                    "Area/Height Ratio": m.area_height_ratio,
                }
                for m in leaf_measurements
            ]
        )
        
        # --- Summary Statistics ---
        st.markdown("#### Summary Statistics")
        # Use only the measured numeric columns for the summary
        summary_df = df[["Area (cm²)", "Height (cm)", "Area/Height Ratio"]].describe().T
        st.dataframe(summary_df, use_container_width=True)

        # --- Detailed Table ---
        st.markdown("#### Detailed Measurements")
        st.dataframe(df, use_container_width=True)

        # --- Download Button ---
        @st.cache_data
        def convert_df_to_excel(df):
            # Use io.BytesIO for in-memory Excel file creation
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='LeafMeasurements')
            return output.getvalue()

        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="Download measurements (Excel)",
            data=excel_data,
            file_name="leaf_phenotype_measurements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
