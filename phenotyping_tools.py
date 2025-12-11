# phenotyping_tools.py

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
# Grid calibration
# ---------------------------------------------------------------------
def _detect_grid_squares(image_bgr: np.ndarray) -> Tuple[Optional[float], List[Tuple[int, int, int, int]]]:
    """
    Detect square-ish contours and estimate pixel area for a 1 cm² grid square.
    (Implementation remains the same)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.8 < aspect_ratio < 1.2:
                 squares.append((x, y, w, h))

    if len(squares) < 5: 
        return None, []

    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    consistent_squares = [
        s for s, area in zip(squares, areas) if 0.7 * median_area < area < 1.3 * median_area
    ]

    if not consistent_squares:
        return None, []
    
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
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------
# Roboflow workflow wrapper
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation workflow...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    """
    Call the Roboflow workflow 'leafy'.
    Returns: (result_dict, was_attempted_with_key)
    """
    api_key = None
    # Check for API key in st.secrets
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        return None, False  # Key not found, attempt was NOT made

    try:
        # Initialize client with the found key
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
    except Exception:
        return None, True # Initialization failed, attempt WAS made

    # Write bytes to a temporary file
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
        # st.exception(e) # Optional: uncomment for deep debugging
        return None, True # Workflow call failed, attempt WAS made
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
        return None, True

    # Check for "output2" and "predictions"
    output2 = obj.get("output2")
    if not isinstance(output2, dict):
        return None, True

    preds = output2.get("predictions")
    if not isinstance(preds, list) or len(preds) == 0:
        return None, True

    return {"predictions": preds}, True # Success


def _mask_from_roboflow_predictions(
    image_shape: Tuple[int, int, int], predictions: List[dict]
) -> np.ndarray:
    """
    Build a binary mask from Roboflow instance segmentation polygons.
    (Implementation remains the same)
    """
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list):
            continue
        poly = np.array([[int(p["x"]), int(p["y"])] for p in pts if "x" in p and "y" in p], dtype=np.int32)
        if poly.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [poly], 255)

    return mask


# ---------------------------------------------------------------------
# Fallback color-based segmentation
# ---------------------------------------------------------------------
def _color_based_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Simple green-ish segmentation as a fallback."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) 

    return mask


# ---------------------------------------------------------------------
# Leaf measurement helpers
# ---------------------------------------------------------------------
def _measure_leaves(mask: np.ndarray, pixels_per_cm2: float) -> List[LeafMeasurement]:
    """
    Use connected components on the binary mask to measure each leaf.
    (Implementation remains the same)
    """
    if pixels_per_cm2 <= 0:
        raise ValueError("pixels_per_cm2 must be positive")

    pixels_per_cm = float(np.sqrt(pixels_per_cm2))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    measurements: List[LeafMeasurement] = []
    
    MIN_AREA_PX = 500

    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        
        if area_px < MIN_AREA_PX: 
            continue

        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        height_px = h 

        area_cm2 = area_px / pixels_per_cm2
        height_cm = height_px / pixels_per_cm
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

        st.markdown(
            """
            Upload a **phenotyping photo** taken on the 1&nbsp;cm grid board.
            """
        )
        
        # Adding an image tag for instructional value
        st.markdown("")


        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

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

        with st.expander("Show grid squares used for calibration", expanded=False):
            overlay = _overlay_grid_boxes(img_bgr, grid_boxes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Detected 1 cm² grid squares", use_container_width=True)

        # --- Segmentation: Roboflow first, fallback to color ---
        with st.spinner("Step 2/3: Segmenting leaves..."):
            # The function now returns (result, attempted_flag)
            rf_result, attempted_rf = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.caption("Leaf segmentation method: **Roboflow instance segmentation workflow (Success)**.")
        else:
            mask = _color_based_mask(img_bgr)
            
            # Show a specific message based on why Roboflow failed
            if attempted_rf:
                st.warning("⚠️ Roboflow workflow failed or returned no predictions. Using color-based fallback.")
            else:
                st.caption("Leaf segmentation method: **Simple color-based fallback (Roboflow API key not configured)**.")


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
            st.image(mask_bin, caption="Binary Leaf Mask", use_container_width=True)

        # --- Table + summary ---
        st.markdown("### Leaf Measurements")

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
        
        st.markdown("#### Summary Statistics")
        summary_df = df[["Area (cm²)", "Height (cm)", "Area/Height Ratio"]].describe().T
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("#### Detailed Measurements")
        st.dataframe(df, use_container_width=True)

        # --- Download Button ---
        @st.cache_data
        def convert_df_to_excel(df):
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
