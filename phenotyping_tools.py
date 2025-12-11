# phenotyping_tools.py

import io
import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# Import the library exactly as requested
from inference_sdk import InferenceHTTPClient

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
# Grid Calibration Helper
# ---------------------------------------------------------------------
def _detect_grid_squares(image_bgr: np.ndarray) -> Tuple[Optional[float], List[Tuple[int, int, int, int]]]:
    """
    Detect square-ish contours and estimate pixel area for a 1 cm² grid square.
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
    consistent_squares = [s for s, area in zip(squares, areas) if 0.7 * median_area < area < 1.3 * median_area]

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
# Roboflow SDK Integration (Updated to match your snippet)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation workflow...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    # 1. Get and Clean API Key
    api_key = None
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        return None, False

    # CRITICAL FIX: Remove accidental whitespace that causes Auth Failures
    api_key = api_key.strip()

    # 2. Setup Client
    try:
        # We explicitly use the URL from your snippet
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
    except Exception as e:
        # This usually only fails if libraries are missing
        st.error(f"Failed to initialize SDK client: {e}")
        return None, True

    # 3. Save Temp File
    tmp_path = "tmp_phenotype_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    # 4. Run Workflow (Exact snippet logic)
    result = None
    try:
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="leafy",
            images={
                "image": tmp_path
            },
            parameters={
                "output_message": "Your model is being initialized, try again in a few seconds."
            },
            use_cache=True
        )
    except Exception as e:
        # Check specifically for Auth errors
        error_str = str(e)
        if "401" in error_str or "403" in error_str or "Unauthorized" in error_str:
             st.error("❌ Roboflow Authentication Failed. Please check the API Key in secrets.toml for typos or spaces.")
        else:
             st.error(f"❌ Roboflow Workflow Error: {e}")
        return None, True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 5. Process Results
    if not isinstance(result, list) or len(result) == 0:
        return None, True

    # Get the first result object
    obj = result[0]
    
    # Extract predictions from "output2"
    preds = obj.get("output2")
    
    if not isinstance(preds, list) or len(preds) == 0:
        return None, True

    return {"predictions": preds}, True

# ---------------------------------------------------------------------
# Mask & Measurement Logic (Unchanged)
# ---------------------------------------------------------------------
def _mask_from_roboflow_predictions(image_shape: Tuple[int, int, int], predictions: List[dict]) -> np.ndarray:
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list): continue
        poly = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32)
        if poly.shape[0] < 3: continue
        cv2.fillPoly(mask, [poly], 255)
    return mask

def _color_based_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) 
    return mask

def _measure_leaves(mask: np.ndarray, pixels_per_cm2: float) -> List[LeafMeasurement]:
    if pixels_per_cm2 <= 0: raise ValueError("pixels_per_cm2 must be positive")
    pixels_per_cm = float(np.sqrt(pixels_per_cm2))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    measurements: List[LeafMeasurement] = []
    MIN_AREA_PX = 500
    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < MIN_AREA_PX: continue
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        area_cm2 = area_px / pixels_per_cm2
        height_cm = h / pixels_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else 0.0
        measurements.append(LeafMeasurement(
            id=len(measurements) + 1,
            area_cm2=round(area_cm2, 2),
            height_cm=round(height_cm, 2),
            area_height_ratio=round(ratio, 2),
        ))
    return measurements

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
class PhenotypingUI:
    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")
        st.markdown("Upload a **phenotyping photo** taken on the 1&nbsp;cm grid board.")
        uploaded = st.file_uploader("Upload phenotyping image", type=["jpg", "jpeg", "png"])

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 1. Grid Calibration
        with st.spinner("Step 1/3: Calibrating grid for scale..."):
            pixels_per_cm2, grid_boxes = _detect_grid_squares(img_bgr)
            
        if pixels_per_cm2 is None:
            st.error("❌ Calibration Failed: Could not detect grid.")
            return

        # 2. Segmentation
        with st.spinner("Step 2/3: Segmenting leaves..."):
            rf_result, attempted_rf = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.caption("Leaf segmentation method: **Roboflow instance segmentation workflow (Success)**.")
        else:
            mask = _color_based_mask(img_bgr)
            if attempted_rf:
                st.warning("⚠️ Roboflow workflow failed. Using color-based fallback.")
            else:
                st.caption("Leaf segmentation method: **Simple color-based fallback**.")

        # 3. Measurements
        mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
        leaf_measurements = _measure_leaves(mask_bin, pixels_per_cm2)
        
        if not leaf_measurements:
            st.error("❌ No leaves detected.")
            return

        st.success(f"✅ Found and measured **{len(leaf_measurements)}** leaves.")
        
        c1, c2 = st.columns(2)
        with c1: st.image(img_rgb, caption="Original", use_container_width=True)
        with c2: st.image(mask_bin, caption="Mask", use_container_width=True)

        # DataFrame
        df = pd.DataFrame([{
            "ID": m.id, 
            "Area (cm²)": m.area_cm2, 
            "Height (cm)": m.height_cm, 
            "Area/Height Ratio": m.area_height_ratio
        } for m in leaf_measurements])
        
        st.dataframe(df, use_container_width=True)
        
        # Download
        def convert_df(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='LeafMeasurements')
            return output.getvalue()

        st.download_button(
            label="Download Excel",
            data=convert_df(df),
            file_name="leaf_measurements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
