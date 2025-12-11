# phenotyping_tools.py

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None  # handled gracefully below


# -----------------------------
# Config
# -----------------------------

ROBOFLOW_WORKSPACE = "rootweiler"
ROBOFLOW_WORKFLOW_ID = "leafy"   # your workflow id


@dataclass
class LeafMeasurement:
    leaf_id: int
    area_cm2: float
    height_cm: float
    width_cm: float
    area_to_height: float


# -----------------------------
# Grid calibration (your logic)
# -----------------------------

def calculate_pixels_per_cm2(image_bgr: np.ndarray) -> Optional[float]:
    """
    Estimate pixels per cm² using the printed grid.
    Each grid square is 1 cm².

    Logic adapted from your earlier function:
    - Find many roughly square contours
    - Filter by size and aspect ratio
    - Use median area of those squares as pixels_per_cm²
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # near-square, not tiny, not huge
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    if len(squares) < 20:
        return None

    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    # Keep ~20 closest to median
    squares_sorted = sorted(
        squares,
        key=lambda s: abs((s[2] * s[3]) - median_area),
    )[:20]

    avg_area = np.mean([w * h for (_, _, w, h) in squares_sorted])

    return float(avg_area)


# -----------------------------
# Simple fallback segmentation
# -----------------------------

def simple_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Simple HSV-based leaf segmentation as fallback if Roboflow fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([0, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


# -----------------------------
# Roboflow call & mask creation
# -----------------------------

def _get_roboflow_client() -> Optional[InferenceHTTPClient]:
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("ROBOFLOW_API_KEY", None)
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key or InferenceHTTPClient is None:
        return None

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    return client


def _run_roboflow_workflow(image_bytes: bytes) -> Optional[dict]:
    """
    Send image to the Roboflow workflow and return the first element
    of the decoded output (your JSON).
    """
    client = _get_roboflow_client()
    if client is None:
        return None

    # Write bytes to a temporary file – the SDK likes file paths
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": tmp_path},
            parameters={"output_message": "rootweiler phenotyping"},
            use_cache=True,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Expecting a list with one dict like the JSON you pasted
    if not isinstance(result, list) or not result:
        return None

    return result[0]


def mask_from_roboflow_output(rf_output: dict) -> Optional[np.ndarray]:
    """
    Build a binary mask from Roboflow workflow output.

    Your JSON looks like:
        [
          {
            "output2": {
              "image": {"width": W, "height": H},
              "predictions": [
                 {"points": [{"x":..., "y":...}, ...], "class": "leaf", ...},
                 ...
              ]
            },
            "output": {... base64 mask ...}
          }
        ]
    """
    if "output2" not in rf_output:
        return None

    model_out = rf_output["output2"]
    img_meta = model_out.get("image", {})
    preds = model_out.get("predictions", [])

    width = int(img_meta.get("width", 0))
    height = int(img_meta.get("height", 0))

    if width <= 0 or height <= 0 or not preds:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)

    for det in preds:
        pts_list = det.get("points", [])
        if not pts_list:
            continue
        pts = np.array([[p["x"], p["y"]] for p in pts_list], dtype=np.int32)
        pts = pts.reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    # Clean up a little
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# -----------------------------
# Leaf measurement
# -----------------------------

def measure_leaves(mask: np.ndarray, pixels_per_cm2: float) -> List[LeafMeasurement]:
    """
    Measure each connected leaf region in the mask.

    Returns list of LeafMeasurement.
    """
    px_per_cm = float(np.sqrt(pixels_per_cm2))
    labeled, num_labels = cv2.connectedComponents(mask)

    measurements: List[LeafMeasurement] = []

    for label_id in range(1, num_labels):
        comp_mask = (labeled == label_id).astype(np.uint8)
        area_px = int(comp_mask.sum())
        if area_px < 2000:  # filter tiny specks
            continue

        ys, xs = np.where(comp_mask > 0)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        h_px = y_max - y_min + 1
        w_px = x_max - x_min + 1

        area_cm2 = area_px / pixels_per_cm2
        height_cm = h_px / px_per_cm
        width_cm = w_px / px_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else np.nan

        measurements.append(
            LeafMeasurement(
                leaf_id=label_id,
                area_cm2=area_cm2,
                height_cm=height_cm,
                width_cm=width_cm,
                area_to_height=ratio,
            )
        )

    return measurements


# -----------------------------
# Streamlit UI
# -----------------------------

class PhenotypingUI:
    """Rootweiler phenotyping tools: leaf segmentation + measurements."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload an image on the 1 cm grid board.  
            Rootweiler will:

            - Segment each **leaf** (using your Roboflow instance segmentation model)
            - Use the printed grid to convert **pixels → cm²**
            - Report leaf count, area, height, width, and area/height ratio
            - Summarize average leaf size and variation
            """
        )

        uploaded = st.file_uploader(
            "Upload a grid-board image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            return

        image_bytes = uploaded.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not read image.")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- Grid calibration ---
        pixels_per_cm2 = calculate_pixels_per_cm2(img_bgr)

        if pixels_per_cm2 is None:
            st.error(
                "Could not detect enough grid squares to calibrate pixels per cm. "
                "Check that the board is visible and in focus."
            )
            return

        px_per_cm = float(np.sqrt(pixels_per_cm2))

        st.markdown(
            f"**Calibrated scale:** ~{pixels_per_cm2:,.1f} pixels per cm² "
            f"(~{px_per_cm:,.1f} pixels per cm)."
        )

        if st.button("Run leaf analysis", type="primary"):
            cls._run_analysis(image_bytes, img_rgb, pixels_per_cm2)

    @classmethod
    def _run_analysis(cls, image_bytes: bytes, img_rgb: np.ndarray, pixels_per_cm2: float):
        st.markdown("### Segmentation overview")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original image**")
            st.image(img_rgb, use_container_width=True)

        # --- Roboflow segmentation ---
        mask = None
        rf_output = _run_roboflow_workflow(image_bytes)

        if rf_output is not None:
            mask = mask_from_roboflow_output(rf_output)

        if mask is None:
            st.warning(
                "Roboflow workflow unavailable or returned no predictions. "
                "Using simple color-based segmentation instead."
            )
            # fallback mask uses the decoded BGR image
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            mask = simple_color_mask(img_bgr)

        # Ensure mask matches displayed image size
        if mask.shape[:2] != img_rgb.shape[:2]:
            mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        with col2:
            st.markdown("**Binary leaf mask**")
            st.image(mask, use_container_width=True, clamp=True)

        # --- Measurements ---
        measurements = measure_leaves(mask, pixels_per_cm2)
        if not measurements:
            st.error("No leaf regions detected after segmentation.")
            return

        df = pd.DataFrame(
            [
                {
                    "Leaf ID": m.leaf_id,
                    "Area (cm²)": round(m.area_cm2, 2),
                    "Height (cm)": round(m.height_cm, 2),
                    "Width (cm)": round(m.width_cm, 2),
                    "Area / height (cm)": round(m.area_to_height, 2),
                }
                for m in measurements
            ]
        ).sort_values("Leaf ID")

        st.markdown("### Leaf measurements")
        st.dataframe(df, use_container_width=True)

        # Summary stats
        areas = np.array([m.area_cm2 for m in measurements])
        heights = np.array([m.height_cm for m in measurements])

        st.markdown("### Summary")
        st.write(f"- **Leaf count:** {len(measurements)}")
        st.write(
            f"- **Average leaf area:** {areas.mean():.2f} cm² "
            f"(std dev: {areas.std(ddof=1):.2f} cm²)"
        )
        st.write(
            f"- **Average leaf height:** {heights.mean():.2f} cm "
            f"(std dev: {heights.std(ddof=1):.2f} cm)"
        )

        st.caption(
            "Area is based on the segmented leaf region; height and width are from the "
            "bounding box in centimetres using the calibrated grid."
        )
