# phenotyping.py

import os
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None  # we will check at runtime

# -------------------------------
# Roboflow configuration
# -------------------------------

WORKSPACE_NAME = "rootweiler"
WORKFLOW_ID = "find-leaves-3"

# This MUST match the node name in your workflow that outputs the leaf predictions.
# After the first run, do st.write(result) to inspect the JSON,
# then set this string to the correct key.
LEAF_STEP_NAME = "leaf-segmentation"  # <-- adjust to your workflow node name


# -------------------------------
# Grid calibration (1 cm squares)
# -------------------------------

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def estimate_grid_pixels_per_cm2(image_bgr: np.ndarray) -> Tuple[Optional[float], Optional[BBox]]:
    """
    Estimate pixels per cm² from the grid board by finding many near-square
    contours and using their median area.

    Returns:
        (pixels_per_cm2, representative_square_bbox)
        - pixels_per_cm2: average pixel area of one cm² square
        - representative_square_bbox: (x, y, w, h) of one selected square for visualization
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # nearly square and not tiny
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    if len(squares) < 20:
        return None, None

    areas = [w * h for (_, _, w, h) in squares]
    median_area = float(np.median(areas))

    # pick squares closest to median area
    squares_sorted = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    top = squares_sorted[:20]
    avg_area = float(np.mean([w * h for (_, _, w, h) in top]))

    # representative square for overlay: the first of the closest group
    rep_square = squares_sorted[0]

    pixels_per_cm2 = avg_area  # one grid square ≈ 1 cm²
    return pixels_per_cm2, rep_square


# -------------------------------
# Roboflow helpers
# -------------------------------

def _get_roboflow_api_key() -> str:
    """Get Roboflow API key from env or Streamlit secrets."""
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        try:
            if "ROBOFLOW_API_KEY" in st.secrets:
                api_key = st.secrets["ROBOFLOW_API_KEY"]
        except Exception:
            pass

    return api_key


def run_roboflow_workflow(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Send an image to the Roboflow workflow and return the raw JSON result.
    """
    if InferenceHTTPClient is None:
        raise RuntimeError(
            "inference-sdk is not installed. "
            "Add `inference-sdk` to requirements.txt."
        )

    api_key = _get_roboflow_api_key()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set. "
            "Set it as an environment variable or in Streamlit secrets."
        )

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Roboflow HTTP client wants a file path
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image_bgr)

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": tmp_path},
            use_cache=True,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


def extract_leaf_masks_from_workflow(
    workflow_result: Dict[str, Any],
    image_shape: Tuple[int, int, int],
) -> List[np.ndarray]:
    """
    Convert Roboflow workflow output into a list of binary masks (one per leaf).

    You MUST adapt the indexing to match your workflow JSON structure.
    After running once, do:

        st.write(result)

    and look for a structure like:
        result["predictions"]["image"]["<some-step>"]["predictions"]

    Then set LEAF_STEP_NAME to that step name.
    """
    h, w = image_shape[:2]

    try:
        step_output = workflow_result["predictions"]["image"][LEAF_STEP_NAME]
        predictions = step_output["predictions"]
    except Exception as e:
        raise KeyError(
            "Could not find leaf predictions in Roboflow result. "
            "Open the JSON once with `st.write(result)` and update "
            "extract_leaf_masks_from_workflow() indexing and LEAF_STEP_NAME.\n"
            f"Error: {e}"
        )

    leaf_masks: List[np.ndarray] = []

    for pred in predictions:
        # For instance segmentation, Roboflow usually returns polygons in 'points'
        points = pred.get("points")
        if points:
            poly = np.array(points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            leaf_masks.append(mask)
        else:
            # If your model returns bitmasks instead, handle them here.
            # E.g. base64-encoded or run-length encoded masks.
            continue

    return leaf_masks


# -------------------------------
# Fallback segmentation (HSV)
# -------------------------------

def simple_hsv_leaf_masks(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Fallback segmentation using a simple green HSV threshold and connected components.
    This is a backup if Roboflow is unavailable.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels = cv2.connectedComponents(mask)
    leaf_masks: List[np.ndarray] = []
    for label in range(1, num_labels):
        leaf_mask = np.zeros_like(mask)
        leaf_mask[labels == label] = 255
        if cv2.countNonZero(leaf_mask) > 5000:  # ignore tiny specks
            leaf_masks.append(leaf_mask)

    return leaf_masks


# -------------------------------
# Measurement helpers
# -------------------------------

def measure_leaves(
    leaf_masks: List[np.ndarray],
    pixels_per_cm2: float,
) -> Tuple[List[Dict[str, float]], float, float]:
    """
    Given a list of per-leaf masks and a scale (px² / cm²),
    compute per-leaf metrics and summary stats.

    Returns:
        leaf_measurements, mean_area_cm2, sd_area_cm2
    """
    measurements: List[Dict[str, float]] = []

    if not leaf_masks or pixels_per_cm2 <= 0:
        return measurements, float("nan"), float("nan")

    px_per_cm = pixels_per_cm2 ** 0.5

    for idx, mask in enumerate(leaf_masks, start=1):
        area_px = int(cv2.countNonZero(mask))
        if area_px == 0:
            continue

        # bounding box for height/width
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        area_cm2 = area_px / pixels_per_cm2
        height_cm = h / px_per_cm
        width_cm = w / px_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else float("nan")

        measurements.append(
            {
                "Leaf ID": idx,
                "Area (cm²)": area_cm2,
                "Height (cm)": height_cm,
                "Width (cm)": width_cm,
                "Area : Height (cm)": ratio,
            }
        )

    if not measurements:
        return measurements, float("nan"), float("nan")

    areas = [m["Area (cm²)"] for m in measurements]
    mean_area = float(np.mean(areas))
    sd_area = float(np.std(areas))

    return measurements, mean_area, sd_area


def build_visualizations(
    image_bgr: np.ndarray,
    leaf_masks: List[np.ndarray],
    grid_square: Optional[BBox],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
      - original_with_grid: original image with a rectangle over the selected grid square
      - label_view: color overlay of leaf masks + boxes
      - binary_view: union of leaves as white on black
    """
    h, w = image_bgr.shape[:2]

    # Original with grid overlay
    orig = image_bgr.copy()
    if grid_square is not None:
        x, y, gw, gh = grid_square
        cv2.rectangle(orig, (x, y), (x + gw, y + gh), (0, 255, 255), 2)

    # Label view
    label_img = np.zeros_like(image_bgr)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (200, 128, 0),
        (0, 128, 200),
    ]

    union_mask = np.zeros((h, w), dtype=np.uint8)

    for i, mask in enumerate(leaf_masks):
        color = colors[i % len(colors)]
        label_img[mask > 0] = color
        union_mask = cv2.bitwise_or(union_mask, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(label_img, (x, y), (x + bw, y + bh), (255, 255, 255), 1)

    # Binary union as 3-channel
    binary_view = np.zeros_like(image_bgr)
    binary_view[union_mask > 0] = (255, 255, 255)

    # Convert BGR → RGB for Streamlit
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    label_rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    binary_rgb = cv2.cvtColor(binary_view, cv2.COLOR_BGR2RGB)

    return orig_rgb, label_rgb, binary_rgb


# -------------------------------
# Streamlit UI
# -------------------------------

class PhenotypingUI:
    """Rootweiler phenotyping module."""

    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a **grid-board lettuce image**.  
            Rootweiler will:

            - Auto-calibrate from the 1 cm squares  
            - Segment leaves (Roboflow workflow)  
            - Report per-leaf **area (cm²)**, **height (cm)**, **width (cm)**, and **area:height**  
            - Summarize **leaf count, mean leaf area, and area deviation**
            """
        )

        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read image with PIL → numpy (BGR for OpenCV)
        pil_img = Image.open(uploaded).convert("RGB")
        image_rgb = np.array(pil_img)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- 1) Grid calibration ---

        pixels_per_cm2, grid_square = estimate_grid_pixels_per_cm2(image_bgr)

        with st.expander("Grid calibration (1 cm squares)", expanded=True):
            if pixels_per_cm2 is None:
                st.error(
                    "Could not reliably detect the grid squares.\n"
                    "Check that the board is visible and well-lit."
                )
            else:
                px_per_cm = pixels_per_cm2 ** 0.5
                st.success(
                    f"Estimated grid spacing: **{px_per_cm:.2f} px per 1 cm** "
                    f"(~{pixels_per_cm2:.0f} px² per cm²)"
                )
                st.caption(
                    "The yellow rectangle over the original image shows one of the "
                    "squares used for calibration."
                )

        if pixels_per_cm2 is None or pixels_per_cm2 <= 0:
            st.stop()

        # --- 2) Leaf segmentation via Roboflow (with fallback) ---

        use_roboflow = st.checkbox("Use Roboflow leaf model (recommended)", value=True)

        if use_roboflow:
            try:
                rf_result = run_roboflow_workflow(image_bgr)
                leaf_masks = extract_leaf_masks_from_workflow(rf_result, image_bgr.shape)
            except Exception as e:
                st.error(
                    "Roboflow workflow failed or is not configured correctly. "
                    "Falling back to simple HSV segmentation.\n\n"
                    f"Details: {e}"
                )
                leaf_masks = simple_hsv_leaf_masks(image_bgr)
        else:
            leaf_masks = simple_hsv_leaf_masks(image_bgr)

        if not leaf_masks:
            st.error("No leaves were detected. Check image or model settings.")
            return

        # --- 3) Measurements ---

        measurements, mean_area, sd_area = measure_leaves(leaf_masks, pixels_per_cm2)

        # --- 4) Visualizations ---

        orig_vis, label_vis, binary_vis = build_visualizations(
            image_bgr, leaf_masks, grid_square
        )

        st.markdown("### Segmentation overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original image**")
            st.image(orig_vis, use_column_width=True)
        with c2:
            st.markdown("**Leaf labels**")
            st.image(label_vis, use_column_width=True)
        with c3:
            st.markdown("**Binary mask (all leaves)**")
            st.image(binary_vis, use_column_width=True)

        # --- 5) Measurement table ---

        st.markdown("### Leaf measurements")

        if measurements:
            df = pd.DataFrame(measurements)
            st.dataframe(df.style.format({"Area (cm²)": "{:.2f}", "Height (cm)": "{:.2f}", "Width (cm)": "{:.2f}", "Area : Height (cm)": "{:.2f}"}),
                         use_container_width=True)

            st.markdown("#### Summary")
            st.write(f"- **Leaf count:** {len(measurements)}")
            if not np.isnan(mean_area):
                st.write(f"- **Average leaf area:** {mean_area:.2f} cm²")
            if not np.isnan(sd_area):
                st.write(f"- **Leaf area standard deviation:** {sd_area:.2f} cm²")
        else:
            st.info("No valid leaf measurements were produced.")
