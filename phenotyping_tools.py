# phenotyping_tools.py

import base64
import io
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

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
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # roughly square and not tiny
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    if len(squares) < 20:
        return None, []

    # Use median area to filter consistent squares
    areas = [w * h for (_, _, w, h) in squares]
    median_area = np.median(areas)

    # Keep squares whose area is closest to median
    squares_sorted = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
    # Use up to 20 best squares
    chosen = squares_sorted[:20]

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
# Roboflow workflow helpers
# ---------------------------------------------------------------------

# ⚠️ Make sure these match the "Deploy Workflow" snippet in Roboflow
WORKSPACE_NAME = "rootweiler"  # your workspace slug
WORKFLOW_ID = "leafy"          # your workflow slug
API_URL = "https://detect.roboflow.com"  # or "https://serverless.roboflow.com" if your snippet says so


def _get_roboflow_api_key() -> Optional[str]:
    """Fetch Roboflow API key from Streamlit secrets."""
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    if "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        return st.secrets["roboflow"]["api_key"]
    return None


def _call_roboflow_workflow_raw(image_bytes: bytes) -> Optional[Any]:
    """
    Lowest-level Roboflow call.

    Uses ONLY `run_workflow(workspace_name=..., workflow_id=..., images=...)`.
    Requires a modern `inference-sdk` that supports workflows.
    """
    api_key = _get_roboflow_api_key()
    if not api_key:
        st.info("Roboflow API key not found in secrets – using color-based segmentation instead.")
        return None

    try:
        client = InferenceHTTPClient(
            api_url=API_URL,
            api_key=api_key,
        )
    except Exception as e:
        st.info(f"Could not initialize Roboflow client ({type(e).__name__}: {e}). Falling back to color segmentation.")
        return None

    # Save bytes to a temp file (SDK is happy with a file path here)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    try:
        # Modern, documented workflow call
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": tmp_path},
            # NOTE: not passing use_cache here to be extra compatible
        )
    except AttributeError as e:
        # This means your inference-sdk is too old and does NOT support workflows via run_workflow
        st.error(
            "Your installed `inference-sdk` does not support `run_workflow`. "
            "Please upgrade with:\n\n"
            "`pip install --upgrade inference-sdk`\n\n"
            f"Details: {type(e).__name__}: {e}"
        )
        return None
    except Exception as e:
        st.info(
            f"Roboflow workflow call failed ({type(e).__name__}: {e}). "
            "Falling back to color-based segmentation."
        )
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return result


def _decode_base64_image(b64_val: str) -> bytes:
    """
    Handles both raw base64 and data URLs (e.g. 'data:image/png;base64,...').
    Returns raw image bytes.
    """
    if b64_val.startswith("data:image"):
        b64_val = b64_val.split(",", 1)[-1]
    return base64.b64decode(b64_val)


def _first_item(result: Any) -> Dict:
    """Normalize workflow result to a single dict."""
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    if isinstance(result, dict):
        return result
    return {}


def _run_roboflow_workflow(image_bytes: bytes) -> Optional[Dict[str, object]]:
    """
    Higher-level wrapper:

    Returns dict:
        {
          "predictions": <list>,
          "mask_image":  <np.ndarray> or None
        }
    or None on failure.
    """
    result = _call_roboflow_workflow_raw(image_bytes)
    if result is None:
        return None

    item = _first_item(result)

    # -------------------------------
    # 1) Predictions (your 'output2')
    # -------------------------------
    preds = None
    if "output2" in item:
        preds = item["output2"]
    elif "model_predictions" in item and isinstance(item["model_predictions"], dict):
        preds = item["model_predictions"].get("predictions") or item["model_predictions"].get("value")
    elif "predictions" in item:
        preds = item["predictions"]

    if not isinstance(preds, list) or len(preds) == 0:
        st.info("Roboflow returned no predictions – using color-based segmentation.")
        return None

    # -----------------------------------------
    # 2) Mask visualization (your 'output')
    # -----------------------------------------
    mask_image = None
    if "output" in item:
        viz = item["output"]
        # Depending on the field type, this may be a plain base64 string or a dict
        if isinstance(viz, dict):
            viz = viz.get("value") or viz.get("base64") or viz.get("data")
        if isinstance(viz, str):
            try:
                img_bytes = _decode_base64_image(viz)
                mask_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                mask_image = None

    return {"predictions": preds, "mask_image": mask_image}


def _mask_from_roboflow_predictions(
    image_shape: Tuple[int, int, int], predictions: List[dict]
) -> np.ndarray:
    """
    Build a binary mask from instance-segmentation polygons returned
    by Roboflow. All leaves -> 255 in a single-channel mask.
    """
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        # Roboflow instance segmentation usually provides "points": [{"x": ..., "y": ...}, ...]
        pts = pred.get("points") or pred.get("polygon") or pred.get("segmentation")
        if not pts or not isinstance(pts, list):
            continue

        poly = []
        for p in pts:
            # Be defensive about the key names
            x = p.get("x") or p.get("X") or p.get("cx")
            y = p.get("y") or p.get("Y") or p.get("cy")
            if x is None or y is None:
                continue
            poly.append([int(round(x)), int(round(y))])

        if len(poly) < 3:
            continue

        poly_np = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [poly_np], 255)

    return mask


# ---------------------------------------------------------------------
# Fallback color-based segmentation (simple HSV threshold)
# ---------------------------------------------------------------------
def _color_based_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Simple green-ish segmentation as a fallback when Roboflow is not available."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

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

    num_labels, labels = cv2.connectedComponents(mask)
    measurements: List[LeafMeasurement] = []

    for label_id in range(1, num_labels):
        component = (labels == label_id)
        area_px = int(np.count_nonzero(component))
        if area_px < 500:  # ignore tiny specks
            continue

        ys, xs = np.where(component)
        if len(ys) == 0:
            continue

        height_px = ys.max() - ys.min() + 1

        area_cm2 = area_px / pixels_per_cm2
        height_cm = height_px / pixels_per_cm
        ratio = area_cm2 / height_cm if height_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                id=len(measurements) + 1,
                area_cm2=area_cm2,
                height_cm=height_cm,
                area_height_ratio=ratio,
            )
        )

    return measurements


# ---------------------------------------------------------------------
# Debug / test UI for Roboflow inside PhenotypingUI
# ---------------------------------------------------------------------
def _render_roboflow_debug_panel(image_bytes: bytes):
    """
    Small debug panel that:
      - Calls the workflow directly on the uploaded image
      - Shows raw JSON
      - Tries to decode any base64 image outputs
      - Shows first 5 predictions if available
    """
    with st.expander("Debug: inspect Roboflow workflow output", expanded=False):
        st.caption(
            f"Workspace: `{WORKSPACE_NAME}` · Workflow: `{WORKFLOW_ID}` · API URL: `{API_URL}`"
        )

        if st.button("Run Roboflow test on this image", key="rf_debug_button"):
            result = _call_roboflow_workflow_raw(image_bytes)
            if result is None:
                st.error("Roboflow call failed – see messages above for details.")
                return

            st.subheader("Raw workflow result")
            st.json(result)

            item = _first_item(result)

            # Try to detect base64 image-like fields
            st.subheader("Decoded image outputs (best guess)")
            image_fields: List[str] = []
            for key, value in item.items():
                if isinstance(value, str) and ("data:image" in value or len(value) > 100):
                    image_fields.append(key)

            if not image_fields:
                st.info(
                    "No obvious base64 image fields found in the result. "
                    "Check your workflow Outputs configuration."
                )
            else:
                for field_name in image_fields:
                    try:
                        img_bytes = _decode_base64_image(item[field_name])
                        st.markdown(f"**Output field:** `{field_name}`")
                        st.image(img_bytes, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Field `{field_name}` looked like base64, "
                            f"but could not be decoded: {type(e).__name__}: {e}"
                        )

            # Predictions preview
            st.subheader("Predictions preview")
            preds = None
            if "output2" in item:
                preds = item["output2"]
            elif "predictions" in item:
                preds = item["predictions"]
            elif "model_predictions" in item and isinstance(item["model_predictions"], dict):
                preds = item["model_predictions"].get("predictions")

            if isinstance(preds, list) and preds:
                st.write(f"Found {len(preds)} predictions. Showing first 5:")
                st.json(preds[:5])
            else:
                st.info(
                    "No prediction list found in `output2` / `predictions` / `model_predictions`. "
                    "Ensure your workflow exposes `$steps.model.predictions` under one of these names."
                )


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
            Rootweiler will:

            - Detect the grid and convert pixels to **cm²**
            - Segment each leaf (via your Roboflow `leafy` workflow if available)
            - Measure leaf **area**, **height**, and **area : height** ratio
            - Compute average leaf size and variability
            """
        )

        uploaded = st.file_uploader(
            "Upload phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Read image
        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # --- Optional debug / test panel for Roboflow ---
        _render_roboflow_debug_panel(image_bytes)

        # --- Grid calibration ---
        pixels_per_cm2, grid_boxes = _detect_grid_squares(img_bgr)
        if pixels_per_cm2 is None:
            st.error(
                "Could not detect enough 1 cm squares on the board. "
                "Check that the grid is in view and in focus."
            )
            return

        pixels_per_cm = float(np.sqrt(pixels_per_cm2))

        st.markdown(
            f"**Grid calibration:** ~{pixels_per_cm:.1f} pixels per cm "
            f"→ ~{pixels_per_cm2:.0f} pixels per cm²"
        )

        # Show overlay of squares used for calibration
        with st.expander("Show grid squares used for calibration", expanded=False):
            overlay = _overlay_grid_boxes(img_bgr, grid_boxes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Detected 1 cm² grid squares", use_container_width=True)

        # --- Segmentation: Roboflow first, fallback to color ---
        rf_result = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.caption("Leaf segmentation: Roboflow instance segmentation workflow (`leafy`).")
        else:
            mask = _color_based_mask(img_bgr)
            st.caption("Leaf segmentation: simple color-based fallback.")

        # Ensure binary mask
        mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)

        # --- Measurements ---
        try:
            leaf_measurements = _measure_leaves(mask_bin, pixels_per_cm2)
        except ValueError as e:
            st.error(str(e))
            return

        if not leaf_measurements:
            st.error("No leaves detected in the mask. Check the image / segmentation.")
            return

        # --- Visuals: original, mask ---
        st.markdown("### Segmentation overview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original image", use_container_width=True)
        with c2:
            st.image(mask_bin, caption="Binary leaf mask", use_container_width=True)

        # --- Table + summary ---
        st.markdown("### Leaf measurements")

        df = pd.DataFrame(
            [
                {
                    "Leaf ID": m.id,
                    "Area (cm²)": round(m.area_cm2, 2),
                    "Height (cm)": round(m.height_cm, 2),
                    "Area : height (cm)": round(m.area_height_ratio, 3),
                }
                for m in leaf_measurements
            ]
        )

        st.dataframe(df, use_container_width=True)

        areas = np.array([m.area_cm2 for m in leaf_measurements])
        mean_area = float(areas.mean())
        std_area = float(areas.std(ddof=1)) if len(areas) > 1 else 0.0

        st.markdown("#### Summary")
        st.write(f"- Leaves counted: **{len(leaf_measurements)}**")
        st.write(f"- Average leaf area: **{mean_area:.2f} cm²**")
        st.write(f"- Leaf area standard deviation: **{std_area:.2f} cm²**")

        st.caption(
            "Area is computed from the segmented leaf pixels and the detected 1 cm grid squares. "
            "Height is the vertical extent of each connected leaf component."
        )
