# phenotyping_tools.py
#
# End-to-end phenotyping:
# - Upload phenotyping image on 1 cm grid board
# - Call Roboflow "leafy" workflow via HTTP
# - Build leaf mask from polygon predictions (if available)
# - Detect grid squares and calibrate pixels -> cm
# - Measure leaf size leaf area, height, width, height/width
# - Display images + table in Streamlit
#
# Keeps the same PhenotypingUI.render() so app.py does not need to change.

import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
@dataclass
class LeafMeasurement:
    id: int
    area_cm2: float
    height_cm: float
    width_cm: float
    height_width_ratio: float


# ---------------------------------------------------------------------
# Grid calibration – detect 1 cm² squares
# ---------------------------------------------------------------------
def _detect_grid_squares(
    image_bgr: np.ndarray,
) -> Tuple[Optional[float], List[Tuple[int, int, int, int]]]:
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
    median_area = float(np.median(areas))

    # Keep squares whose area is closest to median
    squares_sorted = sorted(
        squares, key=lambda s: abs((s[2] * s[3]) - median_area)
    )
    chosen = squares_sorted[:20]

    avg_area = float(np.mean([w * h for (_, _, w, h) in chosen]))
    pixels_per_cm2 = avg_area

    return pixels_per_cm2, chosen


def _overlay_grid_boxes(
    image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Return a copy of the image with green rectangles drawn on detected grid squares."""
    out = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------
# Roboflow workflow – HTTP call
# ---------------------------------------------------------------------
def _get_api_key() -> Optional[str]:
    """
    Get Roboflow API key from Streamlit secrets.

    Supports either:
    - ROBOFLOW_API_KEY = "..."
    or
    - [roboflow]
      api_key = "..."
    in .streamlit/secrets.toml
    """
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    if "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        return st.secrets["roboflow"]["api_key"]
    return None


def _call_workflow_http(
    api_key: str, workspace_name: str, workflow_id: str, image_bytes: bytes
) -> Dict[str, Any]:
    """
    Call Roboflow workflow via the HTTP REST API directly.

    Endpoint:
        POST https://detect.roboflow.com/infer/workflows/<workspace>/<workflow-id>

    Body:
        {
          "api_key": "...",
          "inputs": {
            "image": {
              "type": "base64",
              "value": "<base64-encoded-image>"
            }
          }
        }
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://detect.roboflow.com/infer/workflows/{workspace_name}/{workflow_id}"

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64,
            }
        },
    }

    resp = requests.post(url, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        msg = f"HTTP {resp.status_code}"
        try:
            j = resp.json()
            err_text = j.get("error") or j.get("message")
            if err_text:
                msg += f" – {err_text}"
        except Exception:
            pass
        raise RuntimeError(f"Roboflow HTTP error: {msg}") from e

    try:
        result = resp.json()
    except Exception as e:
        raise RuntimeError(f"Could not decode JSON from Roboflow: {e}") from e

    return result


# ---------------------------------------------------------------------
# Extract polygon predictions from workflow JSON
# ---------------------------------------------------------------------
def _find_polygon_predictions(obj: Any) -> Optional[List[dict]]:
    """
    Recursively search the workflow result for a list of prediction dicts
    that contain "points" (Roboflow's polygon format).

    This makes us robust to different output naming (output1/output2/etc).
    """

    # If it's already a list, check if it looks like predictions
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) and "points" in item for item in obj):
            return obj
        for item in obj:
            res = _find_polygon_predictions(item)
            if res is not None:
                return res

    # If it's a dict, recurse on values
    if isinstance(obj, dict):
        for v in obj.values():
            res = _find_polygon_predictions(v)
            if res is not None:
                return res

    return None


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
        poly = np.array(
            [[p["x"], p["y"]] for p in pts if "x" in p and "y" in p],
            dtype=np.int32,
        )
        if poly.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [poly], 255)

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
        component = labels == label_id
        area_px = int(np.count_nonzero(component))
        if area_px < 500:  # ignore tiny specks
            continue

        ys, xs = np.where(component)
        if len(ys) == 0:
            continue

        height_px = ys.max() - ys.min() + 1
        width_px = xs.max() - xs.min() + 1

        area_cm2 = area_px / pixels_per_cm2
        height_cm = height_px / pixels_per_cm
        width_cm = width_px / pixels_per_cm
        ratio_hw = height_cm / width_cm if width_cm > 0 else 0.0

        measurements.append(
            LeafMeasurement(
                id=len(measurements) + 1,
                area_cm2=area_cm2,
                height_cm=height_cm,
                width_cm=width_cm,
                height_width_ratio=ratio_hw,
            )
        )

    return measurements


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
class PhenotypingUI:
    """Leaf phenotyping tool using Roboflow workflow + grid calibration."""
    
    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a **phenotyping photo** taken on the 1&nbsp;cm grid board.  
            Rootweiler will:

            - Detect the grid and convert pixels to **cm²**
            - Segment each leaf (via your Roboflow **leafy** workflow if available)
            - Measure **Size Leaf Area**, **height**, **width**, and **height : width** ratio
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

        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

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

        with st.expander("Show grid squares used for calibration", expanded=False):
            overlay = _overlay_grid_boxes(img_bgr, grid_boxes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption="Detected 1 cm² grid squares",
                use_container_width=True,
            )

        # --- Segmentation: Roboflow first, fallback to color ---
        api_key = _get_api_key()
        mask_from_rf: Optional[np.ndarray] = None

        if api_key:
            try:
                rf_result = _call_workflow_http(
                    api_key=api_key,
                    workspace_name="rootweiler",
                    workflow_id="leafy",
                    image_bytes=image_bytes,
                )
                preds = _find_polygon_predictions(rf_result)
                if preds:
                    mask_from_rf = _mask_from_roboflow_predictions(
                        img_rgb.shape, preds
                    )
                    st.caption(
                        "Leaf segmentation: Roboflow **leafy** workflow "
                        "(polygon instance segmentation)."
                    )
                else:
                    st.info(
                        "Roboflow call succeeded but no polygon predictions were "
                        "found in the output. Falling back to color-based mask."
                    )
            except Exception as e:
                st.info(
                    f"Roboflow workflow call failed ({type(e).__name__}: {e}). "
                    "Using color-based segmentation instead."
                )

        if mask_from_rf is not None:
            mask = mask_from_rf
        else:
            mask = _color_based_mask(img_bgr)
            st.caption("Leaf segmentation: simple color-based fallback.")

        mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)

        # --- Visuals: original, mask ---
        st.markdown("### Segmentation overview")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original image", use_container_width=True)
        with c2:
            st.image(mask_bin, caption="Binary leaf mask", use_container_width=True)

        # --- Measurements ---
        try:
            leaf_measurements = _measure_leaves(mask_bin, pixels_per_cm2)
        except ValueError as e:
            st.error(str(e))
            return

        if not leaf_measurements:
            st.error(
                "No leaves detected in the mask. "
                "Check the image / segmentation settings."
            )
            return

        # --- Table + summary ---
        st.markdown("### Leaf measurements")

        df = pd.DataFrame(
            [
                {
                    "Leaf": m.id,
                    "Size Leaf Area (cm²)": round(m.area_cm2, 2),
                    "Height (cm)": round(m.height_cm, 2),
                    "Width (cm)": round(m.width_cm, 2),
                    "Height / Width": round(m.height_width_ratio, 2),
                }
                for m in leaf_measurements
            ]
        )

        st.dataframe(df, use_container_width=True)

        # Summary stats
        st.markdown("#### Summary")
        st.write(
            f"- Number of leaves: **{len(leaf_measurements)}**\n"
            f"- Mean Size Leaf Area: **{df['Size Leaf Area (cm²)'].mean():.2f} cm²**\n"
            f"- Mean height: **{df['Height (cm)'].mean():.2f} cm**\n"
            f"- Mean width: **{df['Width (cm)'].mean():.2f} cm**\n"
        )

        # Download table
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Leaf measurements")

        st.download_button(
            label="Download table (Excel)",
            data=buf.getvalue(),
            file_name="leaf_measurements.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )
