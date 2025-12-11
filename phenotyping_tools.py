# phenotyping_tools.py

import os
import io
import json
import tempfile
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from inference_sdk import InferenceHTTPClient
    _ROBOFLOW_AVAILABLE = True
except Exception:
    _ROBOFLOW_AVAILABLE = False


# -------------------------------
# Roboflow config – EDIT IF NEEDED
# -------------------------------
ROBOFLOW_WORKSPACE = "rootweiler"
ROBOFLOW_WORKFLOW_ID = "find-leaves-3"

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# =========================================================
# 1. Grid calibration – reuse your “squares” logic
# =========================================================

def calculate_pixels_per_cm2_and_squares(
    image_bgr: np.ndarray,
) -> Tuple[Optional[float], List[BBox]]:
    """
    Estimate pixels per cm² using square-like objects in the image,
    following the same logic from your old code:

      - find contours on Canny edges
      - keep near-square bounding boxes (w ≈ h)
      - require area > 1000 px
      - if not enough squares, return None

    Returns:
      pixels_per_cm2 (float | None),
      list of chosen square bounding boxes (x, y, w, h) for overlay.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        # near-square
        if abs(w - h) < 10 and w * h > 1000:
            squares.append((x, y, w, h))

    # Need enough squares to be confident
    if len(squares) < 20:
        return None, []

    areas = [w * h for (_, _, w, h) in squares]
    median_area = float(np.median(areas))

    # Choose squares closest to the median area
    squares_sorted = sorted(
        squares,
        key=lambda s: abs((s[2] * s[3]) - median_area),
    )
    selected = squares_sorted[:20]

    average_area = float(np.mean([w * h for (_, _, w, h) in selected]))
    pixels_per_cm2 = average_area

    return pixels_per_cm2, selected


# =========================================================
# 2. Fallback color-based leaf segmentation
# =========================================================

def segment_leaves_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Simple HSV range for lettuce leaves as fallback.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Broad green band – may be tuned later.
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def extract_leaf_contours(mask: np.ndarray, min_area_px: int = 300) -> List[np.ndarray]:
    """
    Find leaf blobs in a binary mask and filter tiny specks.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    large = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    return large


# =========================================================
# 3. Roboflow integration
# =========================================================

def _call_roboflow_workflow(
    image_bytes: bytes,
    filename: str,
    api_key: str,
) -> Any:
    """
    Call Roboflow workflow via the official SDK, sending a temp file path.
    We return whatever the SDK gives us (list/dict), and parse it separately.
    """
    if not _ROBOFLOW_AVAILABLE:
        raise RuntimeError("inference_sdk not installed in this environment.")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Use a temp file; Roboflow docs show passing a path string.
    suffix = os.path.splitext(filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": tmp_path},
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return result


def _find_prediction_lists(obj: Any) -> List[list]:
    """
    Recursively search an arbitrary nested structure for keys named 'predictions'
    that hold a list. This is robust to different workflow node layouts.
    """
    found: List[list] = []

    def _search(o: Any):
        if isinstance(o, dict):
            if "predictions" in o and isinstance(o["predictions"], list):
                found.append(o["predictions"])
            for v in o.values():
                _search(v)
        elif isinstance(o, list):
            for item in o:
                _search(item)

    _search(obj)
    return found


def parse_roboflow_leaf_polygons(
    rf_result: Any,
    image_shape: Tuple[int, int, int],
) -> Optional[List[np.ndarray]]:
    """
    Try to turn Roboflow workflow output into a list of leaf instance polygons.

    We look through the structure for 'predictions' lists containing objects
    with a 'points' key (list of [x, y]).

    Returns:
      list of contours (opencv-style Nx1x2 int32 arrays), or None if nothing usable.
    """
    h, w, _ = image_shape

    preds_lists = _find_prediction_lists(rf_result)
    if not preds_lists:
        return None

    polygons: List[np.ndarray] = []

    for preds in preds_lists:
        for p in preds:
            # Typical instance segmentation output: polygon in 'points'
            pts = p.get("points") or p.get("polygon") or p.get("segmentation")
            if pts and isinstance(pts, list):
                try:
                    arr = np.array(pts, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        # clamp to image bounds
                        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
                        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
                        arr_int = arr.astype(np.int32)
                        polygons.append(arr_int.reshape(-1, 1, 2))
                except Exception:
                    continue

    if not polygons:
        return None

    return polygons


def segment_leaves_with_roboflow(
    image_bgr: np.ndarray,
    image_bytes: bytes,
    filename: str,
    api_key: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Full Roboflow pipeline:
      - Call workflow
      - Parse polygons
      - Build a binary mask + list of contours

    Raises if anything goes wrong so caller can fall back.
    """
    rf_result = _call_roboflow_workflow(image_bytes=image_bytes, filename=filename, api_key=api_key)
    polygons = parse_roboflow_leaf_polygons(rf_result, image_bgr.shape)

    if not polygons:
        raise RuntimeError("No usable leaf polygons found in Roboflow output.")

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)

    # Optional: clean up mask a bit
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contours are just these polygons (we could re-find, but polygons are fine)
    contours = polygons

    return mask, contours


# =========================================================
# 4. Streamlit UI wrapper
# =========================================================

class PhenotypingUI:
    """
    Leaf phenotyping for lettuce:
      - Estimates scale using 1 cm² grid
      - Calls Roboflow instance segmentation when available
      - Falls back to HSV segmentation if needed
      - Computes leaf count + per-leaf area in cm²
    """

    @classmethod
    def render(cls):
        st.subheader("Phenotyping (leaf area & count)")

        st.markdown(
            """
            Upload a canopy image. Rootweiler will:

            - Estimate physical scale using the 1 cm² background grid  
            - Run a **Roboflow instance segmentation** workflow (if configured)  
            - Fall back to a simple color-based mask if the workflow fails  
            - Count leaves and estimate individual leaf areas (cm²)
            """
        )

        uploaded = st.file_uploader(
            "Upload lettuce canopy image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # Load as PIL + OpenCV BGR
        try:
            pil_image = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.markdown("#### Original image")
        st.image(pil_image, use_column_width=True)

        # ---- Grid calibration ----
        pixels_per_cm2, grid_squares = calculate_pixels_per_cm2_and_squares(image_bgr)

        col_cal, col_cal_img = st.columns([1, 2])

        with col_cal:
            st.markdown("##### Grid calibration")
            if pixels_per_cm2 is None:
                st.warning(
                    "Could not reliably detect enough grid squares. "
                    "Areas will be reported only in pixels."
                )
            else:
                st.success(
                    f"Estimated ~**{pixels_per_cm2:.1f} pixels per cm²** "
                    "(using background squares)."
                )

        with col_cal_img:
            overlay = image_bgr.copy()
            for (x, y, w, h) in grid_squares:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption="Grid squares used for calibration",
                use_column_width=True,
            )

        api_key = st.secrets.get("ROBOFLOW_API_KEY") if hasattr(st, "secrets") else None

        if _ROBOFLOW_AVAILABLE and api_key:
            st.caption(
                f"Roboflow workflow: `{ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW_ID}`"
            )
        elif not _ROBOFLOW_AVAILABLE:
            st.caption("Roboflow SDK not installed; using color-based segmentation only.")
        else:
            st.caption("No `ROBOFLOW_API_KEY` in secrets; using color-based segmentation only.")

        if not st.button("Run phenotyping", type="primary"):
            return

        # ---- Segmentation pipeline ----
        mask: np.ndarray
        contours: List[np.ndarray]
        method_label = ""

        # Try Roboflow first
        if _ROBOFLOW_AVAILABLE and api_key:
            try:
                mask, contours = segment_leaves_with_roboflow(
                    image_bgr=image_bgr,
                    image_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    api_key=api_key,
                )
                method_label = "Roboflow instance segmentation"
            except Exception:
                # Fall back to HSV
                mask = segment_leaves_hsv(image_bgr)
                contours = extract_leaf_contours(mask)
                method_label = "Color-based fallback"
        else:
            mask = segment_leaves_hsv(image_bgr)
            contours = extract_leaf_contours(mask)
            method_label = "Color-based fallback"

        if len(contours) == 0:
            st.warning(
                "No leaf objects detected with the current segmentation method."
            )
            return

        # Visualization
        vis = image_bgr.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            st.markdown(f"#### Detected leaves ({method_label})")
            st.image(
                vis_rgb,
                caption=f"Leaf contours (count: {len(contours)})",
                use_column_width=True,
            )
        with col_vis2:
            st.markdown("#### Segmentation mask")
            st.image(
                mask,
                caption="Binary mask used for analysis",
                use_column_width=True,
            )

        # ---- Leaf metrics ----
        leaf_areas_px = [cv2.contourArea(c) for c in contours]

        if pixels_per_cm2:
            leaf_areas_cm2 = [a / pixels_per_cm2 for a in leaf_areas_px]
            total_area_cm2 = float(np.sum(leaf_areas_cm2))
            mean_leaf_cm2 = float(np.mean(leaf_areas_cm2))
            std_leaf_cm2 = float(np.std(leaf_areas_cm2))
        else:
            leaf_areas_cm2 = None
            total_area_cm2 = None
            mean_leaf_cm2 = None
            std_leaf_cm2 = None

        st.markdown("#### Leaf metrics")
        st.write(f"- Number of detected leaves: **{len(contours)}**")

        if pixels_per_cm2 and leaf_areas_cm2 is not None:
            st.write(f"- Total leaf area (approx): **{total_area_cm2:.1f} cm²**")
            st.write(f"- Average leaf area: **{mean_leaf_cm2:.2f} cm²**")
            st.write(f"- Leaf area standard deviation: **{std_leaf_cm2:.2f} cm²**")
        else:
            st.write("- Areas are only available in pixels (no reliable grid calibration).")

        # Small table of per-leaf areas
        with st.expander("Per-leaf areas", expanded=False):
            import pandas as pd

            if pixels_per_cm2 and leaf_areas_cm2 is not None:
                df = pd.DataFrame(
                    {
                        "Leaf #": np.arange(1, len(contours) + 1),
                        "Area (px)": np.round(leaf_areas_px, 1),
                        "Area (cm²)": np.round(leaf_areas_cm2, 3),
                    }
                )
            else:
                df = pd.DataFrame(
                    {
                        "Leaf #": np.arange(1, len(contours) + 1),
                        "Area (px)": np.round(leaf_areas_px, 1),
                    }
                )

            st.dataframe(df, use_container_width=True)
