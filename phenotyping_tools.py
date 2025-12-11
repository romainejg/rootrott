# phenotyping_tools.py

import os
import math
import tempfile
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Try to import cv2 safely (Python 3.13 wheels can be finicky)
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover – we just want to fail gracefully
    cv2 = None  # type: ignore

# Try to import Roboflow inference SDK (optional)
try:
    from inference_sdk import InferenceHTTPClient  # type: ignore
except Exception:  # pragma: no cover
    InferenceHTTPClient = None  # type: ignore


# -----------------------------
# Types
# -----------------------------
BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------
# Grid scale helper
# -----------------------------

def _estimate_pixels_per_cm2_with_grid(
    image_bgr: np.ndarray,
) -> Tuple[Optional[float], Optional[BBox]]:
    """
    Estimate pixels per cm² from grid squares in the background.

    Returns:
      (pixels_per_cm2, sample_square_bbox)

    sample_square_bbox is (x, y, w, h) of one representative square,
    so we can draw it on the original image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_candidates: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter out tiny noise and huge blobs
        if area < 200 or area > 5000:
            continue
        if h == 0:
            continue
        aspect = w / float(h)
        # Roughly square
        if 0.8 < aspect < 1.2:
            square_candidates.append((x, y, w, h))

    if len(square_candidates) == 0:
        return None, None

    areas = [w * h for (_, _, w, h) in square_candidates]
    median_area = float(np.median(areas))

    # Choose the square whose area is closest to the median
    square_candidates.sort(key=lambda b: abs((b[2] * b[3]) - median_area))
    best_bbox = square_candidates[0]
    pixels_per_cm2 = float(best_bbox[2] * best_bbox[3])

    return pixels_per_cm2, best_bbox


# -----------------------------
# Fallback HSV segmentation
# -----------------------------

def _segment_leaves_hsv(image_bgr: np.ndarray) -> List[BBox]:
    """
    Simple color-based segmentation to approximate leaf regions.

    This is a backup when Roboflow is not available or fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Broad green-ish range (tune if needed)
    lower = np.array([30, 30, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter very small specks
        if w * h < 2000:
            continue
        bboxes.append((x, y, w, h))

    return bboxes


# -----------------------------
# Roboflow leaf detection
# -----------------------------

def _run_roboflow_detection(image_path: str) -> Optional[List[BBox]]:
    """
    Call your Roboflow workflow and convert predictions to bounding boxes.

    Returns list of (x, y, w, h) in pixel coordinates,
    or None if anything goes wrong (so the caller can fall back).
    """
    if InferenceHTTPClient is None:
        st.info("Roboflow SDK not installed. Falling back to color-based segmentation.")
        return None

    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        st.info("Roboflow API key not found in Streamlit secrets as 'ROBOFLOW_API_KEY'. "
                "Falling back to color-based segmentation.")
        return None

    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

        # NOTE: `images` must be a dict – path to the local file, not bytes.
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="find-leaves-3",
            images={"image": image_path},
        )
    except Exception as e:
        st.info("Roboflow workflow failed or is not configured correctly. "
                "Falling back to color-based segmentation.")
        st.caption(f"Details: {e}")
        return None

    # Try to find a list of predictions in the response
    preds = None
    if isinstance(result, dict):
        if "predictions" in result and isinstance(result["predictions"], list):
            preds = result["predictions"]
        elif "outputs" in result and isinstance(result["outputs"], list) and result["outputs"]:
            first = result["outputs"][0]
            if isinstance(first, dict) and "predictions" in first and isinstance(first["predictions"], list):
                preds = first["predictions"]

    if preds is None:
        st.info("Roboflow response did not contain a 'predictions' list. "
                "Falling back to color-based segmentation.")
        return None

    boxes: List[BBox] = []
    for p in preds:
        if not isinstance(p, dict):
            continue

        # Common detection format: center x,y plus width/height
        cx = p.get("x")
        cy = p.get("y")
        w = p.get("width")
        h = p.get("height")

        if all(v is not None for v in (cx, cy, w, h)):
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
            continue

        # Alternative nested bbox format
        bbox = p.get("bbox")
        if isinstance(bbox, dict):
            cx = bbox.get("x")
            cy = bbox.get("y")
            w = bbox.get("width")
            h = bbox.get("height")
            if all(v is not None for v in (cx, cy, w, h)):
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    if not boxes:
        st.info("Roboflow returned no leaf detections. "
                "Falling back to color-based segmentation.")
        return None

    return boxes


# -----------------------------
# Streamlit UI
# -----------------------------

class PhenotypingUI:
    """Leaf phenotyping UI: count leaves, measure area, length, and ratios."""

    @classmethod
    def render(cls):
        # Hard fail if OpenCV isn't usable in this environment
        if cv2 is None:
            st.error(
                "Phenotyping requires OpenCV (`opencv-python-headless`). "
                "It is not available in this environment."
            )
            st.info("Ask your admin to add `opencv-python-headless` to requirements.txt.")
            return

        st.subheader("Leaf phenotyping")

        st.markdown(
            """
            Upload a canopy image with a **1 cm² grid background**.

            Rootweiler will attempt to:
            - Count individual leaves (using your Roboflow model if configured)
            - Estimate leaf area (cm²) and bounding length/width (cm)
            - Calculate a simple area/length ratio per leaf
            - Summarize mean and spread of leaf sizes
            """
        )

        uploaded = st.file_uploader(
            "Upload a canopy image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        mode = st.radio(
            "Segmentation mode",
            [
                "Roboflow leaf model (recommended)",
                "Color-based segmentation (backup)",
            ],
            index=0,
            help="Roboflow mode requires an API key in Streamlit secrets as `ROBOFLOW_API_KEY`.",
        )

        # Read bytes once
        file_bytes = uploaded.read()
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode the uploaded image.")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Show original image
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("#### Original image")
            st.image(image_rgb, use_column_width=True)

        # Estimate grid scale and remember one example square
        pixels_per_cm2, square_bbox = _estimate_pixels_per_cm2_with_grid(image_bgr)
        if pixels_per_cm2 is None:
            st.warning(
                "Could not automatically detect the 1 cm² grid. "
                "Leaf areas will be reported in **pixels²**, not cm²."
            )
        else:
            st.success(
                f"Estimated grid: **~{pixels_per_cm2:.0f} pixels per cm²** "
                "(based on a detected grid square)."
            )

        if st.button("Run phenotyping", type="primary"):
            with st.spinner("Segmenting leaves and calculating metrics..."):
                boxes: Optional[List[BBox]] = None

                # Try Roboflow first if requested
                if mode.startswith("Roboflow"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    boxes = _run_roboflow_detection(tmp_path)

                # Fallback to HSV segmentation
                if boxes is None:
                    boxes = _segment_leaves_hsv(image_bgr)

            if not boxes:
                st.error("No leaves found. Try checking the image or switching segmentation mode.")
                return

            # Convert from pixel units to cm using grid estimate
            px_per_cm = math.sqrt(pixels_per_cm2) if pixels_per_cm2 and pixels_per_cm2 > 0 else None

            rows = []
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                area_px = float(w * h)

                if px_per_cm is not None:
                    area_cm2 = area_px / pixels_per_cm2
                    length_cm = h / px_per_cm    # along vertical axis in the image
                    width_cm = w / px_per_cm     # horizontal axis
                    # "Leaf to height" style metric: area per unit length (cm²/cm = cm)
                    area_per_length = area_cm2 / length_cm if length_cm > 0 else np.nan

                    rows.append(
                        {
                            "Leaf #": i,
                            "x_px": x,
                            "y_px": y,
                            "width_cm": round(width_cm, 2),
                            "length_cm": round(length_cm, 2),
                            "area_cm2": round(area_cm2, 2),
                            "area_per_length_cm": round(area_per_length, 2)
                            if not np.isnan(area_per_length)
                            else np.nan,
                        }
                    )
                else:
                    # Fallback: report everything in pixels
                    rows.append(
                        {
                            "Leaf #": i,
                            "x_px": x,
                            "y_px": y,
                            "width_px": w,
                            "length_px": h,
                            "area_px2": area_px,
                        }
                    )

            df = pd.DataFrame(rows)

            # Metrics + summary in right column
            with col_right:
                st.markdown("#### Leaf metrics")
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("#### Summary")
                st.write(f"- Leaf count: **{len(boxes)}**")

                if px_per_cm is not None:
                    st.write(
                        f"- Mean leaf area: **{df['area_cm2'].mean():.2f} cm²**, "
                        f"SD: **{df['area_cm2'].std():.2f} cm²**"
                    )
                    st.write(
                        f"- Mean leaf length: **{df['length_cm'].mean():.2f} cm**, "
                        f"mean width: **{df['width_cm'].mean():.2f} cm**"
                    )
                    st.caption(
                        "Area per length (cm) is effectively an average width of each leaf."
                    )
                else:
                    st.caption(
                        "No grid scale found, so sizes are in pixels – useful for relative comparisons only."
                    )

            # Build overlay: sample grid square + leaf boxes
            overlay = image_bgr.copy()

            # Draw sample 1 cm² square if we found one
            if square_bbox is not None:
                sx, sy, sw, sh = square_bbox
                cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                cv2.putText(
                    overlay,
                    "1 cm² sample",
                    (sx, max(sy - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw leaf boxes
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    str(i),
                    (x, max(y - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            st.markdown("#### Segmentation overlay")
            st.image(overlay_rgb, use_container_width=True)

            # CSV export
            if not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download leaf metrics as CSV",
                    data=csv,
                    file_name="rootweiler_leaf_metrics.csv",
                    mime="text/csv",
                )
