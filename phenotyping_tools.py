# phenotyping_tools.py

import os
import io
import json
import traceback
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Roboflow SDK
try:
    from inference_sdk import InferenceHTTPClient
    _ROBOFLOW_AVAILABLE = True
except Exception:
    _ROBOFLOW_AVAILABLE = False

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

# -------------------------------
# Roboflow config – EDIT IF NEEDED
# -------------------------------
ROBOFLOW_WORKSPACE = "rootweiler"
ROBOFLOW_WORKFLOW_ID = "find-leaves-3"


# -------------------------------
# Utility: patch decode_workflow_outputs
# -------------------------------

def _patch_roboflow_decoder_for_lists() -> None:
    """
    Monkey-patch inference_sdk's decode_workflow_outputs so that if it
    receives a list (or anything unexpected), it simply returns it instead
    of crashing with `'list' object has no attribute 'items'`.

    This does NOT change the HTTP call, only how the result is decoded.
    """
    if not _ROBOFLOW_AVAILABLE:
        return

    try:
        import importlib

        pp = importlib.import_module("inference_sdk.http.utils.post_processing")

        orig = getattr(pp, "decode_workflow_outputs", None)
        if orig is None:
            return

        # If we already patched, don't patch twice
        if getattr(pp, "_rw_patched_decode", False):
            return

        def safe_decode(workflow_outputs: Any, *args, **kwargs):
            try:
                return orig(workflow_outputs, *args, **kwargs)
            except Exception:
                # Just return the raw value so we can inspect it
                return workflow_outputs

        pp.decode_workflow_outputs = safe_decode
        pp._rw_patched_decode = True
    except Exception:
        # If patching fails, we silently continue; worst case the original
        # bug appears and we still show the traceback from our wrapper.
        pass


# -------------------------------
# Scale estimation via background grid
# -------------------------------

def estimate_pixels_per_cm2_and_squares(
    image_bgr: np.ndarray,
    min_area: int = 500,
    max_area: int = 5000,
    square_tolerance: int = 6,
    max_squares: int = 30,
) -> Tuple[Optional[float], List[BBox]]:
    """
    Try to find the calibration squares in the background grid.

    Returns:
      - pixels_per_cm2 (float | None)
      - list of selected square bounding boxes for overlay

    Strategy:
      - Canny edges → contours
      - keep contours with near-square bounding boxes (w ≈ h)
      - filter by reasonable area
      - use the median area as the 1 cm² estimate
      - pick a bunch of squares closest to that median for overlay
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        # approximate squares
        if abs(w - h) <= square_tolerance:
            area = w * h
            if min_area <= area <= max_area:
                candidates.append((x, y, w, h))

    if len(candidates) == 0:
        return None, []

    # Use areas to estimate 1 cm² in pixels
    areas = [w * h for (_, _, w, h) in candidates]
    median_area = float(np.median(areas))

    # Pick up to max_squares closest to the median area for overlay
    candidates_sorted = sorted(
        candidates, key=lambda b: abs((b[2] * b[3]) - median_area)
    )
    selected = candidates_sorted[: max_squares]

    pixels_per_cm2 = median_area
    return pixels_per_cm2, selected


# -------------------------------
# Simple color-based leaf segmentation
# -------------------------------

def segment_leaves_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Very simple HSV-based leaf segmentation.
    This is our fallback when Roboflow is unavailable or fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Broad green-ish range – may need tuning per dataset
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def extract_leaf_contours(mask: np.ndarray, min_area_px: int = 300) -> List[np.ndarray]:
    """
    Find connected leaf components in the mask and filter out tiny specks.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    return large_contours


# -------------------------------
# Roboflow workflow call
# -------------------------------

def _call_roboflow_workflow(
    image_bytes: bytes,
    filename: str,
    api_key: str,
) -> Any:
    """
    Call the Roboflow workflow using the official SDK,
    but with a safer decoder and full debug support.

    Returns:
      Whatever the SDK returns (list/dict). We do not force a structure.
    """
    if not _ROBOFLOW_AVAILABLE:
        raise RuntimeError("inference_sdk not installed in this environment.")

    # Patch decoder so decode_workflow_outputs doesn't crash on lists
    _patch_roboflow_decoder_for_lists()

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Roboflow's docs show using a file path string,
    # so we write the uploaded bytes to a temp file.
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
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return result


# -------------------------------
# Phenotyping UI
# -------------------------------

class PhenotypingUI:
    """
    Phenotyping / leaf analysis tool.

    Currently:
      - Always computes a simple HSV-based segmentation for leaf masks.
      - If Roboflow is configured, attempts to call the workflow and
        exposes the raw JSON + a download button for debugging.

    Once we understand the JSON structure, we can plug Roboflow's masks
    or polygons directly into the analysis.
    """

    @classmethod
    def render(cls):
        st.subheader("Phenotyping (leaf segmentation, beta)")

        st.markdown(
            """
            Upload a canopy image. Rootweiler will:

            - Try to call your **Roboflow workflow** (find-leaves-3) if configured  
            - Show the **raw JSON** result and offer a download for debugging  
            - Always run a fallback **color-based leaf segmentation**  
            - Estimate physical area using the background 1 cm² grid
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
        st.image(pil_image, use_container_width=True)

        # --- Estimate scale from background grid ---
        pixels_per_cm2, grid_squares = estimate_pixels_per_cm2_and_squares(image_bgr)

        with st.expander("Calibration (grid detection)", expanded=False):
            if pixels_per_cm2 is None:
                st.warning(
                    "Could not reliably detect the background grid squares. "
                    "Leaf areas will be reported only in pixels."
                )
            else:
                st.success(
                    f"Estimated ~**{pixels_per_cm2:.1f} pixels per cm²** "
                    "(based on background grid)."
                )

            # Show overlay of selected grid squares
            overlay = image_bgr.copy()
            for (x, y, w, h) in grid_squares:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Grid squares used for calibration", use_container_width=True)

        # --- Roboflow integration + debugging ---
        rf_result = None
        rf_json_str = None
        rf_error_trace = None

        api_key = st.secrets.get("ROBOFLOW_API_KEY") if hasattr(st, "secrets") else None

        st.markdown("#### Roboflow workflow (optional)")
        if not _ROBOFLOW_AVAILABLE:
            st.info("`inference_sdk` not installed; Roboflow integration is disabled.")
        elif not api_key:
            st.info("No `ROBOFLOW_API_KEY` found in Streamlit secrets – skipping Roboflow call.")
        else:
            st.caption(
                f"Using workspace: `{ROBOFLOW_WORKSPACE}`, workflow: `{ROBOFLOW_WORKFLOW_ID}`"
            )

        run_button = st.button("Run analysis", type="primary")

        if not run_button:
            return

        # 1) Try Roboflow, but don't break if it fails
        if _ROBOFLOW_AVAILABLE and api_key:
            with st.spinner("Calling Roboflow workflow..."):
                try:
                    rf_result = _call_roboflow_workflow(
                        image_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        api_key=api_key,
                    )
                    try:
                        rf_json_str = json.dumps(rf_result, indent=2, default=str)
                    except TypeError:
                        # Fallback: best-effort representation
                        rf_json_str = json.dumps(str(rf_result), indent=2)

                    st.success("Roboflow workflow call succeeded.")
                except Exception:
                    rf_error_trace = traceback.format_exc()
                    st.error(
                        "Roboflow workflow failed. Falling back to color-based segmentation."
                    )

        # 2) Debug panel for Roboflow
        with st.expander("Roboflow debug (raw JSON + traceback)", expanded=False):
            if rf_result is not None:
                st.markdown("**Raw Roboflow result**")
                try:
                    st.json(json.loads(rf_json_str))
                except Exception:
                    st.text(rf_json_str)

                st.download_button(
                    "Download Roboflow JSON",
                    data=rf_json_str.encode("utf-8"),
                    file_name="roboflow_workflow_result.json",
                    mime="application/json",
                )
            else:
                st.info("No successful Roboflow result in this run.")

            if rf_error_trace:
                st.markdown("**Roboflow error traceback**")
                st.code(rf_error_trace, language="python")

        # 3) Fallback segmentation (always runs)
        st.markdown("#### Leaf segmentation (color-based fallback)")

        mask = segment_leaves_hsv(image_bgr)
        contours = extract_leaf_contours(mask)

        # Create overlay visualization
        vis = image_bgr.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        col_vis, col_mask = st.columns(2)
        with col_vis:
            st.image(
                vis_rgb,
                caption=f"Detected leaves (contours) – count: {len(contours)}",
                use_container_width=True,
            )
        with col_mask:
            st.image(
                mask,
                caption="Binary mask (greenish pixels)",
                use_container_width=True,
            )

        # 4) Basic leaf metrics
        leaf_areas_px = [cv2.contourArea(c) for c in contours]

        if len(leaf_areas_px) == 0:
            st.warning("No leaf objects detected with the current fallback threshold.")
            return

        if pixels_per_cm2:
            # Convert to cm²
            leaf_areas_cm2 = [a / pixels_per_cm2 for a in leaf_areas_px]
            total_area_cm2 = sum(leaf_areas_cm2)
            mean_leaf_cm2 = float(np.mean(leaf_areas_cm2))
            std_leaf_cm2 = float(np.std(leaf_areas_cm2))
        else:
            leaf_areas_cm2 = None
            total_area_cm2 = None
            mean_leaf_cm2 = None
            std_leaf_cm2 = None

        st.markdown("#### Leaf metrics (fallback segmentation)")
        st.write(f"- Number of detected leaves: **{len(contours)}**")

        if pixels_per_cm2 and leaf_areas_cm2 is not None:
            st.write(f"- Total leaf area (approx): **{total_area_cm2:.1f} cm²**")
            st.write(f"- Average leaf area: **{mean_leaf_cm2:.2f} cm²**")
            st.write(f"- Leaf area standard deviation: **{std_leaf_cm2:.2f} cm²**")
        else:
            st.write("- Areas only available in pixels (no grid calibration).")

        # Optionally show a small table of per-leaf areas
        with st.expander("Per-leaf area table", expanded=False):
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
