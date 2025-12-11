# phenotyping_tools.py - Bare Bones Mask Output

import io
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict

# Roboflow SDK Import
from inference_sdk import InferenceHTTPClient

# ---------------------------------------------------------------------
# Roboflow SDK Integration (Minimal Version - Re-using the last confirmed working logic)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Roboflow segmentation...")
def _run_roboflow_workflow(image_bytes: bytes) -> Tuple[Optional[Dict[str, object]], bool]:
    # 1. Get and Clean API Key
    api_key = None
    if "ROBOFLOW_API_KEY" in st.secrets:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
        api_key = st.secrets["roboflow"]["api_key"]

    if not api_key:
        # Authentication failed/key not found
        return None, False

    api_key = api_key.strip()

    # 2. Setup Client
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
    except Exception:
        return None, True # Attempted with key

    # 3. Save Temp File
    tmp_path = "tmp_phenotype_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    # 4. Run Workflow
    result = None
    try:
        result = client.run_workflow(
            workspace_name="rootweiler",
            workflow_id="leafy",
            images={
                "image": tmp_path
            },
            parameters={
                "output_message": "Segmentation started."
            }
        )
    except Exception:
        # Error during API call (e.g., Auth failure, network issue)
        return None, True 
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 5. Process Results
    if not isinstance(result, list) or len(result) == 0:
        return None, True

    obj = result[0]
    preds = obj.get("output2") # Assuming 'output2' is the segmentation predictions
    
    if not isinstance(preds, list) or len(preds) == 0:
        return None, True

    return {"predictions": preds}, True


def _mask_from_roboflow_predictions(image_shape: Tuple[int, int, int], predictions: List[dict]) -> np.ndarray:
    """
    Build a binary mask from Roboflow instance segmentation polygons.
    """
    h, w, _ = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pred in predictions:
        pts = pred.get("points")
        if not pts or not isinstance(pts, list): continue
        poly = np.array([[int(p["x"]), int(p["y"])] for p in pts if "x" in p and "y" in p], dtype=np.int32)
        if poly.shape[0] < 3: continue
        cv2.fillPoly(mask, [poly], 255)
    return mask


# ---------------------------------------------------------------------
# Streamlit UI (Bare Bones)
# ---------------------------------------------------------------------
class PhenotypingUI:
    """Bare-bones UI to upload image and display segmentation mask."""

    @classmethod
    def render(cls):
        st.subheader("Leaf Segmentation Mask Generator")
        st.markdown("Upload an image to generate the instance segmentation mask.")

        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)

        # --- Segmentation ---
        with st.spinner("Generating mask..."):
            rf_result, attempted_rf = _run_roboflow_workflow(image_bytes)

        if rf_result is not None:
            mask_bin = _mask_from_roboflow_predictions(img_rgb.shape, rf_result["predictions"])
            st.success(f"✅ Segmentation Success: Found {len(rf_result['predictions'])} leaf instances.")
            
            # Use original and mask side-by-side
            c1, c2 = st.columns(2)
            with c1: 
                st.image(img_rgb, caption="Original Image", use_container_width=True)
            with c2: 
                st.image(mask_bin, caption="Generated Binary Mask", use_container_width=True)

        else:
            if attempted_rf:
                st.error("❌ Roboflow failed to return a mask. Check API Key and model logs.")
            else:
                st.error("❌ API Key not configured. Cannot generate mask.")
