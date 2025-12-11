import os
import tempfile
from typing import List, Dict, Any

import cv2
import numpy as np

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None  # We'll check at runtime


WORKSPACE_NAME = "rootweiler"
WORKFLOW_ID = "find-leaves-3"


def _get_api_key() -> str:
    """Read the Roboflow API key from env or (optionally) Streamlit secrets."""
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    # Optional: pick up from Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore

        if not api_key and "ROBOFLOW_API_KEY" in st.secrets:
            api_key = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        pass

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not set. "
            "Set export ROBOFLOW_API_KEY=... or add to Streamlit secrets."
        )
    return api_key


def run_roboflow_workflow(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Send an image to your Roboflow workflow and return the raw JSON result.
    """
    if InferenceHTTPClient is None:
        raise RuntimeError(
            "inference-sdk is not installed. Add `inference-sdk` to requirements.txt"
        )

    api_key = _get_api_key()

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    # Roboflow client wants a file path; write a temporary JPG.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        # OpenCV expects BGR, so we can just write directly
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


def extract_leaf_instances(
    workflow_result: Dict[str, Any],
    image_shape: tuple,
    leaf_step_name: str = "leaf-segmentation",
) -> List[np.ndarray]:
    """
    Convert Roboflow workflow output into a list of binary masks,
    one per leaf (same H×W as the original image).

    IMPORTANT:
    - You *must* set `leaf_step_name` to match the name of the node in
      your workflow that outputs the leaf predictions.
    - To find that, run once and inspect the JSON in Streamlit:

        st.write(result)

    This function assumes an **instance segmentation** model, where each
    prediction has a polygon under `prediction["points"]`.
    """
    h, w = image_shape[:2]

    # ---------------
    # 1) Locate the predictions in your result JSON
    # ---------------
    #
    # The exact structure depends on how you built the workflow.
    # A very common pattern is:
    #
    #   result["predictions"]["image"][leaf_step_name]["predictions"]
    #
    # After you run once, print `result` (or st.write) and update the
    # indexing below to match what you see.
    #

    try:
        step_output = workflow_result["predictions"]["image"][leaf_step_name]
        predictions = step_output["predictions"]
    except Exception as e:
        raise KeyError(
            "Could not find leaf predictions in workflow result.\n"
            "Open your result JSON (st.write(result)) and adjust the indexing in "
            "extract_leaf_instances() to match the structure.\n"
            f"Error was: {e}"
        )

    leaf_masks: List[np.ndarray] = []

    for pred in predictions:
        # For instance segmentation, Roboflow returns a polygon under "points"
        points = pred.get("points")
        if points is None:
            # if your model returns full bitmasks instead, you'd handle "mask" here
            continue

        poly = np.array(points, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        leaf_masks.append(mask)

    return leaf_masks
