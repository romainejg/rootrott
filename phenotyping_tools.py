# phenotyping_tools.py
#
# Bare-bones test version of the phenotyping tools.
# Goal:
# - Upload an image
# - Send it to the Roboflow "leafy" workflow
# - Display the raw JSON response in Streamlit
#
# This keeps the PhenotypingUI class and .render() method
# so that app.py does not need to change.

import io
import os
from typing import Optional

import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient


class PhenotypingUI:
    """
    Minimal phenotyping UI to prove that Roboflow integration works.

    Steps:
    - User uploads an image
    - We display the image
    - We send it to the Roboflow "leafy" workflow
    - We show the raw JSON response

    Once this works, you can plug in grid detection and measurements again.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _get_client(cls) -> Optional[InferenceHTTPClient]:
        """
        Create a Roboflow InferenceHTTPClient from Streamlit secrets.

        Supports either:
        - ROBOFLOW_API_KEY = "..."
        or
        - [roboflow]
          api_key = "..."
        in .streamlit/secrets.toml
        """
        api_key = None

        # Option 1: flat key
        if "ROBOFLOW_API_KEY" in st.secrets:
            api_key = st.secrets["ROBOFLOW_API_KEY"]

        # Option 2: nested key
        elif "roboflow" in st.secrets and "api_key" in st.secrets["roboflow"]:
            api_key = st.secrets["roboflow"]["api_key"]

        if not api_key:
            st.error(
                "Roboflow API key not found.\n\n"
                "Please add one of the following to `.streamlit/secrets.toml`:\n\n"
                'ROBOFLOW_API_KEY = "your_key_here"\n\n'
                "or\n\n"
                "[roboflow]\n"
                'api_key = "your_key_here"\n'
            )
            return None

        try:
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
            return client
        except Exception as e:
            st.error(f"Could not initialize Roboflow client: {e}")
            return None

    @classmethod
    def _run_leafy_workflow(cls, client: InferenceHTTPClient, image_path: str):
        """
        Call the 'leafy' workflow on the given image path.

        Different versions of `inference-sdk` have slightly different
        `run_workflow` signatures. The docs you pasted show:

            run_workflow(..., parameters=..., use_cache=True)

        But your installed version raises:
            unexpected keyword argument 'use_cache'

        So we:
        1. Try the "new" signature with parameters + use_cache.
        2. If we get a TypeError, fall back to a minimal call.
        """
        result = None

        try:
            # First attempt: "new" style call from the docs
            result = client.run_workflow(
                workspace_name="rootweiler",
                workflow_id="leafy",
                images={"image": image_path},
                parameters={
                    "output_message": "Your model is being initialized, try again in a few seconds."
                },
                use_cache=True,
            )
            return result
        except TypeError:
            # This is what you saw:
            #   InferenceHTTPClient.run_workflow() got an unexpected keyword argument 'use_cache'
            # So we retry with the simplest version supported by older clients.
            try:
                result = client.run_workflow(
                    workspace_name="rootweiler",
                    workflow_id="leafy",
                    images={"image": image_path},
                )
                return result
            except Exception as e:
                # Propagate as a normal error for the UI to show
                raise e
        except Exception as e:
            # Any other error bubble up
            raise e

    # ------------------------------------------------------------------
    # Public UI
    # ------------------------------------------------------------------
    @classmethod
    def render(cls):
        st.subheader("Leaf phenotyping (Roboflow test)")

        st.markdown(
            """
            This is a **minimal test** of the Roboflow integration.

            It will:
            1. Let you upload a phenotyping image  
            2. Send it to the Roboflow **leafy** workflow  
            3. Show the raw JSON response in the app  

            Once this works reliably, you can re-introduce grid
            detection, measurements, and nicer visuals on top.
            """
        )

        uploaded = st.file_uploader(
            "Upload a leaf / phenotyping image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        # --------------------------------------------------------------
        # Read the uploaded file and display it
        # --------------------------------------------------------------
        image_bytes = uploaded.read()

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        st.image(pil_img, caption="Uploaded image", use_column_width=True)

        # --------------------------------------------------------------
        # Create Roboflow client
        # --------------------------------------------------------------
        client = cls._get_client()
        if client is None:
            # Error already shown by _get_client
            return

        # --------------------------------------------------------------
        # Trigger workflow call
        # --------------------------------------------------------------
        if st.button("Run Roboflow 'leafy' workflow"):
            tmp_path = "tmp_phenotype_image.jpg"
            result = None

            # Write the image to a temporary file, because run_workflow
            # expects a path in the "images" dict.
            try:
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)

                try:
                    result = cls._run_leafy_workflow(client, tmp_path)
                except Exception as e:
                    st.error(f"Roboflow workflow call failed: {e}")
                    result = None
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            # ----------------------------------------------------------
            # Show the result
            # ----------------------------------------------------------
            if result is not None:
                st.success("Successfully received a response from Roboflow:")
                st.json(result)
            else:
                st.warning("No result returned from Roboflow.")
