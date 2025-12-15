# imaging_tools.py

import io
import os
import zipfile
from typing import List, Tuple

import streamlit as st
from PIL import Image
import numpy as np

# For document image extraction
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import openpyxl


class ImagingToolsUI:
    """Imaging utilities for the Rootweiler 'Imaging' section."""

    @classmethod
    def render(cls):
        st.subheader("Imaging")

        tabs = st.tabs(["Image extractor", "Compressor"])

        with tabs[0]:
            cls._render_extractor()

        with tabs[1]:
            cls._render_compressor()

    # ------------------------------------------------------------------
    # IMAGE EXTRACTOR
    # ------------------------------------------------------------------
    @classmethod
    def _render_extractor(cls):
        st.markdown(
            """
            ### Image extractor
            
            Upload a report or slide deck and pick which images you want to keep.
            Supports **PDF, PowerPoint (PPTX), Word (DOCX), and Excel (XLSX)**.
            """
        )

        uploaded = st.file_uploader(
            "Upload a single document",
            type=["pdf", "pptx", "docx", "xlsx"],
            key="imaging_extractor_uploader",
        )

        if uploaded is not None and st.button("Extract images", type="primary", key="extract_btn"):
            file_bytes = uploaded.getvalue()
            filename = uploaded.name
            try:
                images = cls._extract_images_from_file(file_bytes, filename)
            except Exception as e:
                st.error(f"Could not extract images: {e}")
                return

            if not images:
                st.warning("No images found (or all images were too small).")
                st.session_state["extracted_images"] = []
            else:
                # Store in session so selection persists across reruns
                st.session_state["extracted_images"] = [
                    {
                        "name": name,
                        "bytes": cls._pil_to_bytes(img),
                        "selected": True,
                    }
                    for img, name in images
                ]

        img_state = st.session_state.get("extracted_images", [])

        if not img_state:
            st.info("Extracted images will appear here after you upload and run extraction.")
            return

        st.markdown("#### Review and select images")

        # Select all / none
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Select all", key="select_all_imgs"):
                for item in img_state:
                    item["selected"] = True
        with c2:
            if st.button("Deselect all", key="deselect_all_imgs"):
                for item in img_state:
                    item["selected"] = False

        # Display images in a grid with checkboxes
        cols_per_row = 3
        for i in range(0, len(img_state), cols_per_row):
            row_items = img_state[i : i + cols_per_row]
            cols = st.columns(len(row_items))
            for col, item, idx in zip(cols, row_items, range(i, i + len(row_items))):
                with col:
                    st.image(item["bytes"], use_column_width=True)
                    checked = st.checkbox(
                        item["name"],
                        value=item.get("selected", True),
                        key=f"img_select_{idx}",
                    )
                    item["selected"] = checked

        # Update state
        st.session_state["extracted_images"] = img_state

        # Prepare ZIP download of selected images
        selected = [item for item in img_state if item["selected"]]
        if not selected:
            st.warning("No images selected for download.")
            return

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in selected:
                zf.writestr(item["name"], item["bytes"])
        zip_buffer.seek(0)

        st.download_button(
            "Download selected images (.zip)",
            data=zip_buffer,
            file_name="rootweiler_extracted_images.zip",
            mime="application/zip",
            key="download_extracted_zip",
        )

    # ----------------- extraction helpers ----------------- #

    @staticmethod
    def _extract_images_from_file(file_bytes: bytes, filename: str) -> List[Tuple[Image.Image, str]]:
        """Return list of (PIL.Image, suggested_name) from PDF / PPTX / DOCX / XLSX."""
        ext = os.path.splitext(filename)[1].lower()
        buf = io.BytesIO(file_bytes)

        images: List[Tuple[Image.Image, str]] = []

        if ext == ".pdf":
            doc = fitz.open(stream=buf.read(), filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    im = Image.open(io.BytesIO(img_bytes))
                    if im.width >= 30 and im.height >= 30:
                        name = f"pdf_page_{page_idx+1}_img_{img_index+1}.png"
                        images.append((im, name))

        elif ext == ".docx":
            doc = Document(buf)
            img_index = 0
            for rel in doc.part.rels:
                if "image" in doc.part.rels[rel].target_ref:
                    img_data = doc.part.rels[rel].target_part.blob
                    im = Image.open(io.BytesIO(img_data))
                    if im.width >= 30 and im.height >= 30:
                        img_index += 1
                        name = f"word_img_{img_index}.png"
                        images.append((im, name))

        elif ext == ".pptx":
            prs = Presentation(buf)
            img_index = 0
            for slide_idx, slide in enumerate(prs.slides, start=1):
                for shape in slide.shapes:
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        img_data = shape.image.blob
                        im = Image.open(io.BytesIO(img_data))
                        if im.width >= 30 and im.height >= 30:
                            img_index += 1
                            name = f"ppt_slide_{slide_idx}_img_{img_index}.png"
                            images.append((im, name))

        elif ext == ".xlsx":
            wb = openpyxl.load_workbook(buf)
            img_index = 0
            for sheet in wb:
                # _images is semi-private but commonly used
                for img in getattr(sheet, "_images", []):
                    try:
                        img_data = img.ref.blob  # may differ by openpyxl version
                    except AttributeError:
                        continue
                    im = Image.open(io.BytesIO(img_data))
                    if im.width >= 30 and im.height >= 30:
                        img_index += 1
                        safe_title = "".join(
                            c if c.isalnum() else "_" for c in sheet.title
                        )[:20]
                        name = f"xlsx_{safe_title}_img_{img_index}.png"
                        images.append((im, name))

        return images

    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # COMPRESSOR (same behavior as before)
    # ------------------------------------------------------------------
    @classmethod
    def _render_compressor(cls):
        st.markdown(
            """
            ### Image / file compressor
            
            Upload images or other files, choose a target size, and download a compressed ZIP.
            
            Images are compressed by:
            - Optional resizing to a maximum dimension
            - JPEG re-encoding with adjusted quality to aim for your target size
            
            Non-image files are included unchanged in the ZIP.
            """
        )

        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=None,
            accept_multiple_files=True,
            key="imaging_compressor_uploader",
        )

        if not uploaded_files:
            st.info("Add a few images or files above to get started.")
            return

        st.markdown("#### Compression settings")

        col1, col2 = st.columns(2)

        with col1:
            target_size_kb = st.slider(
                "Approximate target size per image (kB)",
                min_value=50,
                max_value=5000,
                value=500,
                step=50,
                help="The compressor will aim for this size per image by adjusting JPEG quality.",
            )

        with col2:
            resize_option = st.selectbox(
                "Max image dimension (longest side)",
                options=[
                    "No resizing",
                    "800 px",
                    "1200 px",
                    "2000 px",
                ],
                index=1,
                help="Resizing down first often gives better quality at smaller file sizes.",
            )

        max_dim = None
        if resize_option != "No resizing":
            max_dim = int(resize_option.split()[0])

        if st.button("Run compression", type="primary", key="compress_btn"):
            cls._run_compression(uploaded_files, target_size_kb, max_dim)

    @staticmethod
    def _run_compression(uploaded_files, target_size_kb: int, max_dim: int | None):
        compressed_files: list[tuple[str, bytes]] = []
        summary_rows = []

        for f in uploaded_files:
            raw_bytes = f.getvalue()
            orig_size_kb = len(raw_bytes) / 1024.0
            ext = os.path.splitext(f.name)[1].lower()

            is_image = False
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()
                is_image = True
            except Exception:
                is_image = False

            if is_image:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

                compressed_bytes = ImagingToolsUI._compress_image(
                    img,
                    target_size_kb=target_size_kb,
                    max_dim=max_dim,
                )

                new_name = os.path.splitext(f.name)[0] + "_compressed.jpg"
                compressed_files.append((new_name, compressed_bytes))

                new_size_kb = len(compressed_bytes) / 1024.0
                ratio = new_size_kb / orig_size_kb if orig_size_kb > 0 else 1.0

                summary_rows.append(
                    {
                        "File": new_name,
                        "Original (kB)": round(orig_size_kb, 1),
                        "Compressed (kB)": round(new_size_kb, 1),
                        "Size ratio": f"{ratio:.2f}x",
                    }
                )
            else:
                compressed_files.append((f.name, raw_bytes))
                summary_rows.append(
                    {
                        "File": f.name,
                        "Original (kB)": round(orig_size_kb, 1),
                        "Compressed (kB)": round(orig_size_kb, 1),
                        "Size ratio": "1.00x (unchanged)",
                    }
                )

        if not compressed_files:
            st.warning("Nothing to compress – no valid files detected.")
            return

        st.markdown("#### Compression summary")
        st.dataframe(summary_rows, use_container_width=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in compressed_files:
                zf.writestr(name, data)
        zip_buffer.seek(0)

        st.download_button(
            "Download compressed files (.zip)",
            data=zip_buffer,
            file_name="rootweiler_compressed_files.zip",
            mime="application/zip",
            key="download_compressed_zip",
        )

    @staticmethod
    def _compress_image(
        img: Image.Image,
        target_size_kb: int,
        max_dim: int | None = None,
    ) -> bytes:
        """Resize (optional) and JPEG-compress an image, trying to hit target_size_kb."""
        if max_dim is not None:
            w, h = img.size
            longest = max(w, h)
            if longest > max_dim and longest > 0:
                scale = max_dim / float(longest)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.LANCZOS)

        low_q, high_q = 20, 95
        best_bytes = None

        for _ in range(8):
            q = (low_q + high_q) // 2
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data) / 1024.0
            best_bytes = data

            if size_kb > target_size_kb and q > low_q:
                high_q = q - 1
            else:
                low_q = q + 1

        return best_bytes or b""
