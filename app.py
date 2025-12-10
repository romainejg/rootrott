import os
import io

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import leafAnalysis
import jpgExtract
import boxing
import debugger


# -----------------------
# Helpers
# -----------------------

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def crop_and_fit_image(image_path, height):
    image = Image.open(image_path)
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)
    return image.resize((width, height), Image.LANCZOS)


def go_to_page(page_name: str):
    st.session_state["page"] = page_name
    # keep query params in sync so header links work nicely
    try:
        # Newer Streamlit (st.query_params)
        qp = st.query_params
        qp["page"] = page_name
        st.query_params = qp
    except Exception:
        # Fallback for older versions
        try:
            st.experimental_set_query_params(page=page_name)
        except Exception:
            pass
    st.experimental_rerun()


# -----------------------
# Sticky header
# -----------------------

def render_header():
    script_dir = get_script_dir()
    small_logo = os.path.join(script_dir, "logo.png")

    st.markdown(
        """
        <style>
        /* Sticky header bar */
        .rr-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 120px;
            background: #111111;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 3rem;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        .rr-header-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.9rem;
        }

        .rr-logo-box {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: #000000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1rem;
        }

        .rr-header-right {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 600;
        }

        .rr-header-right a {
            color: #ffffff;
            text-decoration: none;
        }

        .rr-header-right a:hover {
            text-decoration: underline;
        }

        .rr-header-cta {
            padding: 0.4rem 1rem;
            background: #ffffff;
            color: #111111 !important;
            border-radius: 999px;
            font-weight: 700;
        }

        /* Push main content down so it isn't hidden under fixed header */
        .block-container {
            padding-top: 80px !important;
        }

        /* Background + home layout */
        .stApp {
            background: radial-gradient(circle at top left, #f5fff7, #eaf4ff 40%, #f7f7ff 80%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .rootrott-main {
            max-width: 1100px;
            margin: 0 auto;
            padding-bottom: 3rem;
        }

        .rootrott-hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            letter-spacing: 0.02em;
        }

        .rootrott-hero-subtitle {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 2rem;
        }

        .rootrott-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.2);
            height: 100%;
        }

        .rootrott-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .rootrott-card p {
            font-size: 0.95rem;
            color: #4b5563;
            margin: 0 0 0.8rem 0;
        }

        .rootrott-card-button button {
            width: 100% !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
        }

        .rootrott-footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px dashed rgba(148, 163, 184, 0.6);
            font-size: 0.8rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_text = "RR"
    if os.path.exists(small_logo):
        # could turn this into an <img> instead, but simple text works well
        logo_text = "RR"

    st.markdown(
        f"""
        <div class="rr-header">
            <div class="rr-header-left">
                <div class="rr-logo-box">{logo_text}</div>
                <div>RootRott.io</div>
            </div>
            <div class="rr-header-right">
                <a href="?page=main">Home</a>
                <a href="?page=extractor">Extractor</a>
                <a href="?page=leaf_analysis">Leaf Analysis</a>
                <a href="?page=boxing">Box Plot</a>
                <a href="?page=debugger">Debug</a>
                <a class="rr-header-cta" href="?page=main#contact">Contact</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# Pages
# -----------------------

def show_main_interface():
    script_dir = get_script_dir()
    logo_path = os.path.join(script_dir, "logo.png")
    enza_path = os.path.join(script_dir, "enza.png")
    cea_path = os.path.join(script_dir, "cea.png")

    st.markdown('<div class="rootrott-main">', unsafe_allow_html=True)

    # Hero section
    col_logo, col_text = st.columns([1, 2])
    with col_logo:
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path).resize((260, 260), Image.LANCZOS)
            st.image(logo_image, use_column_width=False)

    with col_text:
        st.markdown('<div class="rootrott-hero-title">RootRott.io</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rootrott-hero-subtitle">'
            "Leaf imaging tools for controlled environment agriculture.<br>"
            "Upload your files, analyze your plants, and export results in seconds."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Tools")

    row1 = st.columns(2)
    row2 = st.columns(2)

    # Extractor card
    with row1[0]:
        st.markdown(
            """
            <div class="rootrott-card">
                <h4>🧾 Extractor</h4>
                <p>Pull images directly from PDF, Word, PowerPoint, and Excel files for further analysis or archiving.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
        if st.button("Open Extractor", key="home_extractor"):
            go_to_page("extractor")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Leaf Analysis card
    with row1[1]:
        st.markdown(
            """
            <div class="rootrott-card">
                <h4>🌿 Leaf Analysis</h4>
                <p>Measure leaf size from images and export structured Excel reports for your trials.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
        if st.button("Open Leaf Analysis", key="home_leaf"):
            go_to_page("leaf_analysis")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Box Plot card
    with row2[0]:
        st.markdown(
            """
            <div class="rootrott-card">
                <h4>📊 Box Plot</h4>
                <p>Create publication-ready box plots from your Excel datasets with custom ordering and styling.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
        if st.button("Open Box Plot", key="home_box"):
            go_to_page("boxing")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Debugger card
    with row2[1]:
        st.markdown(
            """
            <div class="rootrott-card">
                <h4>🧪 Debugger</h4>
                <p>Visualize segmentation steps and tune HSV & watershed parameters for new cultivars and lighting setups.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rootrott-card-button">', unsafe_allow_html=True)
        if st.button("Open Debugger", key="home_debug"):
            go_to_page("debugger")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="rootrott-footer" id="contact">', unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])

    with fcol1:
        if os.path.exists(enza_path):
            try:
                enza_image = crop_and_fit_image(enza_path, 60)
                st.image(enza_image, caption="Enza Zaden")
            except Exception as e:
                st.write(f"Error loading enza.png: {e}")

    with fcol3:
        if os.path.exists(cea_path):
            try:
                cea_image = crop_and_fit_image(cea_path, 40)
                st.image(cea_image, caption="CEA Seed")
            except Exception as e:
                st.write(f"Error loading cea.png: {e}")

    with fcol2:
        st.markdown("Questions? Email: **you@example.com**")

    st.markdown("</div>", unsafe_allow_html=True)  # footer
    st.markdown("</div>", unsafe_allow_html=True)  # rootrott-main


# ---------- Extractor page ----------

def show_jpg_extract_interface():
    st.title("🧾 Extractor")
    st.write("Upload a document and extract embedded images.")

    uploaded_file = st.file_uploader(
        "Upload PDF, Word, PowerPoint, or Excel file",
        type=["pdf", "docx", "pptx", "xlsx"],
    )

    if uploaded_file is None:
        st.info("Upload a file to begin.")
        return

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.read()

    try:
        images = jpgExtract.extract_images_from_bytes(data, ext)
    except Exception as e:
        st.error(f"Error extracting images: {e}")
        return

    if not images:
        st.warning("No images big enough were found in this file.")
        return

    st.success(f"Found {len(images)} image(s).")

    for idx, (img, suggested_name) in enumerate(images):
        st.image(img, caption=suggested_name, use_column_width=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label=f"Download {suggested_name}",
            data=buf.getvalue(),
            file_name=suggested_name,
            mime="image/png",
            key=f"download_{idx}",
        )


# ---------- Leaf Analysis page ----------

def show_leaf_analysis_interface():
    st.title("🌿 Leaf Analysis")
    st.write("Upload a leaf image to segment and measure leaf objects.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to start analysis.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        return

    # Show original image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original image", use_column_width=True)

    try:
        mask, measurements, px_per_cm2 = leafAnalysis.analyze_image(image_bgr)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return

    # Show mask
    st.image(mask, caption="Segmentation mask", use_column_width=True, clamp=True)

    # Build measurement table
    data_rows = []
    for i, (x, y, w, h) in enumerate(measurements, start=1):
        width_cm = round(w / np.sqrt(px_per_cm2), 1)
        height_cm = round(h / np.sqrt(px_per_cm2), 1)
        data_rows.append(
            {"Object": i, "Width (cm)": width_cm, "Height (cm)": height_cm, "x": x, "y": y}
        )

    if not data_rows:
        st.warning("No objects above the minimum size threshold were found.")
        return

    df = pd.DataFrame(data_rows)
    st.subheader("Measurements")
    st.dataframe(df, use_container_width=True)

    # Excel download
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Measurements")
    st.download_button(
        label="Download measurements as Excel",
        data=buf.getvalue(),
        file_name="leaf_measurements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------- Box Plot page ----------

def show_boxing_interface():
    st.title("📊 Box Plot")
    st.write("Upload an Excel file and create a styled box plot.")

    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info("Upload an Excel file to continue.")
        return

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    if df.empty:
        st.warning("The uploaded Excel file appears to be empty.")
        return

    st.write("Preview of data:")
    st.dataframe(df.head(), use_container_width=True)

    columns = df.columns.tolist()
    x_column = st.selectbox("X-axis column", options=columns)
    y_column = st.selectbox("Y-axis column", options=columns)

    default_title = f"Box Plot of {y_column} by {x_column}"
    plot_title = st.text_input("Plot title", value=default_title)
    xlabel = st.text_input("X-axis label", value=x_column)
    ylabel = st.text_input("Y-axis label", value=y_column)

    if st.button("Generate box plot"):
        try:
            fig = boxing.generate_box_plot_figure(
                data=df,
                x_column=x_column,
                y_column=y_column,
                plot_title=plot_title,
                x_label=xlabel,
                y_label=ylabel,
            )
        except Exception as e:
            st.error(f"Error generating plot: {e}")
            return

        st.pyplot(fig)

        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download plot as PNG",
            data=buf.getvalue(),
            file_name="box_plot.png",
            mime="image/png",
        )


# ---------- Debugger page ----------

def show_debugger_interface():
    st.title("🧪 Leaf Segmentation Debugger")
    st.write("Tune HSV + watershed parameters and see each stage of the pipeline.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
    )

    # Parameter controls
    params = debugger.SegmentationParams()

    st.sidebar.header("Segmentation parameters")

    params.lower_hue = st.sidebar.slider("Lower Hue", 0, 179, params.lower_hue)
    params.upper_hue = st.sidebar.slider("Upper Hue", 0, 179, params.upper_hue)
    params.lower_saturation = st.sidebar.slider(
        "Lower Saturation", 0, 255, params.lower_saturation
    )
    params.upper_saturation = st.sidebar.slider(
        "Upper Saturation", 0, 255, params.upper_saturation
    )
    params.lower_value = st.sidebar.slider("Lower Value", 0, 255, params.lower_value)
    params.upper_value = st.sidebar.slider("Upper Value", 0, 255, params.upper_value)
    params.kernel_size = st.sidebar.slider("Kernel Size", 1, 15, params.kernel_size, step=2)
    params.morph_iterations = st.sidebar.slider("Morph Iterations", 0, 10, params.morph_iterations)
    params.dilate_iterations = st.sidebar.slider("Dilate Iterations", 0, 10, params.dilate_iterations)
    params.dist_transform_threshold = st.sidebar.slider(
        "Dist Transform Threshold", 0.0, 1.0, params.dist_transform_threshold, step=0.05
    )

    if uploaded_file is None:
        st.info("Upload an image to view the debug plots.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        return

    fig = debugger.create_debug_figure(image_bgr, uploaded_file.name, params)
    st.pyplot(fig)

    # Save parameter config as JSON
    config_buf = io.StringIO()
    import json

    json.dump(debugger.asdict(params), config_buf, indent=2) if hasattr(
        debugger, "asdict"
    ) else json.dump(params.__dict__, config_buf, indent=2)
    st.download_button(
        label="Download parameter config (JSON)",
        data=config_buf.getvalue(),
        file_name="segmentation_params.json",
        mime="application/json",
    )


# -----------------------
# Entry point
# -----------------------

def main():
    st.set_page_config(page_title="RootRott.io", layout="wide")

    render_header()

    # Read page from query params if present
    try:
        qp = st.query_params
        queried_page = qp.get("page")
        if queried_page:
            st.session_state["page"] = queried_page
    except Exception:
        # older Streamlit fallback
        pass

    if "page" not in st.session_state:
        st.session_state["page"] = "main"

    page = st.session_state["page"]

    if page == "extractor":
        show_jpg_extract_interface()
    elif page == "leaf_analysis":
        show_leaf_analysis_interface()
    elif page == "boxing":
        show_boxing_interface()
    elif page == "debugger":
        show_debugger_interface()
    else:
        show_main_interface()


if __name__ == "__main__":
    main()

