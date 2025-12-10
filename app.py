import os
import io
import json

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


# -----------------------
# Global styling
# -----------------------

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700;800&display=swap');

        :root {
            --rw-green: #45C96B;
            --rw-purple: #8C8BFF;
            --rw-grey: #E5E5E5;
            --rw-yellow: #FFD750;
            --rw-red: #ED695D;
            --rw-dark: #111111;
            --rw-light-bg: #F5FAFF;
        }

        /* App background + Typography */
        .stApp {
            background: #ffffff;
            font-family: "Rubik", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Urbane Rounded", "Rubik", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #111111;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Sidebar styling – light, readable */
        [data-testid="stSidebar"] {
            background: #F5F7FB;
            color: #111827;
            border-right: 1px solid #E5E7EB;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #111827;
        }

        .rw-sidebar-title {
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #111827;
            margin-bottom: 0.15rem;
        }
        .rw-sidebar-subtitle {
            font-size: 0.8rem;
            color: #4B5563;
            margin-bottom: 0.9rem;
        }

        /* Home hero */
        .rw-hero-hello {
            font-size: 1.4rem;
            color: #4B5563;
            margin-bottom: 0.2rem;
        }
        .rw-hero-name {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }
        .rw-hero-intro {
            font-size: 0.98rem;
            color: #4B5563;
            line-height: 1.6;
        }

        .rw-hero-icon-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1.4rem;
        }
        .rw-hero-icon-circle {
            top: 60px;
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: #111111;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #F9FAFB;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .rw-hero-icon-text {
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6B7280;
        }

        /* Thin divider */
        .rw-divider {
            border: none;
            border-top: 1px solid #E5E7EB;
            margin: 2.5rem 0 1.5rem 0;
        }

        /* Bottom contact row */
        .rw-contact-label {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 0.25rem;
        }
        .rw-contact-value {
            font-size: 1.0rem;
            font-weight: 500;
            color: #111827;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# Sidebar navigation
# -----------------------

def sidebar_nav():
    script_dir = get_script_dir()
    logo_path = os.path.join(script_dir, "logo.png")

    with st.sidebar:
        cols = st.columns([1, 3])
        with cols[0]:
            if os.path.exists(logo_path):
                # Larger icon for sidebar
                logo_img = Image.open(logo_path).resize((80, 80), Image.LANCZOS)
                st.image(logo_img, use_column_width=True)
        with cols[1]:
            st.markdown('<div class="rw-sidebar-title">ROOTWEILER</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rw-sidebar-subtitle">A non-profit experiment in digital support for greenhouse teams.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        section = st.radio(
            "Sections",
            [
                "Home",
                "Calculators",
                "Climate",
                "Phenotyping",
                "Data & Graphs",
                "Imaging",
            ],
            index=0,
        )

        st.markdown("---")
        st.caption("Tip: All processing happens in the cloud. No local installs needed.")

    return section


# -----------------------
# Section: Home (minimal hero layout)
# -----------------------

def render_home():
    script_dir = get_script_dir()
    lettuce1 = os.path.join(script_dir, "lettuce1.jpg")
    enza_path = os.path.join(script_dir, "enza.png")
    cea_path = os.path.join(script_dir, "cea.png")
    logo_path = os.path.join(script_dir, "logo.png")

    # Small icon row at very top of page
    st.markdown(
        '<div class="rw-hero-icon-row">'
        '<div class="rw-hero-icon-circle">R</div>'
        '<div class="rw-hero-icon-text">Rootweiler • greenhouse support</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Hero: image left, text right
    col_img, col_text = st.columns([1.1, 1.4])

    with col_img:
        if os.path.exists(lettuce1):
            hero_img = Image.open(lettuce1)
        elif os.path.exists(logo_path):
            hero_img = Image.open(logo_path)
        else:
            hero_img = None

        if hero_img is not None:
            st.image(hero_img, use_column_width=True)
        else:
            st.write("")

    with col_text:
        st.markdown('<div class="rw-hero-hello">Hello!</div>', unsafe_allow_html=True)
        st.markdown('<div class="rw-hero-name">I’m Rootweiler.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rw-hero-intro">'
            "A quiet, non-profit tool for people who live their days between crops, climate graphs and spreadsheets. "
            "Rootweiler is a place to prototype small digital helpers for controlled environments – without turning "
            "every idea into a full-scale project."
            "<br><br>"
            "Right now it’s just a sketch: a few sections to explore **Calculators, Climate, Phenotyping, Data & Graphs, "
            "and Imaging**. The shape of it will come from the growers, breeders and agronomists who try it."
            "</div>",
            unsafe_allow_html=True,
        )

    # Thin divider
    st.markdown('<hr class="rw-divider">', unsafe_allow_html=True)

    # Bottom row: logo – contact – logo, all centered with similar heights
    col_left, col_center, col_right = st.columns([1, 1.4, 1])

    with col_left:
        if os.path.exists(enza_path):
            try:
                st.image(crop_and_fit_image(enza_path, 40), use_column_width=False)
            except Exception:
                pass

    with col_center:
        st.markdown('<div class="rw-contact-label">Contact</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rw-contact-value">rootweiler (at) example.org</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        if os.path.exists(cea_path):
            try:
                st.image(crop_and_fit_image(cea_path, 40), use_column_width=False)
            except Exception:
                pass


# -----------------------
# Section: Calculators (concept placeholder)
# -----------------------

def render_calculators():
    st.markdown("## Calculators")
    st.markdown(
        """
        Over time, _Calculators_ can host small, focused tools that help with everyday questions in the greenhouse —
        from planning to crop steering.

        This could include things like light planning, time-to-harvest estimates or simple rule-of-thumb checks.
        For now, this area is intentionally open, a space to collect ideas from the teams using it.
        """
    )
    st.info("If you have a specific calculator in mind, make a note of it – that helps decide what lands here first.")


# -----------------------
# Section: Climate (concept placeholder)
# -----------------------

def render_climate():
    st.markdown("## Climate")
    st.markdown(
        """
        _Climate_ is meant as a bridge between environmental data and the crop you see in front of you.

        The long-term idea is to make it easier to:
        - Organize climate exports from different systems  
        - Look at patterns over time rather than single days  
        - Connect climate views with the observations made in Phenotyping and Imaging  

        Right now this is a placeholder for that direction.
        """
    )


# -----------------------
# Section: Imaging
# -----------------------

def render_imaging():
    st.markdown("## Imaging")
    st.markdown(
        """
        _Imaging_ focuses on the visual side of greenhouse work – photos, documents, and other files
        that carry information about the crop and experiments.

        This space will grow into a small collection of ways to bring those visuals into a more workable format.
        """
    )

    tab_extractor, = st.tabs(["Image Extractor"])
    with tab_extractor:
        render_extractor_tool()


def render_extractor_tool():
    st.subheader("Image Extractor")
    st.write("Upload a document and collect the images it contains into a simple gallery.")

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
        st.error(f"Something went wrong while reading the file: {e}")
        return

    if not images:
        st.warning("No suitable images were found in this file.")
        return

    st.success(f"Found {len(images)} image(s).")

    for idx, (img, suggested_name) in enumerate(images):
        st.image(img, caption=None, use_column_width=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            label=f"Download image {idx + 1}",
            data=buf.getvalue(),
            file_name=suggested_name,
            mime="image/png",
            key=f"download_{idx}",
        )


# -----------------------
# Section: Phenotyping
# -----------------------

def render_phenotyping():
    st.markdown("## Phenotyping")
    st.markdown(
        """
        _Phenotyping_ explores ways to look at crops more systematically through images – not to replace
        grower knowledge, but to give it another lens.

        The tools here are starting points for thinking about plant structure, leaf development and canopy patterns.
        """
    )

    tab_leaf, tab_debug = st.tabs(["Leaf views", "Segmentation tuning"])

    with tab_leaf:
        render_leaf_analysis_tool()

    with tab_debug:
        render_debugger_tool()


def render_leaf_analysis_tool():
    st.subheader("Leaf views")
    st.write("Upload a leaf image to explore basic measurements and structure.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_file",
    )

    if uploaded_file is None:
        st.info("Upload an image to start.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption=None, use_column_width=True)

    try:
        mask, measurements, px_per_cm2 = leafAnalysis.analyze_image(image_bgr)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Something went wrong during processing: {e}")
        return

    st.image(mask, caption=None, use_column_width=True, clamp=True)

    rows = []
    for i, (x, y, w, h) in enumerate(measurements, start=1):
        width_cm = round(w / np.sqrt(px_per_cm2), 1)
        height_cm = round(h / np.sqrt(px_per_cm2), 1)
        rows.append(
            {"Object": i, "Width (cm)": width_cm, "Height (cm)": height_cm, "x": x, "y": y}
        )

    if not rows:
        st.warning("No objects above the minimum size threshold were found.")
        return

    df = pd.DataFrame(rows)
    st.markdown("#### Table view")
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Measurements")
    st.download_button(
        label="Download table (Excel)",
        data=buf.getvalue(),
        file_name="leaf_measurements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_debugger_tool():
    st.subheader("Segmentation tuning")
    st.write("Adjust thresholds and see how the image is broken down step by step.")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="debug_file",
    )

    params = debugger.SegmentationParams()

    st.markdown("##### Parameters")
    c1, c2 = st.columns(2)

    with c1:
        params.lower_hue = st.slider("Lower Hue", 0, 179, params.lower_hue)
        params.upper_hue = st.slider("Upper Hue", 0, 179, params.upper_hue)
        params.lower_saturation = st.slider(
            "Lower Saturation", 0, 255, params.lower_saturation
        )
        params.upper_saturation = st.slider(
            "Upper Saturation", 0, 255, params.upper_saturation
        )
        params.lower_value = st.slider("Lower Value", 0, 255, params.lower_value)
        params.upper_value = st.slider("Upper Value", 0, 255, params.upper_value)

    with c2:
        params.kernel_size = st.slider("Kernel Size (odd numbers)", 1, 15, params.kernel_size, step=2)
        params.morph_iterations = st.slider("Morph Iterations", 0, 10, params.morph_iterations)
        params.dilate_iterations = st.slider("Dilate Iterations", 0, 10, params.dilate_iterations)
        params.dist_transform_threshold = st.slider(
            "Distance Transform Threshold", 0.0, 1.0, params.dist_transform_threshold, step=0.05
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

    config_str = json.dumps(params.__dict__, indent=2)
    st.download_button(
        label="Download parameter set (JSON)",
        data=config_str,
        file_name="segmentation_params.json",
        mime="application/json",
    )


# -----------------------
# Section: Data & Graphs
# -----------------------

def render_data_graphs():
    st.markdown("## Data & Graphs")
    st.markdown(
        """
        _Data & Graphs_ is a place to give tables a bit more shape – quick visual checks rather than full analysis.

        The idea is to make it easy to move from a sheet of numbers to a simple graphic you can show in a
        greenhouse meeting or trial discussion.
        """
    )

    tab_box, = st.tabs(["Box view"])
    with tab_box:
        render_box_plot_tool()


def render_box_plot_tool():
    st.subheader("Box view")
    st.write("Upload an Excel file and build a quick box-style overview.")

    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        key="box_file",
    )

    if uploaded_file is None:
        st.info("Upload an Excel file to continue.")
        return

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    if df.empty:
        st.warning("The uploaded file appears to be empty.")
        return

    st.write("Preview:")
    st.dataframe(df.head(), use_container_width=True)

    columns = df.columns.tolist()
    if len(columns) < 2:
        st.warning("Please provide a table with at least two columns.")
        return

    x_column = st.selectbox("Group by (x-axis)", options=columns)
    y_column = st.selectbox("Values (y-axis)", options=columns, index=min(1, len(columns) - 1))

    default_title = f"{y_column} by {x_column}"
    plot_title = st.text_input("Title", value=default_title)
    xlabel = st.text_input("X label", value=x_column)
    ylabel = st.text_input("Y label", value=y_column)

    if st.button("Create view"):
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
            st.error(f"Something went wrong while drawing the plot: {e}")
            return

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download image (PNG)",
            data=buf.getvalue(),
            file_name="box_view.png",
            mime="image/png",
        )


# -----------------------
# Main app
# -----------------------

def main():
    st.set_page_config(page_title="Rootweiler", layout="wide")
    inject_css()

    section = sidebar_nav()

    if section == "Home":
        render_home()
    elif section == "Calculators":
        render_calculators()
    elif section == "Climate":
        render_climate()
    elif section == "Phenotyping":
        render_phenotyping()
    elif section == "Data & Graphs":
        render_data_graphs()
    elif section == "Imaging":
        render_imaging()
    else:
        render_home()


if __name__ == "__main__":
    main()

