import os
import signal
from io import BytesIO

import streamlit as st
from PIL import Image

def close_browser_tab():
    st.markdown(
        """
        <script>
            window.close();
        </script>
        """,
        unsafe_allow_html=True,
    )


# ======================
# Optional: Import Torch/Timm safely
# ======================
try:
    import torch
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"


# ======================
# MODEL LOADER (CACHED)
# ======================
@st.cache_resource
def load_model_safe():
    """Loads the EfficientNet-B0 model with trained weights once per session."""
    if not TORCH_AVAILABLE:
        st.warning("Torch or timm not available. Please install them.")
        return None, None

    try:
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None, None

        model = timm.create_model(
            "tf_efficientnet_b0",
            pretrained=False,
            num_classes=5
        )

        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()

        config = resolve_model_data_config(model)
        preprocess = create_transform(**config)

        return model, preprocess

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# ======================
# SESSION INITIALIZATION
# ======================
def init_session():
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"
    if "result" not in st.session_state:
        st.session_state["result"] = None


# ======================
# SIDEBAR INFORMATION + EXIT
# ======================
def left_side_settings():
    with st.sidebar:
        st.markdown("## ℹ Information")

        with st.expander(" Aim"):
            st.write("Assist in early detection of Diabetic Retinopathy using AI from retinal fundus images.")

        with st.expander(" Purpose"):
            st.write("Provide an easy-to-use platform for preliminary DR screening and awareness.")

        with st.expander(" Guidelines"):
            st.markdown(
                """
            - Upload a clear retinal fundus image (JPG/PNG).  
            - Click Submit to analyze.  
            - View the color-coded result:  
              -  No DR  
              -  Mild  
              -  Moderate  
              -  Severe  
              -  Proliferative DR
            """
            )

        with st.expander(" Tools Used"):
            st.markdown(
                """
            - Streamlit (Web Interface)  
            - PyTorch + timm (Deep Learning)  
            - EfficientNet-B0 (Model Architecture)  
            - Pillow (Image Handling)
            """
            )

        with st.expander(" Result Info"):
            st.markdown(
                """
            Displays the predicted diabetic retinopathy severity level  
            along with a short interpretation message.
            """
            )
        


# ======================
# WELCOME PAGE
# ======================
def welcome_page():
    st.markdown(
        "<h1 style='text-align:center; color:#003366;'>🩸 Diabetic Retinopathy Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>AI-assisted retinal screening using deep learning.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.page = "instructions"
        st.rerun()


# ======================
# INSTRUCTIONS PAGE
# ======================
def instructions_page():
    st.markdown("<h2 style='text-align:center; color:#003366;'> How to Use</h2>", unsafe_allow_html=True)
    st.markdown("""
    -  Browse and upload a clear retinal fundus image (JPG/PNG).  
    -  Click Submit to run the AI model.  
    -  View the color-coded result (No DR → Green, Mild → Yellow, Moderate → Orange, Severe → Pink, Proliferative → Red).  
    -  Use Try Again to analyze another image.
    """)
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state["page"] = "welcome"
            st.rerun()
    with c2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state["page"] = "detection"
            st.rerun()


# ======================
# DETECTION PAGE
# ======================
def detection_page(model, preprocess):
    st.markdown(
        "<h2 style='text-align:center; color:#003366;'> DR Detection Panel</h2>",
        unsafe_allow_html=True,
    )
    st.info("Upload a retinal fundus image and click Submit for Detection to get the predicted severity level.")

    # Layout: Upload on left, preview on right
    col_left, col_right = st.columns([1.2, 1])

    img = None
    image_loaded = False
    image_name = None

    with col_left:
        uploaded_file = st.file_uploader("📁 Upload Retinal Image", type=["jpg", "jpeg", "png"])

        # If you do not want dataset-code search (for speed), we skip the text input:
        image_input = st.text_input(
            "Or enter local file path (optional)", placeholder="Example: C:\\path\\to\\image.png"
        )

    # CASE 1: Uploaded Image
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_name = os.path.splitext(uploaded_file.name)[0]
            image_loaded = True
        except Exception as e:
            st.error(f"⚠ Could not open uploaded image: {e}")
            st.stop()

    # CASE 2: Local file path
    elif image_input.strip():
        path_text = image_input.strip().strip('"').strip("'").replace("\\\\", "\\")
        if os.path.exists(path_text):
            try:
                img = Image.open(path_text).convert("RGB")
                image_name = os.path.splitext(os.path.basename(path_text))[0]
                image_loaded = True
            except Exception as e:
                st.error(f"Error loading image from path: {e}")
        else:
            st.warning("⚠ File path not found. Please check the path and try again.")

    # Show preview if loaded
    with col_right:
        if image_loaded and img is not None:
            st.image(img, caption="🖼 Image Preview", use_container_width=True)
        else:
            st.info("Image preview will appear here after you upload or select a valid image.")

    st.markdown("---")

    # Detection button (centered)
    if image_loaded and img is not None:
        center_col = st.columns(3)[1]
        with center_col:
            if st.button(" Submit for Detection", use_container_width=True):
                try:
                    if model is None or preprocess is None:
                        st.warning("Model not available or failed to load. Using demo result.")
                        import random

                        pred_idx = random.randint(0, 4)
                    else:
                        # Preprocess and predict
                        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            preds = model(img_t)
                            pred_idx = int(preds.argmax(dim=1).item())

                    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
                    colors = ["#27AE60", "#F7DC6F", "#F39C12", "#E05659", "#C0392B"]

                    st.session_state["result"] = {
                        "label": classes[pred_idx],
                        "color": colors[pred_idx],
                    }
                    st.session_state["page"] = "result"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.info("Please upload an image or provide a valid file path to continue.")

    # Bottom Navigation Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(" Back", use_container_width=True):
            st.session_state["page"] = "instructions"
            st.rerun()

    with col2:
        if st.button(" Home", use_container_width=True):
            st.session_state["page"] = "welcome"
            st.rerun()

    with col3:
        if st.button(" Exit", use_container_width=True):
            st.warning("Application is shutting down. You can now close this tab.")
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)
# ======================
# RESULT PAGE
# ======================
def result_page():
    res = st.session_state.get("result")
    if res is None:
        st.warning("No result available. Start detection first.")
        if st.button("Go to Detection"):
            st.session_state.page = "detection"
            st.rerun()
        return

    label = res["label"]
    color = res["color"]

    st.markdown(
        f"<div style='background:{color}; padding:18px; border-radius:12px; text-align:center;'>"
        f"<h2 style='color:white;'>🩸 Prediction: {label}</h2></div>",
        unsafe_allow_html=True,
    )

    messages = {
        "No DR": "No visible DR signs. Continue routine eye checkups.",
        "Mild": "Early lesions detected. Consult ophthalmologist.",
        "Moderate": "Notable damage. Seek medical review soon.",
        "Severe": "Severe changes. Urgent care recommended.",
        "Proliferative DR": "Vision-threatening stage. Immediate treatment needed."
    }

    st.info(messages.get(label, ""))

    col1, col2 = st.columns(2)
    if col1.button(" Try Another", use_container_width=True):
        st.session_state["page"] = "detection"
        st.rerun()
    if col2.button(" Home", use_container_width=True):
        st.session_state["page"] = "welcome"
        st.rerun()


# ======================
# MAIN APP
# ======================
def main():
    st.set_page_config(page_title="DR Detection", page_icon="🩸", layout="centered")
    init_session()
    left_side_settings()

    model, preprocess = load_model_safe()

    page = st.session_state["page"]
    if page == "welcome":
        welcome_page()
    elif page == "instructions":
        instructions_page()
    elif page == "detection":
        detection_page(model, preprocess)
    elif page == "result":
        result_page()


if _name_ == "_main_":
    main()