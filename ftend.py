import streamlit as st
from PIL import Image
import os

try:
    import torch
    import timm
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ======================
# MODEL LOADER (safe)
# ======================
@st.cache_resource
def load_model_safe():
    if not TORCH_AVAILABLE:
        st.warning("Torch or timm not available. Please install them.")
        return None

    try:
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None

        model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=5)
        state_dict = torch.load(model_path, map_location='cpu')

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ======================
# Session Initialization
# ======================
def init_session():
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"
    if "show_info" not in st.session_state:
        st.session_state["show_info"] = False
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "last_uploaded" not in st.session_state:
        st.session_state["last_uploaded"] = None


# ======================
# LEFT-SIDE SETTINGS PANEL (NEW)
# ======================
def left_side_settings():
    with st.sidebar:
        st.markdown("## ⚙️ Settings / Info")

        with st.expander("🎯 Aim", expanded=False):
            st.write("Assist in early detection of Diabetic Retinopathy using AI.")

        with st.expander("📘 Purpose", expanded=False):
            st.write("Provide an easy-to-use platform for preliminary screening of retinal images.")

        with st.expander("📋 Guidelines", expanded=False):
            st.markdown("""
            - Upload a **clear retinal fundus image** (JPG/PNG).  
            - Click **Submit** to analyze.  
            - View the **color-coded result**:  
              🟢 No DR | 🟡 Mild | 🟠 Moderate | 🔴 Severe | 🩸 Proliferative
            """)

        with st.expander("🧰 Tools Used", expanded=False):
            st.markdown("""
            - Streamlit (Frontend)  
            - PyTorch + timm (Model)  
            - EfficientNet-B0 (Architecture)  
            - Pillow (Image Processing)
            """)

        with st.expander("📊 Result Info", expanded=False):
            st.markdown("""
            Displays model prediction with severity level and short interpretation.
            """)


# ======================
# WELCOME PAGE
# ======================
def welcome_page():
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-assisted, easy-to-use retinal screening tool.</p>", unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "instructions"
            st.rerun()


# ======================
# INSTRUCTIONS PAGE
# ======================
def instructions_page():
    st.markdown("<h2 style='text-align:center; color:#0b486b;'>📘 How to Use</h2>", unsafe_allow_html=True)
    st.markdown("""
    - 1️⃣ Browse and upload a clear retinal fundus image (JPG/PNG).  
    - 2️⃣ Click **Submit** to run the AI model.  
    - 3️⃣ View the **color-coded result** (No DR → Green, Mild → Yellow, Moderate → Orange, Severe → Pink, Proliferative → Red).  
    - 4️⃣ Use **Try Again** to analyze another image.
    """)
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state["page"] = "welcome"
            st.rerun()
    with c2:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state["page"] = "detection"
            st.rerun()


# ======================
# DETECTION PAGE (IMPROVED LAYOUT)
# ======================
def detection_page(model):
    st.markdown("<h2 style='text-align:center; color:#0b486b;'>🔍 Detection</h2>", unsafe_allow_html=True)
    st.write("Upload a retinal fundus image (jpg/png).")

    uploaded_file = st.file_uploader("📁 Browse image", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Preview", use_container_width=True)
        st.session_state["last_uploaded"] = uploaded_file

        st.markdown("<br>", unsafe_allow_html=True)
        # Center submit button
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        if st.button("🧠 Submit for Detection", use_container_width=False):
            if model is None:
                st.warning("Model not available or failed to load. Using demo result.")
                import random
                pred_idx = random.randint(0, 4)
            else:
                try:
                    from timm.data import resolve_model_data_config
                    from timm.data.transforms_factory import create_transform

                    config = resolve_model_data_config(model)
                    preprocess = create_transform(**config)
                    img_t = preprocess(img).unsqueeze(0)

                    with torch.no_grad():
                        preds = model(img_t)
                        pred_idx = int(preds.argmax(dim=1).item())
                except Exception as e:
                    st.error("Error during prediction.")
                    pred_idx = 0

            classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            colors = ["#27AE60", "#F7DC6F", "#F39C12", "#E05659", "#C0392B"]
            st.session_state["result"] = {"label": classes[pred_idx], "color": colors[pred_idx]}
            st.session_state["page"] = "result"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Please upload an image to proceed.")

            # bottom buttons (Back - left, Home - center, Reset - right)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Create 3 equal columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Left corner - Back button
    with col1:
        st.markdown("<div style='text-align:left;'>", unsafe_allow_html=True)
        if st.button("◀️ Back", use_container_width=False):
            st.session_state["page"] = "instructions"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Center - Home button
    with col2:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        if st.button("🏠 Home", use_container_width=False):
            st.session_state["page"] = "welcome"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Right corner - Reset button
    with col3:
        st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
        if st.button("🔁 Reset", use_container_width=False):
            st.session_state["last_uploaded"] = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ======================
# RESULT PAGE
# ======================
def result_page():
    res = st.session_state.get("result")
    if res is None:
        st.warning("No result to show. Please run detection first.")
        if st.button("Go to Detection"):
            st.session_state["page"] = "detection"
            st.rerun()
        return

    label = res["label"]
    color = res["color"]

    st.markdown(f"<div style='background:{color}; padding:18px; border-radius:12px; text-align:center;'>"
                f"<h2 style='color:white; margin:0;'>🩸 Prediction: {label}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Interpretation:")
    guidance = {
        "No DR": "No signs of diabetic retinopathy detected. Maintain routine check-ups.",
        "Mild": "Small microaneurysms visible. Recommend ophthalmologist review.",
        "Moderate": "Notable lesions present. Prompt specialist consultation advised.",
        "Severe": "Severe changes present. Urgent clinical attention recommended.",
        "Proliferative DR": "Advanced stage detected. Immediate specialist care required."
    }
    st.info(guidance.get(label, ""))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🔁 Try Another"):
            st.session_state["result"] = None
            st.session_state["page"] = "detection"
            st.rerun()
    with c2:
        if st.button("⤴️ Re-run"):
            st.session_state["page"] = "detection"
            st.rerun()
    with c3:
        if st.button("🏠 Home"):
            st.session_state["result"] = None
            st.session_state["page"] = "welcome"
            st.rerun()


# ======================
# MAIN APP
# ======================
def main():
    st.set_page_config(page_title="DR Detection", page_icon="🩸", layout="centered")
    init_session()
    left_side_settings()  # 👈 NEW: Sidebar replaces top-right info button

    model = load_model_safe()

    page = st.session_state["page"]
    if page == "welcome":
        welcome_page()
    elif page == "instructions":
        instructions_page()
    elif page == "detection":
        detection_page(model)
    elif page == "result":
        result_page()
    else:
        st.session_state["page"] = "welcome"
        welcome_page()


if __name__ == "__main__":
    main()
