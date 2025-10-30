import streamlit as st
from PIL import Image
import os

# Try importing torch/timm only when needed
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
        return None
    try:
        model = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=5)
        # make sure path exists; if not, we return None
        model_path = os.path.join("model", "best_model.pth")
        if not os.path.exists(model_path):
            return None
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        # If model load fails, return None
        return None

# ======================
# Utility: init session state
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
# Top-right Info button
# ======================
def top_right_info_button():
    # create two columns, right one for button to simulate top-right placement
    c1, c2 = st.columns([9,1])
    with c2:
        if st.button("☰"):
            st.session_state["show_info"] = not st.session_state["show_info"]

    # If toggled, show a floating-like box (regular content below header)
    if st.session_state["show_info"]:
        with st.expander("App Info / Manual", expanded=True):
            st.markdown("**Aim:** Assist in early detection of Diabetic Retinopathy using AI.")
            st.markdown("**Model:** EfficientNet-B0 (trained).")
            st.markdown("**How to use:** Upload a clear fundus image → Submit → View color-coded result.")
            st.markdown("---")

# --- PAGE 1: WELCOME ---
def welcome_page():
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-assisted, easy-to-use retinal screening tool.</p>", unsafe_allow_html=True)

    st.write("")  # spacing
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "instructions"
            st.rerun()
# ======================
# PAGE: Instructions
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
    if st.button("Next ➡️", use_container_width=True):
        st.session_state["page"] = "detection"
        st.rerun()
    if st.button("Back to Home"):
        st.session_state["page"] = "welcome"
        st.rerun()

# ======================
# PAGE: Detection (Upload + Submit)
# ======================
def detection_page(model):
    st.markdown("<h2 style='text-align:center; color:#0b486b;'>🔍 Detection</h2>", unsafe_allow_html=True)
    st.write("Upload a retinal fundus image (jpg/png).")

    uploaded_file = st.file_uploader("Browse image", type=["jpg","jpeg","png"], key="uploader")
    if uploaded_file:
        # show preview
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Preview", use_container_width=True)
        st.session_state["last_uploaded"] = uploaded_file

        if st.button("🧠 Submit for Detection", use_container_width=True):
            # If model not available show friendly message
            if model is None:
                st.warning("Model not available or failed to load. Running with dummy/random result for demo.")
                # dummy result (you can remove this block and require model)
                import random
                pred_idx = random.randint(0,4)
            else:
                # preprocess and predict
                try:
                    preprocess = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                    ])
                    img_t = preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        preds = model(img_t)
                        pred_idx = int(preds.argmax(dim=1).item())
                except Exception as e:
                    st.error("Error during prediction. See console for details.")
                    pred_idx = 0

            classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            colors = ["#27AE60", "#F7DC6F", "#F39C12", "#E05659", "#C0392B"]
            st.session_state["result"] = {"label": classes[pred_idx], "color": colors[pred_idx]}
            st.session_state["page"] = "result"
            st.rerun()
    else:
        st.info("Please upload an image to proceed.")

    # small actions
    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀️ Back"):
            st.session_state["page"] = "instructions"
            st.rerun()
    with c2:
        if st.button("🏠 Home"):
            st.session_state["page"] = "welcome"
            st.rerun()

# ======================
# PAGE: Result
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
    # short guidance per class
    guidance = {
        "No DR": "No signs of diabetic retinopathy detected. Maintain routine check-ups.",
        "Mild": "Small microaneurysms visible. Recommend ophthalmologist review.",
        "Moderate": "Notable lesions present. Prompt specialist consultation advised.",
        "Severe": "Severe changes present. Urgent clinical attention recommended.",
        "Proliferative DR": "Advanced stage detected. Immediate specialist care required."
    }
    st.info(guidance.get(label, ""))

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔁 Try Another Image"):
            # reset result and go detect
            st.session_state["result"] = None
            st.session_state["page"] = "detection"
            st.rerun()
    with col2:
        if st.button("⤴️ Re-run (same image)"):
            # keep last_uploaded and go to detection (user will press submit)
            st.session_state["page"] = "detection"
            st.rerun()
    with col3:
        if st.button("🏠 Home"):
            st.session_state["result"] = None
            st.session_state["page"] = "welcome"
            st.rerun()

# ======================
# APP MAIN
# ======================
def main():
    st.set_page_config(page_title="DR Detection", page_icon="🩸", layout="centered")
    init_session()

    # top-right info toggle
    top_right_info_button()

    # Load model once (safe)
    model = load_model_safe()

    # Route pages
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
