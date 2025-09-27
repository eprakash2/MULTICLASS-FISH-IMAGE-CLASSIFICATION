import os
import time
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="🎨 Image Classifier — Demo", page_icon="🐟", layout="centered")

# -----------------------
# UI Styling
# -----------------------
st.markdown("""
<style>
.reportview-container .main {
  background: linear-gradient(135deg, #f6f8ff 0%, #ffffff 40%, #f7fff1 100%);
}
.card {
  padding: 1rem;
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,255,0.8));
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  margin-bottom: 1rem;
}
.small-muted { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Interactive Image Classification Dashboard")
st.write("Upload an image, pick a model (or upload one) and click **Predict**. Shows top-k classes and confidence.")

# -----------------------
# Helpers
# -----------------------
@st.cache_resource(show_spinner=False)
def list_local_models(path="models"):
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path) if f.lower().endswith(".h5")]

@st.cache_resource(show_spinner=False)
def load_model_cached(model_path):
    model = load_model(model_path, compile=False)
    try:
        inp_shape = model.inputs[0].shape.as_list()
    except Exception:
        inp_shape = getattr(model, "input_shape", None)
    if inp_shape is None:
        target = (224, 224, 3)
    elif len(inp_shape) == 4:
        _, h, w, c = inp_shape
        target = (h or 224, w or 224, c or 3)
    else:
        target = (224, 224, 3)
    return model, tuple(map(int, target))

def preprocess_image_pil(pil_img: Image.Image, target_size):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, (target_size[1], target_size[0]), method=Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def top_k_from_pred(preds, k=5):
    preds = np.asarray(preds).squeeze()
    idx = preds.argsort()[::-1][:k]
    return list(zip(idx.tolist(), preds[idx].tolist()))

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("⚙️ Settings")
local_models = list_local_models("models")
model_choice = st.sidebar.selectbox("Choose local .h5 model", ["-- none --"] + local_models)

uploaded_model_file = st.sidebar.file_uploader("Upload custom .h5", type=["h5"])
top_k = st.sidebar.slider("Top K predictions", 1, 10, 3)
show_conf_table = st.sidebar.checkbox("Show full class table", True)

# -----------------------
# Upload image
# -----------------------
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_image = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
    pil_img = None
    if uploaded_image:
        try:
            pil_img = Image.open(uploaded_image)
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
        except Exception:
            st.error("❌ Could not read image.")

with col2:
    st.markdown('<div class="card"><h4>Model status</h4>', unsafe_allow_html=True)
    model = None
    model_input_shape = (224,224,3)

    model_path = None
    if uploaded_model_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp.write(uploaded_model_file.read())
        tmp.flush()
        model_path = tmp.name
    elif model_choice != "-- none --":
        model_path = os.path.join("models", model_choice)

    if model_path:
        try:
            with st.spinner("Loading model..."):
                model, model_input_shape = load_model_cached(model_path)
            st.success(f"✅ Model loaded: `{os.path.basename(model_path)}`")
            st.write(f"Input shape: {model_input_shape}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.warning("No model selected.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Prediction
# -----------------------
if st.button("🔮 Predict Image"):
    if pil_img is None:
        st.error("Upload an image first.")
    elif model is None:
        st.error("Load a model first.")
    else:
        prog = st.progress(0)
        status = st.empty()

        status.info("Preprocessing...")
        arr = preprocess_image_pil(pil_img, model_input_shape)
        prog.progress(40)

        status.info("Running inference...")
        preds = model.predict(arr)
        prog.progress(80)

        topk = top_k_from_pred(preds, k=top_k)
        prog.progress(100)
        status.success("Done ✅")

        st.subheader("📊 Prediction Results")
        best_idx, best_conf = topk[0]
        st.markdown(f"<div class='card'><h2>{best_idx}</h2><p class='small-muted'>Confidence: {best_conf*100:.2f}%</p></div>", unsafe_allow_html=True)

        df_topk = pd.DataFrame({
            "class": [str(i) for i, _ in topk],
            "confidence": [float(p) for _, p in topk]
        })
        st.bar_chart(df_topk.set_index("class"))
