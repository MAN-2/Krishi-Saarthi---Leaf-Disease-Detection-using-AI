import streamlit as st
from PIL import Image
from utils import predict_image, predict_top_k
import base64


def add_bg_image(image_path: str):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                    url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 12px;
    }}
    </style> 
    """
    st.markdown(css, unsafe_allow_html=True)


add_bg_image("back.jpg")  # background  file

#title
st.set_page_config(page_title="🌿 Krishi Saarthi ", layout="centered")
st.title("🌿 Krishi Sarthi : AI Powered Leaf Disease Detection")
st.markdown(
    "Upload a leaf image, and this app will predict the **plant disease** using a trained model."
)

#image upload
uploaded_file = st.file_uploader(
    "📷 Upload a leaf image (JPG, JPEG, PNG, or JFIF)", 
    type=["jpg", "jpeg", "png", "jfif"]
)

#Main
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            label, confidence = predict_image(image)
            top_k = predict_top_k(image, k=3)

        st.success(f"✅ **Predicted Disease:** `{label}` ({confidence * 100:.2f}%)")
        st.subheader("🔝 Top 3 Predictions")
        for cls, conf in top_k:
            st.write(f"- `{cls}`: **{conf * 100:.2f}%**")
else:
    st.info("👈 Please upload an image to get started.")
