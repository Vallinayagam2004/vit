import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from PIL import Image
import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm

# ====== CONFIGURATION ======
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

HUGGINGFACE_REPO = "valli2004/vit_brain"
BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

FILES = {
    "vit_brain_tumor_model.pth": "vit_brain_tumor_model.pth",
    "config.json": "config.json",
    "preprocessor_config.json": "preprocessor_config.json"
}

CLASS_LABELS = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}
# ===========================

def download_file(url, dest: Path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

@st.cache_data(show_spinner=False)
def download_model_files():
    for filename in FILES:
        dest = MODEL_DIR / filename
        if not dest.exists():
            url = f"{BASE_URL}/{filename}"
            try:
                download_file(url, dest)
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename} from Hugging Face: {e}")

def verify_binary_file(filepath: Path):
    with open(filepath, 'rb') as f:
        start = f.read(15)
        if start.startswith(b'<'):
            raise ValueError(f"File '{filepath}' looks like HTML, not a binary checkpoint.")

@st.cache_resource(show_spinner=False)
def load_model(pth_path: Path, config_path: Path):
    verify_binary_file(pth_path)
    verify_binary_file(config_path)

    config = ViTConfig.from_json_file(str(config_path))
    config.num_labels = len(CLASS_LABELS)
    model = ViTForImageClassification(config)

    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_processor(processor_path: Path):
    verify_binary_file(processor_path)
    return ViTImageProcessor.from_pretrained(str(processor_path))

def classify_image(model, processor, image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(), dim=-1).tolist()
    df = pd.DataFrame({
        "Class": [CLASS_LABELS[i] for i in range(len(probs))],
        "Confidence": [p for p in probs]
    })
    df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)
    return df

def main():
    st.set_page_config(
        page_title="Brain Tumor Classifier",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #2C3E50;">🧠 Brain Tumor Classification</h1>
            <p style="font-size:16px; color: #566573;">
                Upload an MRI scan and get predicted probabilities for glioma, meningioma, no tumor, and pituitary.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Instructions")
    st.sidebar.write("""
        1. Upload a brain MRI image (JPEG/PNG).  
        2. Wait for the model to process.  
        3. View confidence scores for each class.  
        4. Always consult medical professionals.
    """)

    with st.spinner("Downloading model files from Hugging Face…"):
        try:
            download_model_files()
        except Exception as e:
            st.error(f"Error downloading model files: {e}")
            return

    PTH_PATH = MODEL_DIR / "vit_brain_tumor_model.pth"
    CONFIG_PATH = MODEL_DIR / "config.json"
    PROCESSOR_PATH = MODEL_DIR / "preprocessor_config.json"

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Scan",
            width="stretch",
            clamp=True
        )

        with st.spinner("Loading model..."):
            try:
                model = load_model(PTH_PATH, CONFIG_PATH)
                processor = load_processor(PROCESSOR_PATH)
            except Exception as e:
                st.error(f"Failed to load model or processor: {e}")
                return

        with st.spinner("Classifying..."):
            results_df = classify_image(model, processor, image)

        st.markdown("## Confidence Scores")
        styled_df = (
            results_df.style
            .format({"Confidence": "{:.2%}"})
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("background-color", "#2C3E50"), ("color", "white"), ("text-align", "center")]},
                {"selector": "td", "props": [("padding", "8px"), ("font-size", "16px")]}
            ])
        )
        st.dataframe(styled_df, use_container_width=True)

        top_class = results_df.loc[0, "Class"]
        top_conf = results_df.loc[0, "Confidence"]
        st.markdown(
            f"""
            <div style="border:2px solid #27AE60; padding:15px; border-radius:5px; margin-top:20px;">
                <h3 style="color:#27AE60;">Top Prediction: {top_class.title()}</h3>
                <p style="font-size:18px;">Confidence: <strong>{top_conf:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <hr>
        <div style="text-align:center; color:#95A5A6; font-size:12px;">
            © 2025 Brain Tumor Classifier • For research use only
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
