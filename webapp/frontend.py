import streamlit as st
import requests
from PIL import Image
from io import BytesIO  
import os

# Configuration de la page Streamlit simple
st.set_page_config(page_title="Dandelion vs Grass Classifier", layout="centered")
st.title("Dandelion vs Grass Classifier")
st.markdown("Drop an image below to predict if it's a **dandelion** or **grass**.")

# Zone de drop d'image
uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        image = Image.open(BytesIO(file_bytes))
        st.image(image, caption="Loaded image", use_column_width=True)
    except Exception as e:
        st.error(f"Invalid image : {e}")

    with st.spinner("Predicting ..."):
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        try:
            API_URL = os.environ.get("API_URL", "http://api:8000/predict")
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Prediction : **{prediction.upper()}**")
            else:
                st.error(f"Erreur de l'API ({response.status_code}) : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
