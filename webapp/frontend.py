import streamlit as st
import requests
from PIL import Image
from io import BytesIO  

# Configuration de la page Streamlit simple
st.set_page_config(page_title="Classifieur Pissenlit vs Herbe", layout="centered")
st.title("Classifieur Pissenlit vs Herbe")
st.markdown("Dépose une image ci-dessous pour prédire si c'est un **pissenlit** ou de l'**herbe**.")

# Zone de drop d'image
uploaded_file = st.file_uploader("Dépose une image ici", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        image = Image.open(BytesIO(file_bytes))
        st.image(image, caption="Image chargée", use_column_width=True)
    except Exception as e:
        st.error(f"Image invalide : {e}")

    with st.spinner("Prédiction en cours..."):
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        try:
            response = requests.post("http://localhost:8000/predict", files=files)
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Prédiction : **{prediction.upper()}**")
            else:
                st.error(f"Erreur de l'API ({response.status_code}) : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
