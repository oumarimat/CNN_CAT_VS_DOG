import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Changer l'arrière-plan et ajouter des styles personnalisés
st.markdown("""
    <style>
    .stApp {
        background-color: #001f3d;  # Bleu marine pour le fond
    }
    
    .stTitle, .stHeader, .stSubheader {
        color: white !important;  # Texte en blanc pour une meilleure visibilité
        font-family: 'Arial', sans-serif;
    }

    h1 {
        color: white;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-top: 50px;
    }
    
    h3 {
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: 300;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #ff6347;  # Rouge corail pour les boutons
        color: white !important;
        border-radius: 12px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ff4500;  # Hover avec couleur orange
        transform: translateY(-4px);  # Effet de soulèvement
    }
    
    .stImage>img {
        border-radius: 12px;  # Coins arrondis pour les images
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .stFileUploader>div {
        background-color: #1e2a47;  # Fond foncé pour le bouton de téléchargement
        color: white !important;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stFileUploader>div:hover {
        background-color: #3b4a7f;  # Hover pour le bouton de téléchargement
    }

    .stText {
        color: white;  # Texte en blanc
        font-size: 18px;
    }

    .stSpinner {
        background-color: #ff6347;  # Spinner de couleur rouge corail
    }

    /* Ajouter du style pour le texte de la confiance */
    .stText {
        color: white !important;  # Texte blanc pour afficher la confiance
    }

    </style>
""", unsafe_allow_html=True)

# Charger le modèle entraîné
MODEL_PATH = "vgg16_best_model.keras"  # Chemin du modèle sauvegardé
model = load_model(MODEL_PATH)

# Définir les classes
CLASSES = {0: "Chat", 1: "Chien"}

# Fonction pour prédire l'image
def predict_image(img):
    img = img.resize((100, 100))  # Redimensionner l'image à 100x100
    img_array = image.img_to_array(img)  # Convertir en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension (batch size)
    img_array /= 255.0  # Normaliser les pixels entre 0 et 1
    prediction = model.predict(img_array)  # Faire la prédiction
    class_idx = 1 if prediction[0] > 0.5 else 0  # Seuil de 0.5
    confidence = prediction[0][0] if class_idx == 1 else 1 - prediction[0][0]  # Confiance
    return CLASSES[class_idx], confidence

# Interface utilisateur Streamlit
st.markdown("<h1>Classificateur de Chiens et Chats 🐾</h1>", unsafe_allow_html=True)
st.markdown("<h3>Téléchargez une image pour prédire si c'est un **chien** ou un **chat**</h3>", unsafe_allow_html=True)

# Chargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)

    # Charger l'image avec PIL
    img = Image.open(uploaded_file)

    # Faire la prédiction
    with st.spinner("Classification en cours..."):
        label, confidence = predict_image(img)

    # Afficher le résultat avec du design
    if label == "Chien":
        st.markdown("<div style='text-align: center; color: #ff6347; font-size: 22px; font-weight: bold;'> 🐶 **C'est un Chien!** 🐶</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; color: #4169e1; font-size: 22px; font-weight: bold;'> 🐱 **C'est un Chat!** 🐱</div>", unsafe_allow_html=True)

    # Afficher la confiance en texte blanc
    st.markdown(f"<div style='text-align: center; color: white; font-size: 18px;'>Confiance : **{confidence*100:.2f}%**</div>", unsafe_allow_html=True)
