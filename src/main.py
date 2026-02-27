import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import utils

# Configuration de la page
st.set_page_config(page_title="Générateur de Panorama", layout="wide")

st.title("📸 Générateur de Panorama (from scratch) - Yang Benjamin")
st.write("Ce projet implémente toute la pipeline de Computer Vision : Détection de coins (Harris), Feature Matching, RANSAC et Stitching.")

# Paramètres
st.sidebar.header("⚙️ Paramètres de l'algorithme")
seuil_harris = st.sidebar.slider("Seuil Harris", min_value=0.001, max_value=0.5, value=0.1, step=0.01)
lowe = st.sidebar.slider("Ratio de Lowe", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
seuil_abs = st.sidebar.number_input("Seuil Absolu (Distance)", value=40)
plot_lines = st.sidebar.checkbox("Afficher les lignes de Matching", value=True)

# Menu principal
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Gauche")
    file1 = st.file_uploader("Choisissez la première image", type=["jpg", "jpeg", "png"], key="img1")

with col2:
    st.subheader("Image Droite")
    file2 = st.file_uploader("Choisissez la deuxième image", type=["jpg", "jpeg", "png"], key="img2")

if file1 and file2:
    # Chargement et conversion en array numpy (en RGB et normalisé entre 0 et 255)
    # 1. Ouvrir avec PIL
    img_L_pil = Image.open(file1).convert('RGB')
    img_R_pil = Image.open(file2).convert('RGB')
    
    # 2. CORRECTION D'ORIENTATION
    img_L_pil = ImageOps.exif_transpose(img_L_pil)
    img_R_pil = ImageOps.exif_transpose(img_R_pil)
    
    # 3. Conversion en array numpy pour les fonctions
    img_L = np.array(img_L_pil, dtype=np.float32)
    img_R = np.array(img_R_pil, dtype=np.float32)
    
    st.write("---")
    if st.button("🚀 Lancer le traitement", type="primary"):
        with st.spinner('Calcul des homographies et fusion des images...'):
            try:
                # Lancement de la pipeline
                panorama, figs = utils.panorama_pipeline(
                    img_L, img_R, 
                    seuil_harris=seuil_harris, 
                    lowe=lowe, 
                    seuil_abs=seuil_abs, 
                    plot=plot_lines
                )
                
                # Affichage des graphes de matching
                if plot_lines and figs:
                    st.subheader("Analyse du Matching (RANSAC)")
                    for fig in figs:
                        st.pyplot(fig)
                        
                # Affichage du résultat final
                if panorama is not None:
                    st.subheader("🌟 Panorama Final")
                    
                    # On clip les valeurs pour l'affichage propre
                    panorama_disp = np.clip(panorama, 0, 255).astype(np.uint8)
                    st.image(panorama_disp, use_container_width=True)
                    
                    # Préparation pour le téléchargement
                    img_pil = Image.fromarray(panorama_disp)
                    buf = BytesIO()
                    img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Télécharger le Panorama",
                        data=byte_im,
                        file_name="mon_panorama.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.error("L'algorithme n'a pas trouvé assez de points de correspondances pour créer le panorama. Essayez d'augmenter le ratio de Lowe ou de baisser le seuil de Harris.")

            except Exception as e:
                st.error(f"Une erreur s'est produite lors du calcul : {e}")