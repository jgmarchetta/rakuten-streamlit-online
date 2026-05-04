import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = BASE_DIR / "assets" / "images"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

HF_BASE_URL = "https://huggingface.co/datasets/jgmarchetta/rakuten-data/resolve/main/"

st.set_page_config(
    page_title="Projet Rakuten - Classification Multimodale",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
body, h1, h2, h3, h4, h5, h6, p, div, span, li, a {
    font-family: Arial, sans-serif;
}
.red-title {
    color: #BF0000;
}
.center-title {
    text-align: center;
}
.small-title {
    font-size: 14px;
    font-weight: bold;
}
.reduced-spacing p {
    margin-bottom: 5px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FONCTIONS DATA
# --------------------------------------------------
@st.cache_data(show_spinner="Chargement des données locales...")
def load_local_csv(path, sep=","):
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner="Chargement depuis Hugging Face...")
def load_remote_csv(filename, sep=","):
    return pd.read_csv(HF_BASE_URL + filename, sep=sep)


def show_image(filename, caption=None, width=None, use_container_width=False):
    image_path = IMG_DIR / filename
    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            width=width,
            use_container_width=use_container_width
        )
    else:
        st.warning(f"Image manquante : {filename}")


def plot_missing_values_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df.isnull(),
        cmap=sns.color_palette(["#828282", "#BF0000"]),
        cbar=False,
        ax=ax
    )
    st.pyplot(fig)


def plot_nan_percentage(df, column_name):
    if column_name not in df.columns:
        st.warning(f"Colonne manquante : {column_name}")
        return

    nan_percentage = (df[column_name].isnull().sum() / len(df)) * 100
    non_nan_percentage = 100 - nan_percentage

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        ["Valeurs non-NaN", "NaN"],
        [non_nan_percentage, nan_percentage],
        color=["#004BAA", "#BF0000"]
    )
    ax.set_xlim(0, 100)

    for i, v in enumerate([non_nan_percentage, nan_percentage]):
        ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")

    st.pyplot(fig)


def plot_duplicate_percentage(df, column_name):
    if column_name not in df.columns:
        st.warning(f"Colonne manquante : {column_name}")
        return

    duplicate_counts = df[column_name].duplicated().value_counts()
    unique_count = duplicate_counts.get(False, 0)
    duplicate_count = duplicate_counts.get(True, 0)

    unique_percentage = (unique_count / len(df)) * 100
    duplicate_percentage = (duplicate_count / len(df)) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        ["Valeurs uniques", "Doublons"],
        [unique_percentage, duplicate_percentage],
        color=["#004BAA", "#BF0000"]
    )
    ax.set_xlim(0, 100)

    for i, v in enumerate([unique_percentage, duplicate_percentage]):
        ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")

    st.pyplot(fig)


# --------------------------------------------------
# FONCTIONS IA
# --------------------------------------------------
def download_from_hf(filename):
    local_path = MODEL_DIR / filename
    if not local_path.exists():
        with st.spinner(f"Téléchargement de {filename}..."):
            response = requests.get(HF_BASE_URL + filename)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
    return local_path


@st.cache_resource(show_spinner="Chargement du modèle IA...")
def load_demo_resources():
    tokenizer_path = download_from_hf("tokenizer.pkl")
    label_encoder_path = download_from_hf("label_encoder.pkl")
    model_path = download_from_hf("model_EfficientNetB0-LSTM.keras")

    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    with open(label_encoder_path, "rb") as handle:
        label_encoder = pickle.load(handle)

    model = tf.keras.models.load_model(model_path)
    categories = load_remote_csv("categories_prdtypecode.csv", sep=";")

    return tokenizer, label_encoder, model, categories


def preprocess_text(tokenizer, text, max_len=100):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_len)


def preprocess_uploaded_image(uploaded_file, image_size=128):
    temp_dir = BASE_DIR / "temp_dir"
    temp_dir.mkdir(exist_ok=True)

    image_path = temp_dir / uploaded_file.name

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = load_img(image_path, target_size=(image_size, image_size))
    img = img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return np.expand_dims(img, axis=0), image_path


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("""
<a href="https://challengedata.ens.fr/participants/challenges/35/" target="_blank">
    <img src="https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo6.svg" style="width: 100%;">
</a>
""", unsafe_allow_html=True)

st.sidebar.title("Sommaire")

pages = [
    "Présentation",
    "Données",
    "Pré-processing",
    "Machine Learning",
    "Deep Learning",
    "Conclusion",
    "Démo IA"
]

page = st.sidebar.radio("Aller vers :", pages)

# --------------------------------------------------
# PAGE PRÉSENTATION
# --------------------------------------------------
if page == "Présentation":
    st.markdown(
        "<h1 class='red-title center-title'>Projet Rakuten - Classification Multimodale</h1>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Contexte", "Objectif du projet"])

    with tab1:
        st.write("""
        Dans le cadre d'un challenge organisé par l'ENS et de notre formation Data Scientist au sein de DataScientest,
        nous avons travaillé sur la classification de produits à grande échelle.

        Le projet vise à prédire le type de chaque produit tel que défini dans le catalogue de Rakuten France.
        """)

        col1, col2, col3 = st.columns(3)
        with col2:
            show_image(
                "rakuten_image_entreprise.jpg",
                caption="Siège social de Rakuten à Futakotamagawa, Tokyo",
                use_container_width=True
            )

        st.write("""
        **Rakuten, Inc. (Rakuten Kabushiki-gaisha)**  
        Société japonaise de services internet créée en février 1997.

        Depuis juin 2010, Rakuten a acquis PriceMinister, premier site de commerce électronique en France.
        """)

    with tab2:
        st.write("""
        **Objectif du projet**

        L’objectif du projet est la classification multimodale à grande échelle des produits.

        Il s’agit de prédire le code type produit à partir de deux sources :

        - les données textuelles,
        - les images des produits.
        """)

        show_image(
            "objectif_projet.png",
            caption="Objectif du projet",
            use_container_width=True
        )

# --------------------------------------------------
# PAGE DONNÉES
# --------------------------------------------------
elif page == "Données":
    st.markdown(
        "<h1 class='red-title center-title'>Exploration des données</h1>",
        unsafe_allow_html=True
    )

    st.write("Le projet comporte 3 jeux de données textuelles et 1 jeu de données images.")

    col1, col2, col3 = st.columns(3)

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("""
        - X_train
        - X_test
        - Y_train
        - Fichier Images scindé en 2 fichiers image_train & image_test
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    selected_dataset = st.selectbox(
        "**Sélectionnez le jeu de données :**",
        ["X_train", "X_test", "Y_train", "Fichier Images Train"]
    )

    if selected_dataset == "X_train":
        st.write("Vous avez sélectionné le jeu de données X_train.")

        try:
            df_train = load_remote_csv("X_train_update.csv")
            st.success("Fichier X_train chargé depuis Hugging Face.")

            st.data_editor(
                df_train.head(),
                column_config={
                    "productid": st.column_config.NumberColumn(format="%d"),
                    "imageid": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
            )

            show_image(
                "image_variable X_train_file.png",
                caption="Schéma des variables du jeu de données X_train",
                width=1000
            )

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                plot_missing_values_heatmap(df_train)

            with col2:
                plot_nan_percentage(df_train, "description")

            with col3:
                plot_duplicate_percentage(df_train, "designation")

            with col4:
                show_image("histogramme_langues_X_train.png", use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du chargement de X_train : {e}")

    elif selected_dataset == "X_test":
        st.write("Vous avez sélectionné le jeu de données X_test.")

        try:
            df_test = load_remote_csv("X_test_update.csv")
            st.success("Fichier X_test chargé depuis Hugging Face.")

            st.data_editor(
                df_test.head(),
                column_config={
                    "productid": st.column_config.NumberColumn(format="%d"),
                    "imageid": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
            )

            show_image(
                "image variable X_test_file.png",
                caption="Schéma des variables du jeu de données X_test",
                width=1000
            )

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                plot_missing_values_heatmap(df_test)

            with col2:
                plot_nan_percentage(df_test, "description")

            with col3:
                plot_duplicate_percentage(df_test, "designation")

            with col4:
                show_image("histogramme_langues_X_test.png", use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du chargement de X_test : {e}")

    elif selected_dataset == "Y_train":
        st.write("Vous avez sélectionné le jeu de données Y_train.")

        try:
            df_target = load_remote_csv("Y_train_CVw08PX.csv")
            st.success("Fichier Y_train chargé depuis Hugging Face.")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(df_target.head(), use_container_width=True)

            with col2:
                show_image(
                    "image_variable Y_train_file.png",
                    caption="Schéma des variables du jeu de données Y_train",
                    use_container_width=True
                )

            col3, col4 = st.columns(2)

            with col3:
                plot_missing_values_heatmap(df_target)

            with col4:
                show_image(
                    "visualisation_pdt_categorie.png",
                    caption="Visualisation du nombre de produits par catégorie",
                    use_container_width=True
                )

            show_image(
                "categorie_produit.png",
                caption="27 catégories de produits",
                width=500
            )

        except Exception as e:
            st.error(f"Erreur lors du chargement de Y_train : {e}")

    elif selected_dataset == "Fichier Images Train":
        st.write("Vous avez sélectionné le fichier d'images du jeu de données images.")
        st.write("Le jeu de données images comporte 2 fichiers : image_train et image_test.")

        col1, col2 = st.columns(2)

        with col1:
            show_image(
                "visualisation_fichier_image.png",
                caption="Visualisation du fichier images_train",
                use_container_width=True
            )

        with col2:
            show_image(
                "dataframe_images_train.png",
                caption="DataFrame du fichier images_train",
                use_container_width=True
            )

        show_rapprochement = st.checkbox("**Rapprochement Textes-Images-Cible**")

        if show_rapprochement:
            st.markdown(
                "<h2 style='text-align: center; color: #004BAA;'>Rapprochement Textes - Images - Variable Cible</h2>",
                unsafe_allow_html=True
            )

            st.write("Voici un exemple de rapprochement entre textes, images et variable cible.")

            show_image(
                "rapprochement texte_image_cible.png",
                caption="Rapprochement texte/image & catégorie de produit",
                width=1000
            )

# --------------------------------------------------
# PAGE PRÉ-PROCESSING
# --------------------------------------------------
elif page == "Pré-processing":
    st.markdown(
        "<h1 class='red-title center-title'>Pré-processing</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    Pour ce projet, nous avons utilisé deux catégories de données non structurées : le texte et l’image.

    Ces données peuvent être vectorisées de plusieurs manières différentes.
    Nous avons donc testé plusieurs scénarios de pré-processing afin d’identifier la stratégie la plus performante.
    """)

    show_image(
        "scenario_preprocessing.png",
        caption="Représentation des scénarios de pré-processing",
        width=1000
    )

    show_scenarios = st.checkbox("**Afficher les scénarios**")

    if show_scenarios:
        st.markdown("""
        **Scénario A :** Vectorisation des images par CNN, vectorisation du texte avec SpaCy sans traduction de texte.

        **Scénario B :** Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et réduction par TruncatedSVD.

        **Scénario C :** Même vectorisation que le scénario B, sans application de réduction.

        **Scénario D :** Vectorisation des images par transformations successives : passage en gris, filtre Gaussien, filtre Laplacian, réduction de taille ; vectorisation du texte avec TF-IDF, tokenisation, lemmatisation, stop-words et TruncatedSVD.

        **Scénario E :** Même vectorisation que le scénario B, avec traduction du texte vers la langue majoritaire : le français.
        """)

# --------------------------------------------------
# PAGE MACHINE LEARNING
# --------------------------------------------------
elif page == "Machine Learning":
    st.markdown(
        "<h1 class='red-title center-title'>Machine Learning</h1>",
        unsafe_allow_html=True
    )

    tabs = st.tabs(["Scénario A", "Scénario B", "Scénario E", "Amélioration", "Optimisation"])

    def add_model_expanders(images):
        for model_name, infos in images.items():
            with st.expander(f"**{model_name}** Score F1-pondéré : {infos['score']}"):
                st.write(f"Détails sur le modèle {model_name}.")
                show_image(
                    infos["path"],
                    caption=f"{model_name} - score F1 pondéré : {infos['score']}",
                    use_container_width=True
                )

    images_scenario_A = {
        "XGboost": {"path": "A_XGboost.png", "score": 0.73},
        "SGD Classifier": {"path": "A_SGD Classifier.png", "score": 0.68},
        "Random Forest": {"path": "A_Random Forest.png", "score": 0.65},
        "Voting Classifier Soft": {"path": "A_VCS.png", "score": 0.73},
        "Voting Classifier Hard": {"path": "A_VCH.png", "score": 0.72},
        "Naive Bayes Gaussien": {"path": "A_NBG.png", "score": 0.46},
    }

    images_scenario_B = {
        "XGboost": {"path": "B_XGboost.png", "score": 0.77},
        "SGD Classifier": {"path": "B_SGD Classifier.png", "score": 0.62},
        "Random Forest": {"path": "B_Random Forest.png", "score": 0.76},
        "Voting Classifier Soft": {"path": "B_VCS.png", "score": 0.77},
        "Voting Classifier Hard": {"path": "B_VCH.png", "score": 0.76},
        "Naive Bayes Gaussien": {"path": "B_NBG.png", "score": 0.51},
    }

    images_scenario_E = {
        "XGboost": {"path": "E_XGboost.png", "score": 0.76},
        "SGD Classifier": {"path": "E_SGD Classifier.png", "score": 0.58},
        "Random Forest": {"path": "E_Random Forest.png", "score": 0.75},
        "Voting Classifier Soft": {"path": "E_VCS.png", "score": 0.75},
        "Voting Classifier Hard": {"path": "E_VCH.png", "score": 0.74},
        "Naive Bayes Gaussien": {"path": "E_NBG.png", "score": 0.50},
    }

    with tabs[0]:
        st.header("Scénario A")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec SpaCy sans traduction de texte.")
        add_model_expanders(images_scenario_A)

    with tabs[1]:
        st.header("Scénario B")
        st.write("""
        Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation,
        lemmatisation, application des stop-words, sans traduction de texte et réduction par TruncatedSVD.
        """)
        add_model_expanders(images_scenario_B)

    with tabs[2]:
        st.header("Scénario E")
        st.write("Même vectorisation que le scénario B, avec traduction du texte dans la langue majoritaire : le français.")
        add_model_expanders(images_scenario_E)

    with tabs[3]:
        st.header("Amélioration")
        st.write("Étape 1 : Recherche des meilleurs hyperparamètres.")

        with st.expander("**Amélioration B** Score F1-pondéré : 0.79"):
            show_image("Ameb.png", caption="Amélioration B", width=600)

    with tabs[4]:
        st.header("Optimisation")
        st.write("Étape 2 : Validation croisée")
        st.write("Étape 3 : Rééchantillonnage et évaluation")

        with st.expander("**SMOTE** Score F1-pondéré : 0.80"):
            show_image("SMOTE.png", caption="SMOTE", width=600)

        with st.expander("**RandomUnderSampler** Score F1-pondéré : 0.74"):
            show_image("RUS.png", caption="RandomUnderSampler", width=600)

# --------------------------------------------------
# PAGE DEEP LEARNING
# --------------------------------------------------
elif page == "Deep Learning":
    st.markdown(
        "<h1 class='red-title center-title'>Deep Learning</h1>",
        unsafe_allow_html=True
    )

    tabs = st.tabs(["Benchmark", "Modèles", "Scores", "Interprétation"])

    with tabs[0]:
        st.header("Benchmark Rakuten")
        st.write("""
        Le challenge Rakuten utilise deux modèles :

        - **Images** : ResNet50 pré-entraîné sur ImageNet
        - **Texte** : CNN simplifié

        **Scores benchmark :**
        - Images : 0.55
        - Texte : 0.81
        """)

    with tabs[1]:
        st.header("Modèles testés")

        with st.expander("DNN - F1 : 0.77"):
            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_dnn.png", caption="Rapport classification")
            with col2:
                show_image("mtx_dnn.png", caption="Matrice de confusion")

        with st.expander("DistilBERT - F1 : 0.92"):
            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_d_bert.png")
            with col2:
                show_image("mtx_d_bert.png")

        with st.expander("EfficientNetB0 + LSTM - F1 : 0.96 ⭐"):
            st.write("""
            Meilleur modèle :
            - Images : EfficientNetB0
            - Texte : LSTM
            """)
            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_eff_lstm.png")
            with col2:
                show_image("mtx_eff_LSTM.png")

    with tabs[2]:
        st.header("Synthèse des performances")
        st.write("""
        Le modèle **EfficientNetB0 + LSTM** est le meilleur :

        - F1-score : **0.96**
        - supérieur au benchmark : 0.81
        - supérieur au meilleur score Challenge : 0.92
        """)
        show_image("score_deep.png", caption="Comparaison des modèles")

    with tabs[3]:
        st.header("Interprétation du modèle")

        sub_tabs = st.tabs(["Texte", "Images"])

        with sub_tabs[0]:
            show_image("txt_inter_1.png", caption="Importance des mots")
            col1, col2 = st.columns(2)
            with col1:
                show_image("txt_inter_2.png")
            with col2:
                show_image("txt_inter_3.png")

        with sub_tabs[1]:
            show_image("img_inter_1.png", caption="Gradients")
            col1, col2 = st.columns(2)
            with col1:
                show_image("img_inter_2.png")
            with col2:
                show_image("img_inter_3.png")

            st.info("""
            Le modèle se concentre principalement sur les contours des objets.
            Limite : les livres, DVD et magazines ont souvent des formes similaires.
            """)

# --------------------------------------------------
# PAGE CONCLUSION
# --------------------------------------------------
elif page == "Conclusion":
    st.markdown(
        "<h1 class='red-title center-title'>Conclusion</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    Les choix effectués tout au long du projet ont été guidés par des objectifs de performance et de robustesse.

    Le modèle hybride alliant **EfficientNetB0 et LSTM** s’est avéré le plus adapté pour la classification des produits e-commerce de Rakuten.

    **Objectif atteint : score final F1-pondéré de 0.96**  
    Benchmark : 0.81 — Meilleur score Challenge : 0.92
    """)

    tabs = st.tabs(["Limites du modèle", "Préconisations et améliorations"])

    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Limites du modèle**

            - La traduction du texte a peu d’impact sur les performances.
            - La précision dépend fortement de la qualité de la description textuelle.
            - Les catégories visuellement proches restent difficiles à distinguer.
            - Les livres, magazines et DVD ont des formes similaires.
            """)

        with col2:
            img_tabs = st.tabs(["Français 1", "Français 2", "Français 3", "Précision", "Images", "Prédictions"])

            with img_tabs[0]:
                show_image("pie_1.png", caption="Le français en rouge", use_container_width=True)
            with img_tabs[1]:
                show_image("pie_2.png", caption="Le français en rouge", use_container_width=True)
            with img_tabs[2]:
                show_image("pie_3.png", caption="Le français en rouge", use_container_width=True)
            with img_tabs[3]:
                show_image("exemple_text.png", caption="Exemple de désignation", use_container_width=True)
            with img_tabs[4]:
                show_image("img_inter_2.png", caption="Cartes de saillance", width=400)
            with img_tabs[5]:
                show_image("mtx_eff_LSTM_rem.png", caption="Matrice de confusion EfficientNetB0-LSTM", use_container_width=True)

    with tabs[1]:
        st.markdown("""
        **Préconisations et améliorations**

        - Traduire uniquement les catégories où le français n’est pas majoritaire.
        - Rééquilibrer les catégories.
        - Augmenter les données image.
        - Tester des modèles multimodaux plus récents.
        - Enrichir les descriptions produits.
        """)

# --------------------------------------------------
# PAGE DÉMO IA
# --------------------------------------------------
elif page == "Démo IA":
    st.markdown(
        "<h1 class='red-title center-title'>Démo IA - Classification de produit</h1>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(['Jeu de données "Test" Rakuten', "Prédiction IA"])

    with tab1:
        st.header('Jeu de données "Test" Rakuten')

        prediction_file = DATA_DIR / "df_prediction_final.csv"

        if prediction_file.exists():
            df_prediction_final = load_local_csv(prediction_file)

            st.data_editor(
                df_prediction_final.head(20),
                column_config={
                    "Code catégorie prédite": st.column_config.NumberColumn(format="%d"),
                    "index": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
            )
        else:
            st.error("Fichier manquant : data/df_prediction_final.csv")

    with tab2:
        st.header("Prédiction sur un nouveau produit")

        st.write("""
        Entrez une description produit et ajoutez une image.
        Le modèle prédit ensuite la catégorie Rakuten la plus probable.
        """)

        try:
            tokenizer, label_encoder, model, categories = load_demo_resources()
            st.success("Modèle IA chargé avec succès.")

            description_text = st.text_input("Entrez la description du produit :")

            uploaded_file = st.file_uploader(
                "Choisissez une image",
                type=["jpg", "jpeg", "png"]
            )

            if st.button("Prédire"):
                if description_text and uploaded_file:
                    text_data = preprocess_text(tokenizer, description_text)
                    image_data, image_path = preprocess_uploaded_image(uploaded_file)

                    predictions = model.predict([text_data, image_data])
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                    category_row = categories[categories["code type"] == predicted_label]
                    category_name = (
                        category_row["désignation de catégorie"].values[0]
                        if not category_row.empty
                        else "Inconnue"
                    )

                    confidence = float(np.max(predictions)) * 100

                    img = Image.open(image_path)

                    st.image(
                        img,
                        caption=f"Prédiction : {predicted_label} - {category_name}",
                        width=300
                    )

                    st.success(f"Catégorie prédite : {predicted_label} - {category_name}")
                    st.info(f"Confiance du modèle : {confidence:.2f} %")

                else:
                    st.warning("Veuillez entrer une description et télécharger une image.")

        except Exception as e:
            st.error(f"Erreur lors du chargement ou de la prédiction : {e}")

        st.divider()

        st.subheader("Rappel des catégories")

        try:
            df_categorie = load_remote_csv("categories_prdtypecode.csv", sep=";")
            st.data_editor(
                df_categorie,
                column_config={
                    "code type": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
            )
        except Exception as e:
            st.warning(f"Impossible de charger les catégories : {e}")

# --------------------------------------------------
# BAS DE SIDEBAR
# --------------------------------------------------
for _ in range(2):
    st.sidebar.text("")

st.sidebar.markdown("""
<div class="reduced-spacing">
    <p class="small-title">Auteurs:</p>
    <p><a href="https://www.linkedin.com/in/labordecaroline/" target="_blank">Caroline LABORDE</a></p>
    <p><a href="https://www.linkedin.com/in/soulaiman-cheddadi/" target="_blank">Soulaiman CHEDDADI</a></p>
    <p><a href="https://www.linkedin.com/in/jean-gabrielmarchetta/" target="_blank">Jean-Gabriel MARCHETTA</a></p>
    <p class="small-title">Mentor:</p>
    <p>Eliott DOUIEB</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.text("")
st.sidebar.text("")

st.sidebar.markdown("""
<a href="https://datascientest.com/" target="_blank">
    <img src="https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png" style="width: 100%;">
</a>
""", unsafe_allow_html=True)

st.sidebar.text("Datascientist - Bootcamp mars 2024")