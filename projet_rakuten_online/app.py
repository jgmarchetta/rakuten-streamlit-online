import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = BASE_DIR / "assets" / "images"

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
# FONCTIONS
# --------------------------------------------------
@st.cache_data(show_spinner="Chargement des données...")
def load_local_csv(path, sep=","):
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner="Chargement du fichier depuis Hugging Face...")
def load_remote_csv(filename, sep=","):
    url = HF_BASE_URL + filename
    return pd.read_csv(url, sep=sep)


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
    "Résultats",
    "Démo"
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
        nous avons pu travailler sur la classification de produits à grande échelle.

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

        L’objectif du projet est la classification multimodale à grande échelle des données de produits en codes de types de produits.

        Il s’agit de prédire le code type des produits à partir de données textuelles et images.
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
                show_image(
                    "histogramme_langues_X_train.png",
                    use_container_width=True
                )

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
                show_image(
                    "histogramme_langues_X_test.png",
                    use_container_width=True
                )

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

    tabs = st.tabs([
        "Scénario A",
        "Scénario B",
        "Scénario E",
        "Amélioration",
        "Optimisation"
    ])

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
        st.write("""
        Vectorisation des images par CNN, vectorisation du texte avec SpaCy sans traduction de texte.
        """)
        st.write("Les modèles :")
        add_model_expanders(images_scenario_A)

    with tabs[1]:
        st.header("Scénario B")
        st.write("""
        Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation,
        lemmatisation, application des stop-words, sans traduction de texte et réduction par TruncatedSVD.
        """)
        st.write("Les modèles :")
        add_model_expanders(images_scenario_B)

    with tabs[2]:
        st.header("Scénario E")
        st.write("""
        Même vectorisation que le scénario B, avec traduction du texte dans la langue majoritaire : le français.
        """)
        st.write("Les modèles :")
        add_model_expanders(images_scenario_E)

    with tabs[3]:
        st.header("Amélioration")
        st.write("Étape 1 : Recherche des meilleurs hyperparamètres.")

        with st.expander("**Amélioration B** Score F1-pondéré : 0.79"):
            st.write("Détails sur l'amélioration du scénario B.")
            show_image(
                "Ameb.png",
                caption="Amélioration B",
                width=600
            )

    with tabs[4]:
        st.header("Optimisation")
        st.write("Étape 2 : Validation croisée")
        st.write("Étape 3 : Rééchantillonnage et évaluation")

        with st.expander("**SMOTE** Score F1-pondéré : 0.80"):
            st.write("Détails sur SMOTE.")
            show_image(
                "SMOTE.png",
                caption="SMOTE",
                width=600
            )

        with st.expander("**RandomUnderSampler** Score F1-pondéré : 0.74"):
            st.write("Détails sur RandomUnderSampler.")
            show_image(
                "RUS.png",
                caption="RandomUnderSampler",
                width=600
            )

# --------------------------------------------------
# PAGE DEEP LEARNING
# --------------------------------------------------
elif page == "Deep Learning":

    st.markdown(
        "<h1 class='red-title center-title'>Deep Learning</h1>",
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "Benchmark",
        "Modèles",
        "Scores",
        "Interprétation"
    ])

    # -----------------------------------------
    # TAB 1 : Benchmark
    # -----------------------------------------
    with tabs[0]:
        st.header("Benchmark Rakuten")

        st.write("""
        Le challenge Rakuten utilise deux modèles :

        - **Images** : ResNet50 (pré-entraîné ImageNet)
        - **Texte** : CNN simplifié

        **Scores benchmark :**
        - Images : 0.55
        - Texte : 0.81
        """)

    # -----------------------------------------
    # TAB 2 : Modèles
    # -----------------------------------------
    with tabs[1]:
        st.header("Modèles testés")

        # -------- DNN --------
        with st.expander("DNN (Dense Neural Network) - F1 : 0.77"):
            st.write("Modèle dense classique sur features texte/image.")

            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_dnn.png", caption="Rapport classification")
            with col2:
                show_image("mtx_dnn.png", caption="Matrice de confusion")

        # -------- DistilBERT --------
        with st.expander("DistilBERT - F1 : 0.92"):
            st.write("Modèle NLP avancé basé sur Transformer.")

            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_d_bert.png")
            with col2:
                show_image("mtx_d_bert.png")

        # -------- EfficientNet + LSTM --------
        with st.expander("EfficientNetB0 + LSTM - F1 : 0.96 ⭐"):
            st.write("""
            Meilleur modèle :
            - Images → EfficientNet
            - Texte → LSTM
            """)

            col1, col2 = st.columns(2)
            with col1:
                show_image("rep_eff_lstm.png")
            with col2:
                show_image("mtx_eff_LSTM.png")

    # -----------------------------------------
    # TAB 3 : Scores
    # -----------------------------------------
    with tabs[2]:
        st.header("Synthèse des performances")

        st.write("""
        Le modèle **EfficientNetB0 + LSTM** est le meilleur :

        - F1-score : **0.96**
        - > Benchmark (0.81)
        - > Challenge (0.92)
        """)

        show_image("score_deep.png", caption="Comparaison des modèles")

    # -----------------------------------------
    # TAB 4 : Interprétation
    # -----------------------------------------
    with tabs[3]:
        st.header("Interprétation du modèle")

        sub_tabs = st.tabs(["Texte", "Images"])

        # ---- TEXTE ----
        with sub_tabs[0]:
            st.subheader("Analyse texte")

            show_image("txt_inter_1.png", caption="Importance des mots")

            col1, col2 = st.columns(2)
            with col1:
                show_image("txt_inter_2.png")
            with col2:
                show_image("txt_inter_3.png")

        # ---- IMAGES ----
        with sub_tabs[1]:
            st.subheader("Analyse images")

            show_image("img_inter_1.png", caption="Gradients")

            col1, col2 = st.columns(2)
            with col1:
                show_image("img_inter_2.png")
            with col2:
                show_image("img_inter_3.png")

            st.info("""
            Le modèle se concentre principalement sur les contours des objets.
            👉 Limite : difficile pour livres / DVD (formes similaires)
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

    Les techniques de réduction de dimension et le choix des algorithmes ont permis d’optimiser les résultats malgré la nature complexe et non structurée des données.

    Le modèle hybride alliant **EfficientNetB0 et LSTM** s’est avéré le plus adapté pour la classification des produits e-commerce de Rakuten.

    **En conclusion :**  
    Objectif atteint ! **Score final : F1-pondéré de 0.96**  
    Benchmark : 0.81 — Meilleur score Challenge : 0.92

    Très bonne prédiction des catégories avec le jeu d’entraînement fourni.
    """)

    tabs = st.tabs(["Limites du modèle", "Préconisations et améliorations"])

    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Limites du modèle**

            - La traduction du texte a peu d’impact sur les performances, car le français est la langue la plus présente.
            - Certaines catégories contiennent moins de textes en français.
            - La précision dépend fortement de la qualité de la description textuelle.
            - Pour les livres, magazines ou DVD, le modèle distingue difficilement les détails fins des images.
            - Le modèle est moins efficace lorsqu’une catégorie contient des produits très différents.
            - Le modèle est moins efficace lorsque plusieurs catégories sont visuellement très proches.
            """)

        with col2:
            img_tabs = st.tabs([
                "Français 1",
                "Français 2",
                "Français 3",
                "Précision",
                "Images",
                "Prédictions"
            ])

            with img_tabs[0]:
                show_image("pie_1.png", caption="Le français en rouge", use_container_width=True)

            with img_tabs[1]:
                show_image("pie_2.png", caption="Le français en rouge", use_container_width=True)

            with img_tabs[2]:
                show_image("pie_3.png", caption="Le français en rouge", use_container_width=True)

            with img_tabs[3]:
                show_image("exemple_text.png", caption="Exemple de désignation", use_container_width=True)
                st.write("""
                Dans cet exemple, la variable **designation** n’est pas assez détaillée.

                Le modèle peut donc avoir du mal à identifier correctement l’objet et à choisir la bonne catégorie,
                notamment lorsqu’il existe plusieurs catégories proches comme les livres, magazines ou DVD.
                """)

            with img_tabs[4]:
                show_image("img_inter_2.png", caption="Cartes de saillance", width=400)

            with img_tabs[5]:
                show_image(
                    "mtx_eff_LSTM_rem.png",
                    caption="Matrice de confusion EfficientNetB0-LSTM",
                    use_container_width=True
                )

    with tabs[1]:
        st.markdown("""
        **Préconisations et améliorations**

        - **Traduction en français :**  
          Appliquer la traduction uniquement sur les catégories où le français n’est pas majoritaire.

        - **Rééquilibrage des données :**  
          Rééquilibrer les données durant le pré-processing afin que chaque catégorie soit représentée plus équitablement.

        - **Augmentation des données :**  
          Utiliser des techniques d’augmentation d’images : rotation, recadrage, ajout de bruit, transformation de contraste.

        - **Nouveaux modèles Deep Learning :**  
          Explorer des modèles multimodaux plus récents, plus rapides et moins gourmands en ressources.

        - **Amélioration des descriptions produits :**  
          Enrichir les descriptions textuelles pour améliorer la discrimination entre catégories proches.
        """)

# --------------------------------------------------
# PAGE RÉSULTATS
# --------------------------------------------------
elif page == "Résultats":
    st.markdown(
        "<h1 class='red-title center-title'>Résultats des modèles</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    Résumé des performances obtenues :

    - Machine Learning : meilleur score autour de 0.80
    - DistilBERT : score F1 pondéré autour de 0.92
    - EfficientNetB0-LSTM : score F1 pondéré autour de 0.96
    """)

    show_image(
        "score_deep.png",
        caption="Synthèse des scores de F1 pondéré",
        use_container_width=True
    )

# --------------------------------------------------
# PAGE DÉMO
# --------------------------------------------------
elif page == "Démo":
    st.markdown(
        "<h1 class='red-title center-title'>Classification de produit</h1>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(['Jeu de données "Test" Rakuten', "Autres données"])

    with tab1:
        st.header('Jeu de données "Test" Rakuten')

        st.write("""
        Ce tableau contient les 20 premières lignes du DataFrame final et affiche les codes des catégories d'objets prédits
        du jeu de données "Test" Rakuten.
        """)

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
        st.header("Autres données")

        st.warning("""
        La prédiction par texte + image sera ajoutée dans une étape suivante.

        Les fichiers modèle `.keras` et `.pkl` sont trop lourds pour GitHub classique.
        Ils pourront être stockés sur Hugging Face dans une prochaine étape.
        """)

        st.write("""
        **Limitations de l'outil**

        Cet outil a été conçu pour répondre à la demande du Challenge Rakuten.
        Les catégories prédites sont celles définies dans le jeu de données Y_train.
        """)

        category_file = DATA_DIR / "categories_prdtypecode.csv"

        if category_file.exists():
            df_categorie = load_local_csv(category_file, sep=";")

            st.subheader("Rappel des catégories")
            st.data_editor(
                df_categorie,
                column_config={
                    "code type": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True,
            )
        else:
            st.error("Fichier manquant : data/categories_prdtypecode.csv")

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