import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = BASE_DIR / "assets" / "images"

@st.cache_data
def load_data(csv_path, sep=","):
    return pd.read_csv(csv_path, sep=sep)

st.set_page_config(
    page_title="Projet Rakuten - Classification Multimodale",
    layout="wide"
)

st.markdown("""
<style>
.red-title {
    color: #BF0000;
}
.center-title {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo6.svg",
    use_container_width=True
)

st.sidebar.title("Sommaire")

pages = [
    "Présentation",
    "Données",
    "Résultats",
    "Démo"
]

page = st.sidebar.radio("Aller vers :", pages)


@st.cache_data
def load_csv(path, sep=","):
    return pd.read_csv(path, sep=sep)


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

        image_path = IMG_DIR / "rakuten_image_entreprise.jpg"

        if image_path.exists():
            col1, col2, col3 = st.columns(3)
            with col2:
                st.image(
                    str(image_path),
                    caption="Siège social de Rakuten à Futakotamagawa, Tokyo",
                    use_container_width=True
                )
        else:
            st.warning("Image manquante : rakuten_image_entreprise.jpg")

        st.write("""
        **Rakuten, Inc. (Rakuten Kabushiki-gaisha)**  
        Société japonaise de services internet créée en février 1997.

        Depuis juin 2010, Rakuten a acquis PriceMinister, premier site de commerce électronique en France.
        """)

    with tab2:
        st.write("""
        **Objectif du projet**

        L’objectif est la classification multimodale à grande échelle des produits.

        Il s’agit de prédire le code type produit à partir de deux sources :

        - les données textuelles,
        - les images des produits.
        """)

        image_path = IMG_DIR / "objectif_projet.png"

        if image_path.exists():
            st.image(
                str(image_path),
                caption="Objectif du projet",
                use_container_width=True
            )
        else:
            st.warning("Image manquante : objectif_projet.png")


elif page == "Données":
    st.markdown(
        "<h1 class='red-title center-title'>Exploration des données</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    Le projet Rakuten repose sur des données textuelles, des images produits et une variable cible correspondant au type de produit.
    Dans cette version online, nous affichons les fichiers disponibles dans le dépôt GitHub.
    """)

    tab1, tab2, tab3 = st.tabs([
        "Prédictions",
        "Catégories",
        "Visualisations"
    ])

    # -----------------------------
    # TAB 1 : Prédictions
    # -----------------------------
    with tab1:
        st.subheader("Jeu de prédictions final")

        prediction_file = DATA_DIR / "df_prediction_final.csv"

        if prediction_file.exists():
            df_prediction = load_data(prediction_file)

            st.write("Aperçu des 20 premières lignes :")
            st.dataframe(df_prediction.head(20), use_container_width=True)

            st.write("Dimensions du fichier :")
            st.info(f"{df_prediction.shape[0]} lignes et {df_prediction.shape[1]} colonnes")

            numeric_columns = df_prediction.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if len(numeric_columns) > 0:
                selected_col = st.selectbox(
                    "Sélectionnez une colonne numérique à visualiser :",
                    numeric_columns
                )

                fig, ax = plt.subplots(figsize=(10, 4))
                df_prediction[selected_col].value_counts().head(20).plot(kind="bar", ax=ax)
                ax.set_title(f"Répartition de la colonne : {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Nombre d'occurrences")
                st.pyplot(fig)
            else:
                st.warning("Aucune colonne numérique disponible pour générer un graphique.")
        else:
            st.error("Fichier manquant : data/df_prediction_final.csv")

    # -----------------------------
    # TAB 2 : Catégories
    # -----------------------------
    with tab2:
        st.subheader("Catégories de produits")

        category_file = DATA_DIR / "categories_prdtypecode.csv"

        if category_file.exists():
            df_categories = pd.read_csv(category_file, sep=";")

            st.write("Liste des catégories disponibles :")
            st.dataframe(df_categories, use_container_width=True)

            st.write("Nombre de catégories :")
            st.info(f"{df_categories.shape[0]} catégories")

            if "code type" in df_categories.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                df_categories["code type"].astype(str).value_counts().plot(kind="bar", ax=ax)
                ax.set_title("Répartition des codes catégories")
                ax.set_xlabel("Code type")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)
        else:
            st.error("Fichier manquant : data/categories_prdtypecode.csv")

    # -----------------------------
    # TAB 3 : Visualisations
    # -----------------------------
    with tab3:
        st.subheader("Images et visualisations du projet")

        images_to_show = {
            "Objectif du projet": "objectif_projet.png",
            "Siège Rakuten": "rakuten_image_entreprise.jpg",
            "Synthèse des scores Deep Learning": "score_deep.png",
            "Catégories de produits": "categorie_produit.png",
            "Visualisation produits par catégorie": "visualisation_pdt_categorie.png",
            "Schéma X_train": "image_variable X_train_file.png",
            "Schéma Y_train": "image_variable Y_train_file.png",
        }

        selected_image_label = st.selectbox(
            "Sélectionnez une visualisation :",
            list(images_to_show.keys())
        )

        image_name = images_to_show[selected_image_label]
        image_path = IMG_DIR / image_name

        if image_path.exists():
            st.image(
                str(image_path),
                caption=selected_image_label,
                use_container_width=True
            )
        else:
            st.warning(f"Image manquante dans assets/images/ : {image_name}")

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

    image_path = ASSETS_DIR / "score_deep.png"

    if image_path.exists():
        st.image(str(image_path), caption="Synthèse des scores", use_container_width=True)
    else:
        st.info("Ajoute score_deep.png dans assets/images/ pour afficher le graphique.")


elif page == "Démo":
    st.markdown(
        "<h1 class='red-title center-title'>Démo de classification</h1>",
        unsafe_allow_html=True
    )

    st.warning("""
    La démo de prédiction sera ajoutée dans une deuxième étape.
    
    Pour l’instant, on vérifie d’abord que l’application se déploie correctement en ligne.
    """)

    category_file = DATA_DIR / "categories_prdtypecode.csv"

    if category_file.exists():
        df_cat = load_csv(category_file, sep=";")
        st.subheader("Catégories disponibles")
        st.dataframe(df_cat, use_container_width=True)
    else:
        st.info("Ajoute categories_prdtypecode.csv dans data/ pour afficher les catégories.")