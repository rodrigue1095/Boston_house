
# ============================================================
# APPLICATION COMPLETE DE DATA SCIENCE ET MACHINE LEARNING
# Analyse de donnees et prediction du prix des maisons
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. CONFIGURATION DE L'APPLICATION
# ============================================================

st.set_page_config(
    page_title="Data Science - Prix des maisons",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Application de Data Science et Machine Learning")
st.markdown(
    """
    **Analyse exploratoire, visualisation, entraînement des modèles
    et prédiction interactive du prix des maisons.**
    """
)

st.divider()


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def load_data(uploaded_file):
    """Charge un fichier CSV importé ou le fichier local."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=None, engine="python")

    try:
        return pd.read_csv(
            "HousingData.csv",
            sep=None,
            engine="python"
        )
    except FileNotFoundError:
        return None


def create_one_hot_encoder():
    """Compatible avec plusieurs versions de scikit-learn."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def prepare_features(data, target):
    """Sépare la cible des variables explicatives."""
    if target not in data.columns:
        raise ValueError("La variable cible est introuvable.")

    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # La cible doit être numérique pour une régression.
    y = pd.to_numeric(df[target], errors="coerce")

    # Supprimer les lignes dont la cible est manquante.
    valid_rows = y.notna()
    X = df.loc[valid_rows].drop(columns=[target])
    y = y.loc[valid_rows]

    # Supprimer les colonnes totalement vides.
    X = X.dropna(axis=1, how="all")

    # Détecter les colonnes numériques qui sont en réalité
    # stockées sous forme de texte.
    for column in X.columns:
        if X[column].dtype == "object":
            converted = pd.to_numeric(X[column], errors="coerce")
            if converted.notna().mean() >= 0.90:
                X[column] = converted

    # Écarter les variables constantes.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        raise ValueError(
            "Aucune variable explicative exploitable n'a été trouvée."
        )

    if len(X) < 5:
        raise ValueError(
            "Le jeu de données doit contenir au moins 5 lignes valides."
        )

    return X, y


def build_pipeline(X, model_name, random_state=7):
    """Construit le prétraitement et le modèle."""
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", create_one_hot_encoder())
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline,
             categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "Régression linéaire": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Arbre de décision": DecisionTreeRegressor(
            random_state=random_state
        ),
        "K plus proches voisins": KNeighborsRegressor(
            n_neighbors=5
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state
        ),
        "AdaBoost": AdaBoostRegressor(
            random_state=random_state,
            n_estimators=100
        ),
        "SVR": SVR()
    }

    if model_name not in models:
        raise ValueError("Modèle non reconnu.")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_name])
    ])


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier CSV",
    type=["csv"]
)

data = load_data(uploaded_file)

if data is None:
    st.warning(
        "Aucun fichier de données n'a été trouvé. "
        "Importez un fichier CSV dans la barre latérale, "
        "ou placez HousingData.csv dans le même dossier que app.py."
    )
    st.code(
        "pip install streamlit pandas numpy matplotlib scikit-learn"
    )
    st.stop()

if data.empty:
    st.error("Le fichier importé ne contient aucune donnée.")
    st.stop()

data.columns = data.columns.astype(str).str.strip()

# Supprimer les colonnes dont le nom est vide ou dupliqué.
if any(not column for column in data.columns):
    st.error("Le fichier contient un nom de colonne vide.")
    st.stop()

if data.columns.duplicated().any():
    st.error(
        "Le fichier contient des noms de colonnes dupliqués. "
        "Renommez-les avant de continuer."
    )
    st.stop()

st.sidebar.success("Fichier chargé avec succès.")


# ============================================================
# 4. NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Choisir une page",
    [
        "Accueil",
        "Exploration des données",
        "Visualisations",
        "Corrélations",
        "Machine Learning",
        "Prédiction du prix"
    ]
)


# ============================================================
# 5. PAGE ACCUEIL
# ============================================================

if page == "Accueil":

    st.header("📊 Tableau de bord")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre de lignes", f"{data.shape[0]:,}")

    with col2:
        st.metric("Nombre de colonnes", f"{data.shape[1]:,}")

    with col3:
        st.metric(
            "Valeurs manquantes",
            f"{int(data.isna().sum().sum()):,}"
        )

    st.subheader("Aperçu du jeu de données")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Informations générales")

    info_df = pd.DataFrame({
        "Colonne": data.columns,
        "Type": [str(data[c].dtype) for c in data.columns],
        "Valeurs manquantes": [
            int(data[c].isna().sum()) for c in data.columns
        ],
        "Valeurs distinctes": [
            int(data[c].nunique(dropna=True)) for c in data.columns
        ]
    })

    st.dataframe(info_df, use_container_width=True)

    st.info(
        "Utilisez le menu à gauche pour explorer les données, "
        "visualiser les variables, entraîner un modèle et estimer "
        "le prix d'une maison."
    )


# ============================================================
# 6. EXPLORATION DES DONNEES
# ============================================================

elif page == "Exploration des données":

    st.header("🔎 Exploration des données")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Aperçu",
        "Statistiques",
        "Valeurs manquantes",
        "Types de données"
    ])

    with tab1:
        st.subheader("Premières lignes")
        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Dernières lignes")
        st.dataframe(data.tail(10), use_container_width=True)

    with tab2:
        st.subheader("Statistiques descriptives")
        st.dataframe(
            data.describe(include="all").T,
            use_container_width=True
        )

    with tab3:
        missing = data.isna().sum()
        missing_df = pd.DataFrame({
            "Colonne": missing.index,
            "Valeurs manquantes": missing.values,
            "Pourcentage (%)": (
                missing.values / len(data) * 100
            ).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        missing_only = missing_df[
            missing_df["Valeurs manquantes"] > 0
        ]

        if not missing_only.empty:
            st.bar_chart(
                missing_only.set_index("Colonne")[
                    "Valeurs manquantes"
                ]
            )
        else:
            st.success("Aucune valeur manquante détectée.")

    with tab4:
        types_df = pd.DataFrame({
            "Colonne": data.columns,
            "Type": [str(data[c].dtype) for c in data.columns],
            "Valeurs distinctes": [
                data[c].nunique(dropna=True) for c in data.columns
            ]
        })
        st.dataframe(types_df, use_container_width=True)


# ============================================================
# 7. VISUALISATIONS
# ============================================================

elif page == "Visualisations":

    st.header("📈 Visualisation des données")

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        st.warning(
            "Aucune variable numérique n'est disponible "
            "pour ces visualisations."
        )
    else:
        plot_type = st.selectbox(
            "Type de graphique",
            [
                "Histogramme",
                "Boîte à moustaches",
                "Nuage de points",
                "Matrice de dispersion"
            ]
        )

        if plot_type == "Histogramme":
            selected_columns = st.multiselect(
                "Variables à représenter",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )

            if selected_columns:
                fig, ax = plt.subplots(figsize=(10, 5))

                for column in selected_columns:
                    values = pd.to_numeric(
                        data[column], errors="coerce"
                    ).dropna()

                    ax.hist(
                        values,
                        bins=25,
                        alpha=0.45,
                        label=column
                    )

                ax.set_title("Distribution des variables")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Fréquence")
                ax.legend()
                ax.grid(alpha=0.2)

                st.pyplot(fig)
                plt.close(fig)

        elif plot_type == "Boîte à moustaches":
            column = st.selectbox(
                "Choisir une variable",
                numeric_columns
            )

            values = pd.to_numeric(
                data[column], errors="coerce"
            ).dropna()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(values, vert=False)
            ax.set_title(f"Boîte à moustaches : {column}")
            ax.set_xlabel(column)
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        elif plot_type == "Nuage de points":
            if len(numeric_columns) < 2:
                st.warning(
                    "Il faut au moins deux variables numériques."
                )
            else:
                x_col = st.selectbox(
                    "Variable sur l'axe X",
                    numeric_columns,
                    index=0
                )

                y_col = st.selectbox(
                    "Variable sur l'axe Y",
                    numeric_columns,
                    index=min(1, len(numeric_columns) - 1)
                )

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(
                    data[x_col],
                    data[y_col],
                    alpha=0.6
                )
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} en fonction de {x_col}")
                ax.grid(alpha=0.2)

                st.pyplot(fig)
                plt.close(fig)

        else:
            selected_columns = st.multiselect(
                "Variables de la matrice",
                numeric_columns,
                default=numeric_columns[:min(4, len(numeric_columns))]
            )

            if len(selected_columns) >= 2:
                from pandas.plotting import scatter_matrix

                fig = plt.figure(figsize=(10, 8))
                scatter_matrix(
                    data[selected_columns],
                    figsize=(10, 8),
                    diagonal="hist",
                    alpha=0.6
                )
                plt.suptitle("Matrice de dispersion")
                st.pyplot(plt.gcf())
                plt.close("all")
            else:
                st.info("Sélectionnez au moins deux variables.")


# ============================================================
# 8. CORRELATIONS
# ============================================================

elif page == "Corrélations":

    st.header("🔗 Analyse des corrélations")

    numeric_data = data.select_dtypes(
        include=np.number
    )

    if numeric_data.shape[1] < 2:
        st.warning(
            "Il faut au moins deux variables numériques "
            "pour calculer une matrice de corrélation."
        )
    else:
        corr = numeric_data.corr()

        st.subheader("Matrice de corrélation")
        st.dataframe(corr.round(3), use_container_width=True)

        fig, ax = plt.subplots(
            figsize=(
                max(8, len(corr.columns) * 0.7),
                max(6, len(corr.columns) * 0.6)
            )
        )

        image = ax.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )
        ax.set_yticklabels(corr.columns)

        fig.colorbar(image, ax=ax, label="Corrélation")
        ax.set_title("Carte thermique des corrélations")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Corrélations les plus fortes")

        pairs = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        ).stack().reset_index()

        if not pairs.empty:
            pairs.columns = ["Variable 1", "Variable 2", "Corrélation"]
            pairs["Corrélation absolue"] = pairs[
                "Corrélation"
            ].abs()

            pairs = pairs.sort_values(
                "Corrélation absolue",
                ascending=False
            )

            st.dataframe(
                pairs.drop(columns=["Corrélation absolue"]).head(15),
                use_container_width=True
            )


# ============================================================
# 9. MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.header("🤖 Entraînement et évaluation des modèles")

    st.write(
        "Choisissez la variable cible, les modèles à comparer "
        "et la proportion des données réservée au test."
    )

    # Les colonnes numériques sont proposées en premier.
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    target_options = numeric_columns if numeric_columns else list(data.columns)

    default_target = (
        "MEDV" if "MEDV" in target_options
        else target_options[-1]
    )

    target = st.selectbox(
        "Variable cible à prédire",
        target_options,
        index=target_options.index(default_target)
    )

    models_to_train = st.multiselect(
        "Modèles à comparer",
        [
            "Régression linéaire",
            "Ridge",
            "Arbre de décision",
            "K plus proches voisins",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "SVR"
        ],
        default=[
            "Régression linéaire",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    test_size = st.slider(
        "Proportion des données pour le test (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    random_state = st.number_input(
        "Graine aléatoire",
        min_value=0,
        max_value=99999,
        value=7,
        step=1
    )

    use_cv = st.checkbox(
        "Activer la validation croisée",
        value=False
    )

    use_grid_search = st.checkbox(
        "Optimiser Gradient Boosting avec GridSearchCV",
        value=False
    )

    if use_grid_search and "Gradient Boosting" not in models_to_train:
        st.warning(
            "Ajoutez Gradient Boosting à la liste des modèles "
            "pour utiliser cette optimisation."
        )

    if st.button("🚀 Entraîner les modèles", type="primary"):

        if not models_to_train:
            st.error("Sélectionnez au moins un modèle.")
            st.stop()

        try:
            X, y = prepare_features(data, target)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=int(random_state)
            )

            results = []
            trained_models = {}
            test_predictions = {}

            progress = st.progress(0)
            status = st.empty()

            for index, model_name in enumerate(models_to_train):

                status.write(f"Entraînement : {model_name}")

                pipeline = build_pipeline(
                    X_train,
                    model_name,
                    random_state=int(random_state)
                )

                if (
                    use_grid_search
                    and model_name == "Gradient Boosting"
                ):
                    param_grid = {
                        "model__n_estimators": [50, 100, 200],
                        "model__learning_rate": [0.03, 0.05, 0.1],
                        "model__max_depth": [2, 3]
                    }

                    search = GridSearchCV(
                        pipeline,
                        param_grid,
                        cv=3,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1
                    )

                    search.fit(X_train, y_train)
                    pipeline = search.best_estimator_

                    st.write(
                        "Meilleurs paramètres :",
                        search.best_params_
                    )

                else:
                    pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                mse = mean_squared_error(y_test, predictions)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                cv_rmse = np.nan

                if use_cv:
                    n_splits = min(5, len(X_train))

                    if n_splits >= 2:
                        cv = KFold(
                            n_splits=n_splits,
                            shuffle=True,
                            random_state=int(random_state)
                        )

                        scores = cross_val_score(
                            pipeline,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1
                        )

                        cv_rmse = float(
                            np.sqrt(-scores).mean()
                        )

                results.append({
                    "Modèle": model_name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R²": r2,
                    "RMSE validation croisée": cv_rmse
                })

                trained_models[model_name] = pipeline
                test_predictions[model_name] = predictions

                progress.progress(
                    (index + 1) / len(models_to_train)
                )

            status.success("Entraînement terminé.")

            results_df = pd.DataFrame(results).sort_values(
                "RMSE",
                ascending=True
            )

            st.session_state["ml_results"] = results_df
            st.session_state["ml_models"] = trained_models
            st.session_state["ml_target"] = target
            st.session_state["ml_columns"] = X.columns.tolist()
            st.session_state["ml_X_reference"] = X.copy()
            st.session_state["ml_test_X"] = X_test.copy()
            st.session_state["ml_test_y"] = y_test.copy()
            st.session_state["ml_test_predictions"] = test_predictions
            st.session_state["ml_train_y"] = y_train.copy()

            st.subheader("Résultats des modèles")

            st.dataframe(
                results_df.round(4),
                use_container_width=True
            )

            best_model_name = results_df.iloc[0]["Modèle"]
            best_rmse = results_df.iloc[0]["RMSE"]
            best_r2 = results_df.iloc[0]["R²"]

            col1, col2, col3 = st.columns(3)

            col1.metric("Meilleur modèle", best_model_name)
            col2.metric("RMSE du meilleur modèle", f"{best_rmse:.4f}")
            col3.metric("R² du meilleur modèle", f"{best_r2:.4f}")

            st.subheader("Comparaison des RMSE")

            chart_data = results_df.set_index("Modèle")[["RMSE"]]
            st.bar_chart(chart_data)

            csv_results = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Télécharger les résultats CSV",
                data=csv_results,
                file_name="resultats_machine_learning.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Erreur pendant l'entraînement : {error}")

    # Les résultats restent visibles après un changement de page
    # ou un nouveau rendu Streamlit.
    if "ml_results" in st.session_state:
        st.divider()
        st.subheader("Derniers résultats enregistrés")
        st.dataframe(
            st.session_state["ml_results"].round(4),
            use_container_width=True
        )


# ============================================================
# 10. PREDICTION INTERACTIVE DU PRIX
# ============================================================

elif page == "Prédiction du prix":

    st.header("🏡 Prédire le prix d'une maison")

    st.write(
        """
        Après avoir entraîné un modèle dans la page **Machine Learning**,
        choisissez ce modèle puis renseignez les caractéristiques de la
        maison. L'application utilisera le modèle entraîné pour produire
        une estimation.
        """
    )

    if (
        "ml_models" not in st.session_state
        or not st.session_state["ml_models"]
    ):
        st.warning(
            "Aucun modèle entraîné n'est disponible. "
            "Ouvrez la page Machine Learning, choisissez la cible "
            "et entraînez au moins un modèle avant de revenir ici."
        )

    else:
        trained_models = st.session_state["ml_models"]
        X_reference = st.session_state["ml_X_reference"]
        target = st.session_state["ml_target"]

        model_names = list(trained_models.keys())

        default_model = (
            st.session_state["ml_results"].iloc[0]["Modèle"]
            if "ml_results" in st.session_state
            else model_names[0]
        )

        default_index = (
            model_names.index(default_model)
            if default_model in model_names else 0
        )

        chosen_model = st.selectbox(
            "Modèle de prédiction",
            model_names,
            index=default_index
        )

        st.subheader("Caractéristiques de la maison")

        st.caption(
            "Les valeurs numériques sont initialisées avec la médiane "
            "des données utilisées pour l'entraînement. Les variables "
            "catégorielles sont initialisées avec leur valeur la plus fréquente."
        )

        with st.form("house_prediction_form"):

            input_values = {}

            numeric_features = X_reference.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = [
                col for col in X_reference.columns
                if col not in numeric_features
            ]

            if numeric_features:
                st.markdown("#### Variables numériques")

                numeric_cols = st.columns(2)

                for i, column in enumerate(numeric_features):
                    series = pd.to_numeric(
                        X_reference[column], errors="coerce"
                    )

                    median_value = series.median()

                    if pd.isna(median_value):
                        median_value = 0.0

                    minimum = series.min()
                    maximum = series.max()

                    if pd.isna(minimum):
                        minimum = float(median_value)

                    if pd.isna(maximum):
                        maximum = float(median_value)

                    with numeric_cols[i % 2]:
                        input_values[column] = st.number_input(
                            label=column,
                            value=float(median_value),
                            min_value=None,
                            max_value=None,
                            help=(
                                f"Valeur observée dans les données : "
                                f"{minimum:.4g} à {maximum:.4g}"
                            ),
                            key=f"numeric_{column}"
                        )

            if categorical_features:
                st.markdown("#### Variables catégorielles")

                categorical_cols = st.columns(2)

                for i, column in enumerate(categorical_features):
                    series = X_reference[column].dropna()

                    if series.empty:
                        choices = ["Inconnu"]
                    else:
                        choices = series.astype(str).unique().tolist()

                    mode_values = series.astype(str).mode()
                    default_value = (
                        mode_values.iloc[0]
                        if not mode_values.empty
                        else choices[0]
                    )

                    default_index = (
                        choices.index(default_value)
                        if default_value in choices else 0
                    )

                    with categorical_cols[i % 2]:
                        input_values[column] = st.selectbox(
                            column,
                            choices,
                            index=default_index,
                            key=f"categorical_{column}"
                        )

            submitted = st.form_submit_button(
                "🔮 Estimer le prix de la maison",
                type="primary"
            )

        if submitted:
            try:
                new_house = pd.DataFrame(
                    [input_values],
                    columns=X_reference.columns
                )

                # Reproduire les types attendus par le modèle.
                for column in numeric_features:
                    new_house[column] = pd.to_numeric(
                        new_house[column],
                        errors="coerce"
                    )

                model = trained_models[chosen_model]
                prediction = float(model.predict(new_house)[0])

                st.success("Prédiction réalisée avec succès.")

                st.metric(
                    label=f"Prix estimé ({target})",
                    value=f"{prediction:,.2f}"
                )

                st.caption(
                    "L'estimation est exprimée dans l'unité de la variable "
                    "cible de votre jeu de données. Si la cible est MEDV "
                    "dans le jeu Boston Housing classique, elle peut "
                    "correspondre à des milliers de dollars ; vérifiez "
                    "la définition de votre fichier."
                )

                st.subheader("Récapitulatif des caractéristiques")
                st.dataframe(new_house, use_container_width=True)

                prediction_df = new_house.copy()
                prediction_df[f"Prediction_{target}"] = prediction

                csv_prediction = prediction_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "📥 Télécharger cette prédiction",
                    data=csv_prediction,
                    file_name="prediction_prix_maison.csv",
                    mime="text/csv"
                )

            except Exception as error:
                st.error(f"Erreur lors de la prédiction : {error}")

        # Comparer les prédictions du modèle sur le jeu de test.
        with st.expander("Voir les prédictions sur les données de test"):
            if "ml_test_predictions" in st.session_state:
                model_predictions = st.session_state[
                    "ml_test_predictions"
                ][chosen_model]

                comparison = pd.DataFrame({
                    "Valeur réelle": st.session_state["ml_test_y"],
                    "Valeur prédite": model_predictions
                })

                comparison["Erreur absolue"] = (
                    comparison["Valeur réelle"]
                    - comparison["Valeur prédite"]
                ).abs()

                st.dataframe(
                    comparison.head(100).round(4),
                    use_container_width=True
                )


# ============================================================
# 11. PIED DE PAGE
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <p><b>Application de Data Science et Machine Learning</b></p>
        <p>Exploration • Visualisation • Modélisation • Prédiction</p>
    </div>
    """,
    unsafe_allow_html=True
)
