
# ============================================================
# COMPLETE DATA SCIENCE AND MACHINE LEARNING APPLICATION
# Data Analysis and House Price Prediction
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
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Science - House Prices",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Data Science and Machine Learning Application")

st.markdown(
    """
    **Exploratory data analysis, visualization, model training,
    and interactive house price prediction.**
    """
)

st.divider()


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def load_data(uploaded_file):
    """Load an uploaded CSV file or a local dataset."""
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
    """Compatible with multiple scikit-learn versions."""
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
    """Separate the target variable from explanatory variables."""
    if target not in data.columns:
        raise ValueError("The target variable was not found.")

    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(df[target], errors="coerce")

    valid_rows = y.notna()
    X = df.loc[valid_rows].drop(columns=[target])
    y = y.loc[valid_rows]

    X = X.dropna(axis=1, how="all")

    for column in X.columns:
        if X[column].dtype == "object":
            converted = pd.to_numeric(X[column], errors="coerce")
            if converted.notna().mean() >= 0.90:
                X[column] = converted

    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        raise ValueError(
            "No usable explanatory variables were found."
        )

    if len(X) < 5:
        raise ValueError(
            "The dataset must contain at least 5 valid rows."
        )

    return X, y


def build_pipeline(X, model_name, random_state=7):
    """Build the preprocessing pipeline and regression model."""

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
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(alpha=1.0),

        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state
        ),

        "K-Nearest Neighbors": KNeighborsRegressor(
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

        "Support Vector Regression (SVR)": SVR()
    }

    if model_name not in models:
        raise ValueError("Unrecognized model.")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_name])
    ])


# ============================================================
# 3. DATA LOADING
# ============================================================

st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

data = load_data(uploaded_file)

if data is None:
    st.warning(
        "No dataset was found. Upload a CSV file using the sidebar, "
        "or place HousingData.csv in the same folder as app.py."
    )

    st.code(
        "pip install streamlit pandas numpy matplotlib scikit-learn"
    )

    st.stop()

if data.empty:
    st.error("The uploaded file contains no data.")
    st.stop()

data.columns = data.columns.astype(str).str.strip()

if any(not column for column in data.columns):
    st.error("The dataset contains an empty column name.")
    st.stop()

if data.columns.duplicated().any():
    st.error(
        "The dataset contains duplicate column names. "
        "Please rename them before continuing."
    )
    st.stop()

st.sidebar.success("Dataset successfully loaded!")


# ============================================================
# 4. NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "Data Exploration",
        "Data Visualization",
        "Correlation Analysis",
        "Machine Learning",
        "House Price Prediction"
    ]
)


# ============================================================
# 5. HOME PAGE
# ============================================================

if page == "Home":

    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Rows", f"{data.shape[0]:,}")

    with col2:
        st.metric("Number of Columns", f"{data.shape[1]:,}")

    with col3:
        st.metric(
            "Missing Values",
            f"{int(data.isna().sum().sum()):,}"
        )

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("General Information")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": [str(data[c].dtype) for c in data.columns],
        "Missing Values": [
            int(data[c].isna().sum()) for c in data.columns
        ],
        "Unique Values": [
            int(data[c].nunique(dropna=True)) for c in data.columns
        ]
    })

    st.dataframe(info_df, use_container_width=True)

    st.info(
        "Use the sidebar menu to explore your data, "
        "visualize variables, train machine learning models, "
        "and estimate house prices."
    )


# ============================================================
# 6. DATA EXPLORATION
# ============================================================

elif page == "Data Exploration":

    st.header("🔎 Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Preview",
        "Descriptive Statistics",
        "Missing Values",
        "Data Types"
    ])

    with tab1:
        st.subheader("First Rows")
        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Last Rows")
        st.dataframe(data.tail(10), use_container_width=True)

    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(
            data.describe(include="all").T,
            use_container_width=True
        )

    with tab3:
        missing = data.isna().sum()

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values,
            "Percentage (%)": (
                missing.values / len(data) * 100
            ).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        missing_only = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if not missing_only.empty:
            st.bar_chart(
                missing_only.set_index("Column")[
                    "Missing Values"
                ]
            )
        else:
            st.success("No missing values detected.")

    with tab4:
        types_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": [str(data[c].dtype) for c in data.columns],
            "Unique Values": [
                data[c].nunique(dropna=True) for c in data.columns
            ]
        })

        st.dataframe(types_df, use_container_width=True)


# ============================================================
# 7. DATA VISUALIZATION
# ============================================================

elif page == "Data Visualization":

    st.header("📈 Data Visualization")

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        st.warning(
            "No numerical variables are available for visualization."
        )

    else:
        plot_type = st.selectbox(
            "Select Chart Type",
            [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Scatter Matrix"
            ]
        )

        if plot_type == "Histogram":

            selected_columns = st.multiselect(
                "Select Variables",
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

                ax.set_title("Variable Distributions")
                ax.set_xlabel("Values")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(alpha=0.2)

                st.pyplot(fig)
                plt.close(fig)

        elif plot_type == "Box Plot":

            column = st.selectbox(
                "Select a Variable",
                numeric_columns
            )

            values = pd.to_numeric(
                data[column], errors="coerce"
            ).dropna()

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.boxplot(values, vert=False)
            ax.set_title(f"Box Plot: {column}")
            ax.set_xlabel(column)
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        elif plot_type == "Scatter Plot":

            if len(numeric_columns) < 2:

                st.warning(
                    "At least two numerical variables are required."
                )

            else:

                x_col = st.selectbox(
                    "X-axis Variable",
                    numeric_columns,
                    index=0
                )

                y_col = st.selectbox(
                    "Y-axis Variable",
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
                ax.set_title(f"{y_col} vs {x_col}")
                ax.grid(alpha=0.2)

                st.pyplot(fig)
                plt.close(fig)

        else:

            selected_columns = st.multiselect(
                "Select Variables for the Scatter Matrix",
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

                plt.suptitle("Scatter Matrix")

                st.pyplot(plt.gcf())
                plt.close("all")

            else:
                st.info("Please select at least two variables.")


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

elif page == "Correlation Analysis":

    st.header("🔗 Correlation Analysis")

    numeric_data = data.select_dtypes(
        include=np.number
    )

    if numeric_data.shape[1] < 2:

        st.warning(
            "At least two numerical variables are required "
            "to calculate a correlation matrix."
        )

    else:

        corr = numeric_data.corr()

        st.subheader("Correlation Matrix")

        st.dataframe(
            corr.round(3),
            use_container_width=True
        )

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

        fig.colorbar(image, ax=ax, label="Correlation")

        ax.set_title("Correlation Heatmap")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Strongest Correlations")

        pairs = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        ).stack().reset_index()

        if not pairs.empty:

            pairs.columns = [
                "Variable 1",
                "Variable 2",
                "Correlation"
            ]

            pairs["Absolute Correlation"] = pairs[
                "Correlation"
            ].abs()

            pairs = pairs.sort_values(
                "Absolute Correlation",
                ascending=False
            )

            st.dataframe(
                pairs.drop(
                    columns=["Absolute Correlation"]
                ).head(15),
                use_container_width=True
            )


# ============================================================
# 9. MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.header("🤖 Model Training and Evaluation")

    st.write(
        "Select the target variable, the models to compare, "
        "and the proportion of data reserved for testing."
    )

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    target_options = (
        numeric_columns if numeric_columns else list(data.columns)
    )

    default_target = (
        "MEDV" if "MEDV" in target_options
        else target_options[-1]
    )

    target = st.selectbox(
        "Target Variable to Predict",
        target_options,
        index=target_options.index(default_target)
    )

    models_to_train = st.multiselect(
        "Select Models to Compare",
        [
            "Linear Regression",
            "Ridge Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "Support Vector Regression (SVR)"
        ],
        default=[
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    random_state = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=7,
        step=1
    )

    use_cv = st.checkbox(
        "Enable Cross-Validation",
        value=False
    )

    use_grid_search = st.checkbox(
        "Optimize Gradient Boosting using GridSearchCV",
        value=False
    )

    if (
        use_grid_search
        and "Gradient Boosting" not in models_to_train
    ):

        st.warning(
            "Add Gradient Boosting to the model list "
            "to use this optimization."
        )

    if st.button("🚀 Train Models", type="primary"):

        if not models_to_train:

            st.error("Please select at least one model.")
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

                status.write(f"Training: {model_name}")

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
                        "Best Parameters:",
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
                    "Model": model_name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R²": r2,
                    "Cross-Validation RMSE": cv_rmse
                })

                trained_models[model_name] = pipeline
                test_predictions[model_name] = predictions

                progress.progress(
                    (index + 1) / len(models_to_train)
                )

            status.success("Model training completed successfully!")

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

            st.subheader("Model Performance Results")

            st.dataframe(
                results_df.round(4),
                use_container_width=True
            )

            best_model_name = results_df.iloc[0]["Model"]
            best_rmse = results_df.iloc[0]["RMSE"]
            best_r2 = results_df.iloc[0]["R²"]

            col1, col2, col3 = st.columns(3)

            col1.metric("Best Model", best_model_name)

            col2.metric(
                "Best Model RMSE",
                f"{best_rmse:.4f}"
            )

            col3.metric(
                "Best Model R²",
                f"{best_r2:.4f}"
            )

            st.subheader("RMSE Comparison")

            chart_data = results_df.set_index("Model")[["RMSE"]]

            st.bar_chart(chart_data)

            csv_results = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Results as CSV",
                data=csv_results,
                file_name="machine_learning_results.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(f"Training Error: {error}")

    if "ml_results" in st.session_state:

        st.divider()

        st.subheader("Latest Saved Results")

        st.dataframe(
            st.session_state["ml_results"].round(4),
            use_container_width=True
        )


# ============================================================
# 10. INTERACTIVE HOUSE PRICE PREDICTION
# ============================================================

elif page == "House Price Prediction":

    st.header("🏡 Predict House Prices")

    st.write(
        """
        After training a model on the **Machine Learning** page,
        select the model and enter the characteristics of the house.
        The application will use the trained model to generate
        a price estimate.
        """
    )

    if (
        "ml_models" not in st.session_state
        or not st.session_state["ml_models"]
    ):

        st.warning(
            "No trained model is available. "
            "Go to the Machine Learning page, select a target variable, "
            "and train at least one model before returning here."
        )

    else:

        trained_models = st.session_state["ml_models"]

        X_reference = st.session_state["ml_X_reference"]

        target = st.session_state["ml_target"]

        model_names = list(trained_models.keys())

        default_model = (
            st.session_state["ml_results"].iloc[0]["Model"]
            if "ml_results" in st.session_state
            else model_names[0]
        )

        default_index = (
            model_names.index(default_model)
            if default_model in model_names else 0
        )

        chosen_model = st.selectbox(
            "Prediction Model",
            model_names,
            index=default_index
        )

        st.subheader("House Characteristics")

        st.caption(
            "Numerical values are initialized using the median "
            "of the training data. Categorical variables are "
            "initialized using their most frequent value."
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

                st.markdown("#### Numerical Features")

                numeric_cols = st.columns(2)

                for i, column in enumerate(numeric_features):

                    series = pd.to_numeric(
                        X_reference[column],
                        errors="coerce"
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
                                f"Observed data range: "
                                f"{minimum:.4g} to {maximum:.4g}"
                            ),
                            key=f"numeric_{column}"
                        )

            if categorical_features:

                st.markdown("#### Categorical Features")

                categorical_cols = st.columns(2)

                for i, column in enumerate(categorical_features):

                    series = X_reference[column].dropna()

                    if series.empty:
                        choices = ["Unknown"]
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
                "🔮 Predict House Price",
                type="primary"
            )

        if submitted:

            try:

                new_house = pd.DataFrame(
                    [input_values],
                    columns=X_reference.columns
                )

                for column in numeric_features:

                    new_house[column] = pd.to_numeric(
                        new_house[column],
                        errors="coerce"
                    )

                model = trained_models[chosen_model]

                prediction = float(
                    model.predict(new_house)[0]
                )

                st.success("Prediction completed successfully!")

                st.metric(
                    label=f"Estimated House Price ({target})",
                    value=f"{prediction:,.2f}"
                )

                st.caption(
                    "The estimate is expressed in the unit of the "
                    "target variable in your dataset. If the target "
                    "is MEDV in the classic Boston Housing dataset, "
                    "it may represent thousands of dollars. "
                    "Please check your dataset documentation."
                )

                st.subheader("House Characteristics Summary")

                st.dataframe(
                    new_house,
                    use_container_width=True
                )

                prediction_df = new_house.copy()

                prediction_df[f"Prediction_{target}"] = prediction

                csv_prediction = prediction_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "📥 Download This Prediction",
                    data=csv_prediction,
                    file_name="house_price_prediction.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(f"Prediction Error: {error}")

        with st.expander(
            "View Predictions on Test Data"
        ):

            if "ml_test_predictions" in st.session_state:

                model_predictions = st.session_state[
                    "ml_test_predictions"
                ][chosen_model]

                comparison = pd.DataFrame({
                    "Actual Value": st.session_state["ml_test_y"],
                    "Predicted Value": model_predictions
                })

                comparison["Absolute Error"] = (
                    comparison["Actual Value"]
                    - comparison["Predicted Value"]
                ).abs()

                st.dataframe(
                    comparison.head(100).round(4),
                    use_container_width=True
                )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <p><b>Data Science and Machine Learning Application</b></p>
        <p>Data Exploration • Visualization • Modeling • Prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)
