import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import os

# Helper Functions
def load_data(file_path):
    """Load data from a file path."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def save_data(data, file_path):
    """Save data to a file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
    st.success(f"Data saved successfully at: {file_path}")

def encode_categorical_features(X):
    """Encode categorical features using OneHotEncoder."""
    categorical_features = X.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_encoded = encoder.fit_transform(X[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
        X = pd.concat([X.drop(categorical_features, axis=1), X_encoded_df], axis=1)
    return X

def scale_features(X, scaler_option):
    """Scale features based on the selected scaler."""
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.write("Features scaled using StandardScaler (mean=0, std=1).")
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.write("Features scaled using MinMaxScaler (scaled to [0, 1]).")
    st.session_state.scaler = scaler
    return X

def select_features(X, y, k):
    """Select top k features using univariate statistical tests."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_feature_names), selected_feature_names

def train_model(model, X_train, y_train, param_grid=None, tuning_method=None, cv_folds=5):
    """Train a model with optional hyperparameter tuning."""
    if tuning_method == "Grid Search":
        search = GridSearchCV(model, param_grid, cv=cv_folds, scoring="r2")
        search.fit(X_train, y_train)
        return search.best_estimator_
    elif tuning_method == "Random Search":
        search = RandomizedSearchCV(model, param_grid, cv=cv_folds, n_iter=10, scoring="r2", random_state=42)
        search.fit(X_train, y_train)
        return search.best_estimator_
    elif tuning_method == "Bayesian Optimization":
        search = BayesSearchCV(model, param_grid, cv=cv_folds, n_iter=10, scoring="r2", random_state=42)
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "MSE": mse,
        "R²": r2
    }

def plot_residuals(y_test, y_pred):
    """Plot residuals for the model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_test - y_pred, s=10, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    st.pyplot(fig)

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)
        fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance", height=300)
        st.plotly_chart(fig)

def plot_prediction_vs_actual(y_test, y_pred):
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, s=10, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

def regression_page():
    st.header("📈 Regression Analysis")

    # Define default data directory
    data_dir = os.path.join("data", "model")
    os.makedirs(data_dir, exist_ok=True)

    # Load default preprocessed data if it exists
    default_data_path = os.path.join(data_dir, "preprocessed_data.csv")
    default_data = load_data(default_data_path)

    # Upload multiple files
    st.subheader("Upload Datasets")
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    # Save uploaded files to the data directory
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            save_data(pd.read_csv(uploaded_file), file_path)

    # List all datasets in the data directory
    dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not dataset_files:
        st.warning("No datasets found. Please upload at least one dataset.")
        return

    # Let the user choose which dataset to use
    selected_dataset = st.selectbox("Select a dataset:", dataset_files)
    data_path = os.path.join(data_dir, selected_dataset)
    data = load_data(data_path)

    if data is not None:
        st.write("Selected Dataset:")
        st.dataframe(data)

        # Select target column
        target_column = st.selectbox("Select the target column:", data.columns, help="Choose the column you want to predict.")

        if target_column in data.columns:
            # Manual feature selection
            st.subheader("Feature Selection")
            features = st.multiselect("Select features:", data.columns, default=list(data.columns), help="Choose the features to include in the model.")
            if not features:
                st.error("Please select at least one feature.")
                return

            X = data[features]
            y = data[target_column]

            # Encode categorical features
            X = encode_categorical_features(X)

            # Feature Scaling
            scaler_option = st.selectbox("Select feature scaling:", ["None", "StandardScaler", "MinMaxScaler"], help="Choose a scaler to normalize your data.")
            if scaler_option != "None":
                X = scale_features(X, scaler_option)

            # Split the dataset
            test_size = st.slider("Select test size:", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model selection
            st.subheader("Select Models for Comparison")
            model_options = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
                "Random Forest Regression": RandomForestRegressor(random_state=42),
                "Support Vector Regression (SVR)": SVR(),
                "Gradient Boosting Regression": GradientBoostingRegressor(random_state=42),
                "XGBoost Regression": XGBRegressor(random_state=42),
                "LightGBM Regression": LGBMRegressor(random_state=42),
            }
            selected_models = st.multiselect("Choose models to compare:", list(model_options.keys()))

            # Hyperparameter tuning options
            st.subheader("Hyperparameter Tuning")
            tuning_method = st.selectbox(
                "Choose a hyperparameter tuning method:",
                ["None", "Grid Search", "Random Search", "Bayesian Optimization"]
            )

            # Define hyperparameter grids for each model
            param_grids = {
                "Random Forest Regression": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting Regression": {
                    "n_estimators": [10, 50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 10]
                },
                "XGBoost Regression": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "LightGBM Regression": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                }
            }

            # Cross-validation
            cv_folds = st.slider("Number of cross-validation folds:", 2, 10, 5)

            # Train and evaluate models
            if st.button("Train and Compare Models"):
                results = []
                predictions = {}

                for model_name in selected_models:
                    with st.spinner(f"Training {model_name}..."):
                        model = model_options[model_name]
                        param_grid = param_grids.get(model_name, {})

                        # Train the model with hyperparameter tuning
                        model = train_model(model, X_train, y_train, param_grid, tuning_method, cv_folds)

                        # Evaluate the model
                        metrics = evaluate_model(model, X_test, y_test)
                        results.append({"Model": model_name, **metrics})

                        # Store predictions for visualization
                        y_pred = model.predict(X_test)
                        predictions[model_name] = y_pred

                # Display comparison table
                st.subheader("Model Comparison")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # Plot comparison metrics
                st.subheader("Comparison Metrics")
                fig = px.bar(results_df, x="Model", y=["MSE", "R²"], barmode="group", title="Model Comparison Metrics")
                st.plotly_chart(fig)

                # Visualizations
                st.subheader("Visualizations")
                for model_name, y_pred in predictions.items():
                    st.write(f"**{model_name}**")
                    
                    # Create columns for side-by-side visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        plot_residuals(y_test, y_pred)
                    
                    with col2:
                        plot_prediction_vs_actual(y_test, y_pred)

                    # Feature importance (for tree-based models)
                    if hasattr(model_options[model_name], "feature_importances_"):
                        st.write("**Feature Importance**")
                        plot_feature_importance(model_options[model_name], X.columns)

                # Save the best model
                best_model_name = results_df.loc[results_df["R²"].idxmax(), "Model"]
                best_model = model_options[best_model_name]
                st.session_state.best_model = best_model
                st.session_state.X_columns = X.columns
                st.success(f"✅ Best model: {best_model_name}")

                # Export the best model
                model_file = f"{best_model_name}.pkl"
                joblib.dump(best_model, model_file)
                with open(model_file, "rb") as f:
                    st.download_button(
                        label="Download Best Model",
                        data=f,
                        file_name=model_file,
                        mime="application/octet-stream"
                    )

                # Set a flag to indicate that models have been trained
                st.session_state.models_trained = True
        else:
            st.error("Target column not found in the dataset.")
    else:
        st.warning("No data found. Please upload a dataset to proceed.")