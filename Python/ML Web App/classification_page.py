import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib
import lime
import lime.lime_tabular
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

def encode_target_variable(y):
    """Encode the target variable if it is categorical."""
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        st.session_state.label_encoder = label_encoder
    else:
        y_encoded = y
    return y_encoded

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
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    st.session_state.scaler = scaler
    return X

def train_model(model, X_train, y_train, param_grid=None, tuning_method=None):
    """Train a model with optional hyperparameter tuning."""
    if tuning_method == "Grid Search":
        search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
        search.fit(X_train, y_train)
        return search.best_estimator_
    elif tuning_method == "Random Search":
        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, scoring="accuracy", random_state=42)
        search.fit(X_train, y_train)
        return search.best_estimator_
    elif tuning_method == "Bayesian Optimization":
        search = BayesSearchCV(model, param_grid, cv=3, n_iter=10, scoring="accuracy", random_state=42)
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None and len(np.unique(y_test)) == 2 else None

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot a confusion matrix for the model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.write(f"Confusion Matrix for {model_name}")
    st.pyplot(fig)

def plot_lime_explanation(explainer, X_test, model, instance_idx):
    """Plot LIME explanation for a single instance."""
    exp = explainer.explain_instance(X_test.iloc[instance_idx].values, model.predict_proba, num_features=5)
    st.write(f"LIME Explanation for Instance {instance_idx}")
    st.write(exp.as_list())
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

def feature_selection(X, y, method, k=10):
    """Perform feature selection based on the selected method."""
    if method == "Univariate Feature Selection":
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        X_selected = pd.DataFrame(X_selected, columns=selected_features)  # Convert to DataFrame
        return X_selected, selected_features
    elif method == "Recursive Feature Elimination (RFE)":
        model = LogisticRegression()
        selector = RFE(model, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.support_]
        X_selected = pd.DataFrame(X_selected, columns=selected_features)  # Convert to DataFrame
        return X_selected, selected_features
    elif method == "Feature Importance (Random Forest)":
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        selected_features = X.columns[np.argsort(importances)[-k:]]
        X_selected = X[selected_features]  # Already a DataFrame
        return X_selected, selected_features
    else:
        return X, X.columns

def classification_page():
    st.header("🔮 Classification")

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
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # Encode the target variable
            y_encoded = encode_target_variable(y)

            # Encode categorical features
            X = encode_categorical_features(X)

            # Feature Scaling
            scaler_option = st.selectbox("Select feature scaling:", ["None", "StandardScaler", "MinMaxScaler"], help="Choose a scaler to normalize your data.")
            if scaler_option != "None":
                X = scale_features(X, scaler_option)

            # Feature Selection
            st.subheader("Feature Selection")
            feature_selection_method = st.selectbox(
                "Choose a feature selection method:",
                ["None", "Univariate Feature Selection", "Recursive Feature Elimination (RFE)", "Feature Importance (Random Forest)"]
            )
            if feature_selection_method != "None":
                k = st.slider("Select the number of features to keep:", 1, len(X.columns), min(10, len(X.columns)))
                X_selected, selected_features = feature_selection(X, y_encoded, feature_selection_method, k)
                st.write(f"Selected Features: {list(selected_features)}")
                X = X_selected  # X is now a DataFrame

            # Split the dataset
            test_size = st.slider("Select test size:", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

            # Model selection
            st.subheader("Select Models for Comparison")
            model_options = {
                "RandomForestClassifier": RandomForestClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "LogisticRegression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "LightGBM": LGBMClassifier(),
                "Naive Bayes": GaussianNB()
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
                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "SVM": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                "DecisionTreeClassifier": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [10, 50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 10]
                },
                "LogisticRegression": {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                },
                "XGBoost": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "LightGBM": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "Naive Bayes": {}  # No hyperparameters to tune
            }

            # Train and evaluate models
            if st.button("Train and Compare Models"):
                results = []
                confusion_matrices = []

                for model_name in selected_models:
                    with st.spinner(f"Training {model_name}..."):
                        model = model_options[model_name]
                        param_grid = param_grids[model_name]

                        # Train the model with hyperparameter tuning
                        model = train_model(model, X_train, y_train, param_grid, tuning_method)

                        # Evaluate the model
                        metrics = evaluate_model(model, X_test, y_test)
                        results.append({"Model": model_name, **metrics})

                        # Store confusion matrix for side-by-side display
                        y_pred = model.predict(X_test)
                        confusion_matrices.append((model_name, confusion_matrix(y_test, y_pred)))

                # Display comparison table
                st.subheader("Model Comparison")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # Plot comparison metrics
                st.subheader("Comparison Metrics")
                fig = px.bar(results_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score"], barmode="group", title="Model Comparison Metrics")
                st.plotly_chart(fig)

                # Display confusion matrices side by side
                st.subheader("Confusion Matrices")
                if len(confusion_matrices) > 0:
                    cols = st.columns(2)
                    for idx, (model_name, cm) in enumerate(confusion_matrices):
                        with cols[idx % 2]:
                            plot_confusion_matrix(y_test, y_pred, model_name)

                # Save the best model
                best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
                best_model = model_options[best_model_name]
                st.session_state.best_model = best_model
                st.session_state.X_columns = X.columns
                st.success(f"✅ Best model: {best_model_name}")

                # Export the best model
                model_dir = os.path.join("saved_models")  # Define the directory for saving models
                os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

                model_file = os.path.join(model_dir, f"{best_model_name}.pkl")  # Define the full file path
                joblib.dump(best_model, model_file)  # Save the model to the specified file

                # Provide a download button for the saved model
                with open(model_file, "rb") as f:
                    st.download_button(
                        label="Download Best Model",
                        data=f,
                        file_name=f"{best_model_name}.pkl",
                        mime="application/octet-stream"
                    )

                # Set a flag to indicate that models have been trained
                st.session_state.models_trained = True

        # Model Interpretability (only show if models have been trained)
        if "models_trained" in st.session_state and st.session_state.models_trained:
            st.subheader("Model Interpretability")
            interpretability_method = st.selectbox(
                "Choose an interpretability method:",
                ["None", "LIME"]
            )

            if interpretability_method == "LIME":
                st.write("Calculating LIME explanations...")
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=X.columns,
                    class_names=np.unique(y_encoded).astype(str),
                    mode="classification"
                )
                instance_idx = st.slider("Select an instance to explain:", 0, len(X_test) - 1, 0)
                plot_lime_explanation(explainer, X_test, st.session_state.best_model, instance_idx)
        else:
            st.error("Target column not found in the dataset.")
    else:
        st.warning("No data found. Please upload a dataset to proceed.")