import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples
from sklearn.ensemble import IsolationForest
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

def scale_features(X, scaler_option):
    """Scale features based on the selected scaler."""
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        scaler = None  # No scaling
        X_scaled = X  # No scaling
    st.session_state.scaler = scaler  # Save scaler for later use
    return X_scaled

def plot_silhouette(X, labels):
    """Plot silhouette plot for clustering evaluation."""
    silhouette_avg = silhouette_score(X, labels)
    st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")

    st.write("**Silhouette Plot:**")
    silhouette_values = silhouette_samples(X, labels)
    y_lower = 10

    fig, ax = plt.subplots()
    for i in range(len(np.unique(labels))):
        cluster_silhouette_values = silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    st.pyplot(fig)

def plot_cluster_heatmap(data, features):
    """Plot heatmap of cluster means."""
    st.write("**Cluster Heatmap:**")
    cluster_means = data.groupby("Cluster")[features].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

def detect_outliers(model, X, data, features):
    """Detect and visualize outliers using the given model."""
    outlier_labels = model.fit_predict(X)
    data["Outlier"] = np.where(outlier_labels == -1, "Outlier", "Inlier")

    st.write("Outlier Detection Results:")
    st.dataframe(data)

    # Plot outliers
    fig = px.scatter(data, x=features[0], y=features[1], color="Outlier", title="Outlier Detection Visualization")
    st.plotly_chart(fig)

    # Outlier statistics
    outlier_count = (outlier_labels == -1).sum()
    st.write(f"**Number of Outliers Detected:** {outlier_count}")

def clustering_page():
    st.header("Clustering and Outlier Detection")

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

        # Select features for unsupervised learning
        features = st.multiselect("Select features for unsupervised learning:", data.columns)

        if len(features) > 0:
            X = data[features]

            # Feature Scaling
            scaler_option = st.selectbox("Select feature scaling:", ["None", "StandardScaler", "MinMaxScaler"], help="Choose a scaler to normalize your data.")
            X_scaled = scale_features(X, scaler_option)

            # Select model
            model_options = {
                "K-Means": KMeans(n_clusters=3),
                "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
                "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
                "PCA": PCA(n_components=2),
                "Isolation Forest": IsolationForest(contamination=0.1)  # Outlier detection
            }
            selected_model = st.selectbox("Select a model:", list(model_options.keys()))

            # Train the model
            if st.button("Train Model"):
                model = model_options[selected_model]

                if selected_model == "PCA":
                    st.write("Reduce dimensionality using Principal Component Analysis (PCA).")
                    X_transformed = model.fit_transform(X_scaled)
                    st.write("PCA Components:")
                    st.dataframe(pd.DataFrame(X_transformed, columns=[f"PC{i+1}" for i in range(X_transformed.shape[1])]))

                    # Plot PCA components
                    fig = px.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], title="PCA Visualization")
                    st.plotly_chart(fig)
                elif selected_model == "Isolation Forest":
                    st.write("Detecting outliers using Isolation Forest.")
                    detect_outliers(model, X_scaled, data, features)
                else:
                    st.write("Cluster data using the selected unsupervised model.")
                    labels = model.fit_predict(X_scaled)
                    data["Cluster"] = labels

                    st.write("Cluster Assignments:")
                    st.dataframe(data)

                    # Plot clusters
                    if selected_model != "DBSCAN":
                        fig = px.scatter(data, x=features[0], y=features[1], color="Cluster", title="Cluster Visualization")
                        st.plotly_chart(fig)

                    # Clustering Evaluation
                    st.subheader("Clustering Evaluation")

                    if selected_model != "PCA":
                        # Silhouette Score and Plot
                        plot_silhouette(X_scaled, labels)

                        # Davies-Bouldin Index
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                        st.write(f"**Davies-Bouldin Index:** {davies_bouldin:.2f}")

                        # Calinski-Harabasz Index
                        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                        st.write(f"**Calinski-Harabasz Index:** {calinski_harabasz:.2f}")

                        # Cluster Heatmap
                        plot_cluster_heatmap(data, features)

                    # Outlier Detection for DBSCAN
                    if selected_model == "DBSCAN":
                        st.subheader("Outlier Detection (DBSCAN)")
                        detect_outliers(model, X_scaled, data, features)

                # Export the trained model
                if selected_model != "PCA" and selected_model != "Isolation Forest":
                    model_dir = os.path.join("saved_models")  # Define the directory for saving models
                    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

                    model_file = os.path.join(model_dir, f"{selected_model}.pkl")  # Define the full file path
                    joblib.dump(model, model_file)  # Save the model to the specified file

                    # Provide a download button for the saved model
                    with open(model_file, "rb") as f:
                        st.download_button(
                            label="Download Model",
                            data=f,
                            file_name=f"{selected_model}.pkl",
                            mime="application/octet-stream"
                        )

        else:
            st.warning("Select at least one feature for unsupervised learning.")
    else:
        st.warning("No data found. Please upload a dataset to proceed.")