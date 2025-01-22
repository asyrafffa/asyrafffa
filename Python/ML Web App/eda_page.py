import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ttest_ind, shapiro

# Ensure the data directory exists
os.makedirs("data/preprocessing", exist_ok=True)

# Initialize session state for EDA
def initialize_eda_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None

# Load uploaded files into session state
def load_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_files:
            # Save the uploaded file to the preprocessing folder
            file_path = os.path.join("data/preprocessing", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Load the dataset into session state
            st.session_state.uploaded_files[uploaded_file.name] = pd.read_csv(file_path)

# Display dataset preview
def display_dataset_preview(data, num_rows=5):
    st.write(f"Dataset Preview (showing {num_rows} rows):")
    st.dataframe(data.head(num_rows))

# Handle data exploration and visualization
def handle_data_exploration(data):
    st.header("Exploratory Data Analysis (EDA)")

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Feature Distribution", "Correlation Matrix", "Pair Plots", "Missing Data Analysis",
        "Outlier Detection", "Categorical Data Visualization", "Time Series Analysis",
        "Multivariate Analysis", "Statistical Tests", "Interactive Visualizations"
    ])

    with tab1:
        st.subheader("Advanced Distribution Plots")
        if len(numeric_columns) > 0:
            selected_column = st.selectbox(
                "Select a numeric column to visualize:", 
                numeric_columns, 
                key="dist_select_column"  # Unique key
            )
            plot_type = st.selectbox(
                "Select plot type:", 
                ["Histogram", "Box Plot", "Density Plot", "Violin Plot", "CDF"], 
                key="dist_plot_type"  # Unique key
            )

            if plot_type == "Histogram":
                bins = st.slider(
                    "Number of bins:", 
                    5, 100, 20, 
                    key="hist_bins"  # Unique key
                )
                fig = px.histogram(data, x=selected_column, nbins=bins, title=f"Histogram of {selected_column}")
            elif plot_type == "Box Plot":
                fig = px.box(data, y=selected_column, title=f"Box Plot of {selected_column}")
            elif plot_type == "Density Plot":
                fig = px.density_contour(data, x=selected_column, title=f"Density Plot of {selected_column}")
            elif plot_type == "Violin Plot":
                fig = px.violin(data, y=selected_column, box=True, points="all", title=f"Violin Plot of {selected_column}")
            elif plot_type == "CDF":
                fig = px.ecdf(data, x=selected_column, title=f"CDF of {selected_column}")

            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns found for distribution plots.")

    with tab2:
        st.subheader("Correlation Matrix")
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns to compute correlation matrix.")

    with tab3:
        st.subheader("Pair Plots")
        if len(numeric_columns) > 1:
            pair_plot_columns = st.multiselect(
                "Select columns for pair plot:", 
                numeric_columns, 
                default=numeric_columns[:3], 
                key="pair_plot_columns"  # Unique key
            )
            if len(pair_plot_columns) > 1:
                fig = px.scatter_matrix(data[pair_plot_columns], title="Pair Plots")
                st.plotly_chart(fig)
            else:
                st.warning("Select at least 2 columns for pair plot.")
        else:
            st.warning("Not enough numeric columns for pair plots.")

    with tab4:
        st.subheader("Missing Data Analysis")
        if data.isnull().sum().sum() > 0:
            st.write("Summary of Missing Values:")
            missing_data = data.isnull().sum().reset_index()
            missing_data.columns = ["Column", "Missing Values"]
            st.dataframe(missing_data)
        else:
            st.success("No missing values found in the dataset.")

    with tab5:
        st.subheader("Outlier Detection")
        if len(numeric_columns) > 0:
            selected_column = st.selectbox(
                "Select a numeric column for outlier detection:", 
                numeric_columns, 
                key="outlier_select_column"  # Unique key
            )
            outlier_method = st.selectbox(
                "Select outlier detection method:", 
                ["Z-Score", "IQR"], 
                key="outlier_method"  # Unique key
            )

            if outlier_method == "Z-Score":
                z_scores = (data[selected_column] - data[selected_column].mean()) / data[selected_column].std()
                outliers = data[(z_scores > 3) | (z_scores < -3)]
            elif outlier_method == "IQR":
                Q1 = data[selected_column].quantile(0.25)
                Q3 = data[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound)]

            st.write(f"Outliers detected using {outlier_method}:")
            st.dataframe(outliers)

            fig = px.box(data, y=selected_column, title=f"Box Plot of {selected_column} (Outliers Highlighted)")
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns found for outlier detection.")

    with tab6:
        st.subheader("Categorical Data Visualization")
        if len(categorical_columns) > 0:
            selected_cat_column = st.selectbox(
                "Select a categorical column:", 
                categorical_columns, 
                key="cat_select_column"  # Unique key
            )
            plot_type = st.selectbox(
                "Select plot type:", 
                ["Bar Plot", "Pie Chart"], 
                key="cat_plot_type"  # Unique key
            )

            if plot_type == "Bar Plot":
                fig = px.bar(data[selected_cat_column].value_counts(), title=f"Bar Plot of {selected_cat_column}")
            elif plot_type == "Pie Chart":
                fig = px.pie(data[selected_cat_column].value_counts(), names=data[selected_cat_column].unique(), title=f"Pie Chart of {selected_cat_column}")

            st.plotly_chart(fig)
        else:
            st.warning("No categorical columns found for visualization.")

    with tab7:
        st.subheader("Time Series Analysis")
        if len(numeric_columns) > 0:
            time_column = st.selectbox(
                "Select a time column:", 
                data.columns, 
                key="time_column"  # Unique key
            )
            value_column = st.selectbox(
                "Select a value column:", 
                numeric_columns, 
                key="value_column"  # Unique key
            )
            fig = px.line(data, x=time_column, y=value_column, title=f"Time Series Plot of {value_column}")
            st.plotly_chart(fig)

            if st.checkbox("Show Seasonality Decomposition", key="seasonality_decomp"):
                decomposition = seasonal_decompose(data[value_column], period=12, model='additive')
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
                decomposition.trend.plot(ax=ax1, title="Trend")
                decomposition.seasonal.plot(ax=ax2, title="Seasonality")
                decomposition.resid.plot(ax=ax3, title="Residuals")
                decomposition.observed.plot(ax=ax4, title="Observed")
                st.pyplot(fig)
        else:
            st.warning("No numeric columns found for time series analysis.")

    with tab8:
        st.subheader("Multivariate Analysis")
        if len(numeric_columns) > 2:
            x_col = st.selectbox(
                "Select X-axis column:", 
                numeric_columns, 
                key="x_col"  # Unique key
            )
            y_col = st.selectbox(
                "Select Y-axis column:", 
                numeric_columns, 
                key="y_col"  # Unique key
            )
            z_col = st.selectbox(
                "Select Z-axis column:", 
                numeric_columns, 
                key="z_col"  # Unique key
            )
            fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, title=f"3D Scatter Plot of {x_col}, {y_col}, {z_col}")
            st.plotly_chart(fig)

            if st.checkbox("Show Parallel Coordinates Plot", key="parallel_coords"):
                parallel_columns = st.multiselect(
                    "Select columns for parallel coordinates:", 
                    numeric_columns, 
                    default=numeric_columns[:3], 
                    key="parallel_columns"  # Unique key
                )
                if len(parallel_columns) > 1:
                    fig = px.parallel_coordinates(data[parallel_columns], title="Parallel Coordinates Plot")
                    st.plotly_chart(fig)
                else:
                    st.warning("Select at least 2 columns for parallel coordinates.")
        else:
            st.warning("Not enough numeric columns for multivariate analysis.")

    with tab9:
        st.subheader("Statistical Tests")
        if len(numeric_columns) > 1:
            test_type = st.selectbox(
                "Select a statistical test:", 
                ["T-Test", "ANOVA", "Normality Test", "Correlation Test"], 
                key="stat_test_type"  # Unique key
            )
            if test_type == "T-Test":
                group_column = st.selectbox(
                    "Select a categorical column for grouping:", 
                    categorical_columns, 
                    key="group_column"  # Unique key
                )
                value_column = st.selectbox(
                    "Select a numeric column for testing:", 
                    numeric_columns, 
                    key="t_test_value_column"  # Unique key
                )
                group1 = data[data[group_column] == data[group_column].unique()[0]][value_column]
                group2 = data[data[group_column] == data[group_column].unique()[1]][value_column]
                t_stat, p_value = ttest_ind(group1, group2)
                st.write(f"T-Statistic: {t_stat}, P-Value: {p_value}")
            elif test_type == "Normality Test":
                selected_column = st.selectbox(
                    "Select a numeric column for normality test:", 
                    numeric_columns, 
                    key="normality_column"  # Unique key
                )
                stat, p_value = shapiro(data[selected_column])
                st.write(f"Shapiro-Wilk Test Statistic: {stat}, P-Value: {p_value}")
        else:
            st.warning("Not enough numeric columns for statistical tests.")

    with tab10:
        st.subheader("Interactive Visualizations")
        if len(numeric_columns) > 1:
            x_col = st.selectbox(
                "Select X-axis column:", 
                numeric_columns, 
                key="interactive_x_col"  # Unique key
            )
            y_col = st.selectbox(
                "Select Y-axis column:", 
                numeric_columns, 
                key="interactive_y_col"  # Unique key
            )
            color_col = st.selectbox(
                "Select a column for color encoding:", 
                categorical_columns, 
                key="color_column"  # Unique key
            )
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot of {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for interactive visualizations.")

# Main function for the EDA page
def eda_page():
    st.header("Exploratory Data Analysis (EDA)")
    initialize_eda_session_state()

    # Upload multiple datasets
    st.write("Upload your datasets (CSV format):")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, key="file_uploader")
    load_uploaded_files(uploaded_files)

    # Select a dataset for EDA
    if st.session_state.uploaded_files:
        st.session_state.selected_dataset = st.selectbox(
            "Select a dataset for EDA:",
            list(st.session_state.uploaded_files.keys()),
            key="dataset_selectbox"  # Unique key
        )

        if st.session_state.selected_dataset:
            data = st.session_state.uploaded_files[st.session_state.selected_dataset]
            num_rows = st.number_input(
                "Number of rows to display in preview:", 
                min_value=1, max_value=len(data), value=5, 
                key="num_rows"  # Unique key
            )
            display_dataset_preview(data, num_rows)

            # Perform EDA
            handle_data_exploration(data)
    else:
        st.info("Upload CSV files to proceed.")