import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import os

# Ensure the data directory exists
os.makedirs("data/preprocessing", exist_ok=True)
os.makedirs("data/model", exist_ok=True)

# Initialize session state
def initialize_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None

# Load uploaded files into session state and save them to the preprocessing folder
def load_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_files:
            # Save the uploaded file to the preprocessing folder
            file_path = os.path.join("data/preprocessing", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Load the dataset into session state
            st.session_state.uploaded_files[uploaded_file.name] = pd.read_csv(file_path)

# Display dataset preview and summary
def display_dataset_preview(data, num_rows=5):
    st.write(f"Dataset Preview (showing {num_rows} rows):")
    st.dataframe(data.head(num_rows))

    if st.button("Generate Data Summary", key="data_summary_button"):
        summary = data.describe(include='all').T
        summary['Unique Values'] = data.nunique()
        st.dataframe(summary)

# Apply filters to the dataset
def apply_filter(data, column, condition, value):
    if condition in ["contains", "does not contain"]:
        if condition == "contains":
            return data[data[column].astype(str).str.contains(value, case=False, na=False)]
        else:
            return data[~data[column].astype(str).str.contains(value, case=False, na=False)]
    else:
        query = f"`{column}` {condition} {value}"
        return data.query(query)

# Handle data type conversion
def handle_change_dtype(data, column_to_convert, new_dtype):
    try:
        if new_dtype == "datetime64":
            data[column_to_convert] = pd.to_datetime(data[column_to_convert], errors="coerce")
        else:
            data[column_to_convert] = data[column_to_convert].astype(new_dtype)
        st.success(f"Column `{column_to_convert}` converted to {new_dtype} successfully!")
    except Exception as e:
        st.error(f"Failed to convert column `{column_to_convert}` to {new_dtype}: {e}")
    return data

# Handle data cleaning
def handle_data_cleaning(data, cleaning_option, fill_value=None):
    if cleaning_option == "Remove Duplicates":
        data = data.drop_duplicates()
        st.success("Duplicates removed successfully!")
    elif cleaning_option == "Drop Missing Values":
        data = data.dropna()
        st.success("Missing values dropped successfully!")
    elif cleaning_option == "Fill Missing Values":
        data = data.fillna(fill_value)
        st.success(f"Missing values filled with `{fill_value}` successfully!")
    return data

# Handle data export
def handle_data_export(data, export_format):
    if export_format == "CSV":
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
        )
    elif export_format == "Excel":
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            data.to_excel(writer, index=False)
        st.download_button(
            label="Download Excel",
            data=excel_file.getvalue(),
            file_name="preprocessed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif export_format == "JSON":
        json_data = data.to_json(orient="records")
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="preprocessed_data.json",
            mime="application/json",
        )

# Handle data exploration and visualization
def handle_data_exploration(data):
    st.header("Data Exploration and Visualization")

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Feature Distribution", "Correlation Matrix", "Pair Plots", "Missing Data Analysis",
        "Outlier Detection", "Categorical Data Visualization"
    ])

    with tab1:
        st.subheader("Feature Distribution Plots")
        if len(numeric_columns) > 0:
            selected_column = st.selectbox("Select a numeric column to visualize:", numeric_columns)
            plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot", "Density Plot"])

            if plot_type == "Histogram":
                bins = st.slider("Number of bins:", 5, 100, 20)
                fig = px.histogram(data, x=selected_column, nbins=bins, title=f"Histogram of {selected_column}")
            elif plot_type == "Box Plot":
                fig = px.box(data, y=selected_column, title=f"Box Plot of {selected_column}")
            elif plot_type == "Density Plot":
                fig = px.density_contour(data, x=selected_column, title=f"Density Plot of {selected_column}")

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
            pair_plot_columns = st.multiselect("Select columns for pair plot:", numeric_columns, default=numeric_columns[:3])
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
            selected_column = st.selectbox("Select a numeric column for outlier detection:", numeric_columns)
            outlier_method = st.selectbox("Select outlier detection method:", ["Z-Score", "IQR"])

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
            selected_cat_column = st.selectbox("Select a categorical column:", categorical_columns)
            plot_type = st.selectbox("Select plot type:", ["Bar Plot", "Pie Chart"])

            if plot_type == "Bar Plot":
                fig = px.bar(data[selected_cat_column].value_counts(), title=f"Bar Plot of {selected_cat_column}")
            elif plot_type == "Pie Chart":
                fig = px.pie(data[selected_cat_column].value_counts(), names=data[selected_cat_column].unique(), title=f"Pie Chart of {selected_cat_column}")

            st.plotly_chart(fig)
        else:
            st.warning("No categorical columns found for visualization.")

# Main function for the preprocessing page
def preprocessing_page():
    st.header("Preprocessing")
    initialize_session_state()

    # Upload multiple datasets
    st.write("Upload your datasets (CSV format):")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    load_uploaded_files(uploaded_files)

    # Select a dataset to preprocess
    if st.session_state.uploaded_files:
        st.session_state.selected_dataset = st.selectbox(
            "Select a dataset to preprocess:",
            list(st.session_state.uploaded_files.keys())
        )

        if st.session_state.selected_dataset:
            data = st.session_state.uploaded_files[st.session_state.selected_dataset]
            num_rows = st.number_input("Number of rows to display in preview:", min_value=1, max_value=len(data), value=5)
            display_dataset_preview(data, num_rows)

            st.subheader("Preprocessing Options")
            st.write("Choose an action to apply to the dataset:")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Filter Data", key="filter_data_button"):
                    st.session_state.current_action = "filter"
            with col2:
                if st.button("Change Data Types", key="change_dtype_button"):
                    st.session_state.current_action = "change_dtype"
            with col3:
                if st.button("Drop Columns", key="drop_columns_button"):
                    st.session_state.current_action = "drop_columns"

            col4, col5, col6 = st.columns(3)
            with col4:
                if st.button("Rename Columns", key="rename_columns_button"):
                    st.session_state.current_action = "rename_columns"
            with col5:
                if st.button("Replace Values", key="replace_values_button"):
                    st.session_state.current_action = "replace_values"
            with col6:
                if st.button("Fill Up/Down", key="fill_up_down_button"):
                    st.session_state.current_action = "fill_up_down"

            col7 = st.columns(1)[0]
            with col7:
                if st.button("Data Cleaning", key="data_cleaning_button"):
                    st.session_state.current_action = "data_cleaning"

            if "current_action" in st.session_state:
                if st.session_state.current_action == "filter":
                    handle_filter(data)
                elif st.session_state.current_action == "change_dtype":
                    column_to_convert = st.selectbox("Select a column to change data type:", data.columns)
                    new_dtype = st.selectbox("Select the new data type:", ["int64", "float64", "object", "category", "bool", "datetime64"])
                    if st.button("Change Data Type", key="change_dtype_apply_button"):
                        data = handle_change_dtype(data, column_to_convert, new_dtype)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data
                elif st.session_state.current_action == "drop_columns":
                    columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
                    if st.button("Drop Columns", key="drop_columns_apply_button"):
                        data = data.drop(columns=columns_to_drop)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data
                        st.success(f"Dropped columns: {columns_to_drop}")
                elif st.session_state.current_action == "rename_columns":
                    column_to_rename = st.selectbox("Select a column to rename:", data.columns)
                    new_column_name = st.text_input("Enter the new column name:")
                    if st.button("Rename Column", key="rename_column_apply_button"):
                        data.rename(columns={column_to_rename: new_column_name}, inplace=True)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data
                        st.success(f"Column `{column_to_rename}` renamed to `{new_column_name}` successfully!")
                elif st.session_state.current_action == "replace_values":
                    replace_column = st.selectbox("Select a column to replace values:", data.columns)
                    old_value = st.text_input("Enter the value to replace:")
                    new_value = st.text_input("Enter the new value:")
                    if st.button("Replace Values", key="replace_values_apply_button"):
                        data[replace_column] = data[replace_column].replace(old_value, new_value)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data
                        st.success(f"Replaced `{old_value}` with `{new_value}` in column `{replace_column}` successfully!")
                elif st.session_state.current_action == "fill_up_down":
                    fill_column = st.selectbox("Select a column to fill:", data.columns)
                    fill_method = st.radio("Select fill method:", ["Forward Fill (Up)", "Backward Fill (Down)"])
                    if st.button("Fill Values", key="fill_values_apply_button"):
                        if fill_method == "Forward Fill (Up)":
                            data[fill_column].fillna(method="ffill", inplace=True)
                        elif fill_method == "Backward Fill (Down)":
                            data[fill_column].fillna(method="bfill", inplace=True)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data
                        st.success(f"Filled missing values in column `{fill_column}` using {fill_method} successfully!")
                elif st.session_state.current_action == "data_cleaning":
                    cleaning_option = st.radio("Select cleaning option:", ["Remove Duplicates", "Drop Missing Values", "Fill Missing Values"])
                    fill_value = None
                    if cleaning_option == "Fill Missing Values":
                        fill_value = st.text_input("Enter the value to fill missing values with:")
                    if st.button("Apply Cleaning", key="apply_cleaning_button"):
                        data = handle_data_cleaning(data, cleaning_option, fill_value)
                        st.session_state.uploaded_files[st.session_state.selected_dataset] = data

            st.subheader("Updated Dataset Preview")
            st.write(f"Showing {num_rows} rows after applying preprocessing steps:")
            st.dataframe(data.head(num_rows))

            # Data Export Section
            st.subheader("Export Preprocessed Data")
            export_format = st.selectbox("Select export format:", ["CSV", "Excel", "JSON"])
            handle_data_export(data, export_format)

            # Save preprocessed data to the classification folder
            if st.button("Save Preprocessed Data"):
                preprocessed_file_path = os.path.join("data/model", "preprocessed_data.csv")
                data.to_csv(preprocessed_file_path, index=False)
                st.success(f"Preprocessed data saved to {preprocessed_file_path}")

            handle_data_exploration(data)
    else:
        st.info("Upload CSV files to proceed.")
