import streamlit as st
import importlib

# Set page title and layout
st.set_page_config(page_title="ML Tool", layout="wide")
st.title("ML Tool")

PAGES = {
    "Preprocessing": "preprocessing_page",
    "EDA": "eda_page",
    "Classification": "classification_page",
    "Clustering": "clustering_page",
    "Time Series": "time_series_page",
    "Regression": "regression_page",
    "OCR": "ocr_page"
}

# Sidebar Navigation with Dropdown
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    list(PAGES.keys()),
    index=0,  # Default to Preprocessing
    help="Select the page you want to navigate to."
)

# Function to load and display the selected page
def load_page(page_name):
    if page_name not in PAGES:
        st.error(f"Page '{page_name}' not found.")
        return

    module_name = PAGES[page_name]
    module = importlib.import_module(module_name)
    page_function = getattr(module, module_name)
    page_function()

# Load the selected page
load_page(page)
