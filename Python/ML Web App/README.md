# Streamlit Machine Learning Web App

A multi-page Streamlit web application for end-to-end machine learning tasks, including data preprocessing, exploratory data analysis (EDA), classification, clustering, time series analysis, regression, and OCR (Optical Character Recognition). This project is designed to showcase my skills in building interactive and user-friendly machine learning applications.

---

## Features

The application is divided into multiple pages, each dedicated to a specific machine learning task:

1. **Preprocessing**:
   - Handle missing values, outliers, and data normalization.
   - Encode categorical variables using OneHotEncoder or LabelEncoder.
   - Scale features using StandardScaler or MinMaxScaler.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize data distributions, correlations, and trends.
   - Generate interactive plots using Plotly, Seaborn, and Matplotlib.
   - Perform statistical analysis (e.g., t-tests, Shapiro-Wilk test).

3. **Classification**:
   - Train and evaluate classification models (e.g., Random Forest, SVM, XGBoost).
   - Perform hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
   - Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

4. **Clustering**:
   - Apply clustering algorithms (e.g., K-Means, DBSCAN, Agglomerative Clustering).
   - Visualize clusters using PCA for dimensionality reduction.
   - Evaluate clustering performance using silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.

5. **Time Series Analysis**:
   - Decompose time series data into trend, seasonality, and residuals.
   - Forecast using ARIMA, SARIMAX, and Prophet models.
   - Evaluate forecasts using metrics like RMSE and MAE.

6. **Regression**:
   - Train and evaluate regression models (e.g., Linear Regression, Random Forest, XGBoost).
   - Perform feature selection and hyperparameter tuning.
   - Evaluate models using metrics like RMSE and R².

7. **OCR (Optical Character Recognition)**:
   - Extract text from images using PaddleOCR.
   - Display extracted text and bounding boxes.

---

## Installation

1. **Clone the repository**

2. **Setup Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
  ```
  pip install -r requirements.txt
  ```

4. **Run the streamlit app**
  ```
  streamlit run app.py
  ```

5. **Open your browser and navigate to http://localhost:8501.**

## Usage

1. **Upload your data:**
   - Use the file uploader to upload a CSV file
   - The app supports common data formats for preprocessing and analysis.

2. **Navigate through pages:**
   - Use the sidebar to switch between different pages (e.g., Preprocessing, EDA, Classification).

3. **Interact with visualizations:**
   - Explore interactive plots and customize them using available options.

4. **Train and evaluate models:**
   - Select algorithms, tune hyperparameters, and evaluate model performance.

5. **Perform OCR:**
   - Upload an image and extract text using the OCR functionality.

## Project Structure

```
streamlit-ml-web-app/
├── app.py                  # Main Streamlit application
├── pages/                  # Streamlit pages for each feature
├── preprocessing.py        # Preprocessing page
├── eda.py                  # EDA page
├── classification.py       # Classification page
├── clustering.py           # Clustering page
├── time_series.py          # Time series page
├── regression.py           # Regression page
├── ocr.py                  # OCR page
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
