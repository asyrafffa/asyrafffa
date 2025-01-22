import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL, seasonal_decompose
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

def detect_datetime_columns(data):
    """Detect datetime columns in the dataset."""
    datetime_columns = []
    for col in data.columns:
        try:
            pd.to_datetime(data[col])
            datetime_columns.append(col)
        except:
            pass
    return datetime_columns

def plot_time_series(data, target_column):
    """Plot the time series data."""
    fig = px.line(data, x=data.index, y=target_column, title=f"Time Series of {target_column}")
    st.plotly_chart(fig)

def perform_stl_decomposition(data, target_column):
    """Perform STL decomposition and plot the results."""
    stl = STL(data[target_column], seasonal=13)
    result = stl.fit()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    ax1.plot(result.observed)
    ax1.set_ylabel("Observed")
    ax2.plot(result.trend)
    ax2.set_ylabel("Trend")
    ax3.plot(result.seasonal)
    ax3.set_ylabel("Seasonal")
    ax4.plot(result.resid)
    ax4.set_ylabel("Residual")
    plt.tight_layout()
    st.pyplot(fig)

def perform_classical_decomposition(data, target_column, model):
    """Perform classical decomposition and plot the results."""
    decomposition = seasonal_decompose(data[target_column], model=model, period=12)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    ax1.plot(decomposition.observed)
    ax1.set_ylabel("Observed")
    ax2.plot(decomposition.trend)
    ax2.set_ylabel("Trend")
    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel("Seasonal")
    ax4.plot(decomposition.resid)
    ax4.set_ylabel("Residual")
    plt.tight_layout()
    st.pyplot(fig)

def check_stationarity(data, target_column):
    """Check stationarity using the Augmented Dickey-Fuller test."""
    result = adfuller(data[target_column].dropna())
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    st.write(f"Critical Values: {result[4]}")
    
    if result[1] <= 0.05:
        st.success("✅ The time series is stationary.")
    else:
        st.warning("⚠️ The time series is non-stationary. Consider applying transformations.")

def apply_transformation(data, target_column, transformation):
    """Apply transformation to the target column."""
    if transformation == "Differencing":
        data[f"{target_column}_transformed"] = data[target_column].diff().dropna()
    elif transformation == "Log Transformation":
        data[f"{target_column}_transformed"] = np.log(data[target_column])
    elif transformation == "Log + Differencing":
        data[f"{target_column}_transformed"] = np.log(data[target_column]).diff().dropna()
    return data

def train_arima(train_data, test_data, target_column, order):
    """Train an ARIMA model."""
    try:
        model = ARIMA(train_data[target_column], order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_data))
        return model_fit, predictions
    except Exception as e:
        st.error(f"Error training ARIMA model: {e}")
        return None, None

def train_sarima(train_data, test_data, target_column, order, seasonal_order):
    """Train a SARIMA model."""
    try:
        model = SARIMAX(train_data[target_column], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_data))
        return model_fit, predictions
    except Exception as e:
        st.error(f"Error training SARIMA model: {e}")
        return None, None

def train_prophet(train_data, test_data, datetime_column, target_column):
    """Train a Prophet model."""
    try:
        prophet_data = train_data.reset_index().rename(columns={datetime_column: 'ds', target_column: 'y'})
        model = Prophet()
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        predictions = forecast['yhat'][-len(test_data):]
        return model, predictions
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None, None

def evaluate_model(test_data, predictions, target_column):
    """Evaluate the model using MSE and MAE."""
    mse = mean_squared_error(test_data[target_column], predictions)
    mae = mean_absolute_error(test_data[target_column], predictions)
    return mse, mae

def plot_predictions_vs_actual(test_data, predictions, target_column):
    """Plot predictions vs actual values."""
    fig = px.line()
    fig.add_scatter(x=test_data.index, y=test_data[target_column], name="Actual")
    fig.add_scatter(x=test_data.index, y=predictions, name="Predictions")
    st.plotly_chart(fig)

def time_series_page():
    st.header("⏳ Time Series Analysis")

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
        st.dataframe(data.head())

        # Detect datetime columns
        datetime_columns = detect_datetime_columns(data)
        if not datetime_columns:
            st.warning("No datetime column found in the dataset. Please ensure your data has a datetime column.")
            return

        # Select datetime and target columns
        datetime_column = st.selectbox("Select the datetime column:", datetime_columns)
        target_column = st.selectbox("Select the target column:", data.columns)

        if datetime_column and target_column:
            # Ensure the datetime column is in the correct format
            data[datetime_column] = pd.to_datetime(data[datetime_column])
            data = data.set_index(datetime_column)

            # Display the time series data
            st.subheader("Time Series Data")
            st.dataframe(data.head())

            # Plot the time series
            st.subheader("Time Series Plot")
            plot_time_series(data, target_column)

            # Seasonality Decomposition
            st.subheader("Seasonality Decomposition")
            decomposition_method = st.selectbox(
                "Choose a decomposition method:",
                ["None", "STL Decomposition", "Classical Decomposition (Additive)", "Classical Decomposition (Multiplicative)"]
            )

            if decomposition_method != "None":
                if decomposition_method == "STL Decomposition":
                    st.write("Performing STL Decomposition...")
                    perform_stl_decomposition(data, target_column)
                elif decomposition_method.startswith("Classical Decomposition"):
                    model = "additive" if "Additive" in decomposition_method else "multiplicative"
                    st.write(f"Performing {decomposition_method}...")
                    perform_classical_decomposition(data, target_column, model)

            # Stationarity Check
            st.subheader("Stationarity Check")
            if st.button("Check Stationarity"):
                check_stationarity(data, target_column)

            # Transformation Options
            st.subheader("Transformations to Achieve Stationarity")
            transformation = st.selectbox(
                "Choose a transformation:",
                ["None", "Differencing", "Log Transformation", "Log + Differencing"]
            )

            # Apply transformation
            if transformation != "None":
                data = apply_transformation(data, target_column, transformation)
                st.write("Transformed Data:")
                st.dataframe(data[[target_column, f"{target_column}_transformed"]].head())

                # Plot transformed time series
                st.subheader("Transformed Time Series Plot")
                plot_time_series(data, f"{target_column}_transformed")

                # Check stationarity of transformed data
                if st.button("Check Stationarity of Transformed Data"):
                    check_stationarity(data, f"{target_column}_transformed")

            # Data Splitting Options
            st.subheader("Data Splitting for Forecasting")
            split_method = st.selectbox(
                "Choose a data splitting method:",
                ["Single Train-Test Split", "Rolling Time Series Split (Walk-Forward Validation)"]
            )

            # Use transformed data if a transformation is applied
            target_column_transformed = f"{target_column}_transformed" if transformation != "None" else target_column

            if split_method == "Single Train-Test Split":
                train_size = st.slider("Select training data size (%):", 70, 90, 80)
                train_size = int(len(data) * (train_size / 100))
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
            else:
                n_splits = st.number_input("Number of splits for walk-forward validation:", min_value=2, value=3)
                test_size = st.number_input("Size of each test set (in time steps):", min_value=1, value=10)

                splits = []
                for i in range(n_splits):
                    train_end = len(data) - test_size * (n_splits - i)
                    test_end = train_end + test_size
                    train_data = data.iloc[:train_end]
                    test_data = data.iloc[train_end:test_end]
                    splits.append((train_data, test_data))

                st.write(f"Number of splits: {n_splits}")
                st.write(f"Test size per split: {test_size}")

            # Model selection
            st.subheader("Select a Time Series Model")
            model_options = {
                "ARIMA": train_arima,
                "SARIMA": train_sarima,
                "Prophet": train_prophet
            }
            selected_model = st.selectbox("Choose a model:", list(model_options.keys()))

            # Allow user to input ARIMA/SARIMA hyperparameters
            if selected_model == "ARIMA":
                st.subheader("ARIMA Hyperparameters")
                p = st.number_input("Enter value for p (AR term):", min_value=0, value=1)
                d = st.number_input("Enter value for d (Differencing term):", min_value=0, value=1)
                q = st.number_input("Enter value for q (MA term):", min_value=0, value=1)
                order = (p, d, q)
            elif selected_model == "SARIMA":
                st.subheader("SARIMA Hyperparameters")
                p = st.number_input("Enter value for p (AR term):", min_value=0, value=1)
                d = st.number_input("Enter value for d (Differencing term):", min_value=0, value=1)
                q = st.number_input("Enter value for q (MA term):", min_value=0, value=1)
                P = st.number_input("Enter value for P (Seasonal AR term):", min_value=0, value=1)
                D = st.number_input("Enter value for D (Seasonal Differencing term):", min_value=0, value=1)
                Q = st.number_input("Enter value for Q (Seasonal MA term):", min_value=0, value=1)
                s = st.number_input("Enter value for s (Seasonality period):", min_value=1, value=12)
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)

            # Train the model
            if st.button("Train Model"):
                if split_method == "Single Train-Test Split":
                    with st.spinner("Training model..."):
                        if selected_model == "Prophet":
                            model, predictions = model_options[selected_model](train_data, test_data, datetime_column, target_column_transformed)
                        elif selected_model == "ARIMA":
                            model, predictions = model_options[selected_model](train_data, test_data, target_column_transformed, order)
                        elif selected_model == "SARIMA":
                            model, predictions = model_options[selected_model](train_data, test_data, target_column_transformed, order, seasonal_order)

                        # Store model and predictions in session state
                        st.session_state.model_fit = model
                        st.session_state.predictions = predictions

                        # Evaluate the model
                        mse, mae = evaluate_model(test_data, predictions, target_column_transformed)
                        st.success(f"✅ Model trained successfully! MSE: {mse:.2f}, MAE: {mae:.2f}")

                        # Plot predictions vs actual
                        st.subheader("Predictions vs Actual")
                        plot_predictions_vs_actual(test_data, predictions, target_column_transformed)

                else:
                    st.write("Performing walk-forward validation...")
                    mse_scores = []
                    mae_scores = []

                    for i, (train_data, test_data) in enumerate(splits):
                        with st.spinner(f"Training and evaluating split {i + 1}/{n_splits}..."):
                            if selected_model == "Prophet":
                                model, predictions = model_options[selected_model](train_data, test_data, datetime_column, target_column_transformed)
                            elif selected_model == "ARIMA":
                                model, predictions = model_options[selected_model](train_data, test_data, target_column_transformed, order)
                            elif selected_model == "SARIMA":
                                model, predictions = model_options[selected_model](train_data, test_data, target_column_transformed, order, seasonal_order)

                            # Evaluate the model
                            mse, mae = evaluate_model(test_data, predictions, target_column_transformed)
                            mse_scores.append(mse)
                            mae_scores.append(mae)

                    # Display average evaluation metrics
                    st.success(f"✅ Walk-forward validation completed! Average MSE: {np.mean(mse_scores):.2f}, Average MAE: {np.mean(mae_scores):.2f}")

                    # Plot evaluation metrics across splits
                    st.subheader("Evaluation Metrics Across Splits")
                    metrics_df = pd.DataFrame({
                        "Split": range(1, n_splits + 1),
                        "MSE": mse_scores,
                        "MAE": mae_scores
                    })
                    st.dataframe(metrics_df)

                    fig = px.line(metrics_df, x="Split", y=["MSE", "MAE"], title="Evaluation Metrics Across Splits")
                    st.plotly_chart(fig)

            # Forecast future values (only for single train-test split)
            if split_method == "Single Train-Test Split" and "model_fit" in st.session_state:
                st.subheader("Forecast Future Values")
                future_steps = st.number_input("Enter the number of future steps to forecast:", min_value=1, value=10)
                if st.button("Forecast"):
                    if selected_model == "Prophet":
                        future = st.session_state.model_fit.make_future_dataframe(periods=future_steps)
                        forecast = st.session_state.model_fit.predict(future)
                        future_predictions = forecast['yhat'][-future_steps:]
                    else:
                        future_predictions = st.session_state.model_fit.forecast(steps=future_steps)

                    st.write("Forecasted Values:")
                    st.write(future_predictions)

                    # Plot future predictions
                    fig = px.line(x=range(len(future_predictions)), y=future_predictions, title="Future Forecast")
                    st.plotly_chart(fig)
    else:
        st.warning("No data found. Please upload a dataset to proceed.")