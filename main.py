import streamlit as st
import pandas as pd
from prophet import Prophet

# Streamlit UI
st.title("Stock Trend Prediction using Prophet")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("Original Data Preview:")
    st.write(df.head())

    # Ensure required columns are present
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("Error: CSV must contain 'ds' (date) and 'y' (target) columns.")
        st.stop()

    # Ensure 'ds' is datetime and 'y' is numeric
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['ds'] = df['ds'].dt.date  # Extract only the date part
    df['ds'] = pd.to_datetime(df['ds'])  # Convert to datetime after extracting date
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop invalid rows
    df.dropna(subset=['ds', 'y'], inplace=True)

    # Debugging: Check if DataFrame is empty after cleaning
    if df.empty:
        st.error("Error: DataFrame is empty after cleaning. Check your data!")
        st.stop()

    # Debugging: Display cleaned data types
    st.write("Data Types:")
    st.write(df.dtypes)

    # Debugging: Display sample 'ds' values
    st.write("Sample 'ds' values after conversion:", df['ds'].head())

    # Check if 'y' is a numeric Series
    if not isinstance(df['y'], pd.Series) or not pd.api.types.is_numeric_dtype(df['y']):
        st.error("Error: 'y' must be a numeric Series.")
        st.stop()

    # Fit the Prophet model (ensure df has valid 'ds' and 'y')
    try:
        m = Prophet()
        m.fit(df)
    except Exception as e:
        st.error(f"Error while fitting Prophet model: {e}")
        st.stop()

    # Forecast future dates
    try:
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
    except Exception as e:
        st.error(f"Error while forecasting: {e}")
        st.stop()

    # Display Forecast
    st.write("Forecasted Data:")
    st.write(forecast.tail())

    # Plot Forecast
    try:
        fig = m.plot(forecast)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error while plotting forecast: {e}")
        st.stop()
