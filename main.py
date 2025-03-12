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
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop invalid rows
    df.dropna(subset=['ds', 'y'], inplace=True)

    # Validate data after conversion
    if df.empty:
        st.error("Error: DataFrame is empty after cleaning. Check your data!")
        st.stop()

    st.write("Cleaned Data Preview:")
    st.write(df.head())

    # Fit the Prophet model
    m = Prophet()
    m.fit(df)

    # Forecast future dates
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Display Forecast
    st.write("Forecasted Data:")
    st.write(forecast.tail())

    # Plot Forecast
    fig = m.plot(forecast)
    st.pyplot(fig)
