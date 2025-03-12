import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf

st.title("Stock Forecast App")

with st.sidebar:
    st.header("Upload Data or Select Stock")
    uploaded_file = st.file_uploader("Upload CSV with stock data", type=["csv"])
    stock_options = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    selected_stock = st.selectbox("Or select a stock:", stock_options)
    years = st.slider("Years of prediction:", 1, 4, 1)
    period = years * 365

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Select dataset for prediction")
    st.write(f"Using uploaded data")
else:
    st.subheader("Select dataset for prediction")
    st.write(f"Selected: {selected_stock}")
    data = yf.download(selected_stock, period="5y")
    data.reset_index(inplace=True)

st.text("Loading data... done!")

st.subheader("Raw data")
st.dataframe(data.head())

try:
    df_prophet = pd.DataFrame()
    if 'Date' in data.columns:
        df_prophet['ds'] = pd.to_datetime(data['Date'])
    elif 'ds' in data.columns:
        df_prophet['ds'] = pd.to_datetime(data['ds'])
    else:
        df_prophet['ds'] = pd.to_datetime(data.index)
    if 'Close' in data.columns:
        df_prophet['y'] = data['Close'].astype(float)
    elif 'y' in data.columns:
        df_prophet['y'] = data['y'].astype(float)
    else:
        st.error("No 'Close' or 'y' column found in data")
        st.stop()

    st.subheader("Time Series data with Rangeslider")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet['ds'], df_prophet['y'])
    ax.set_title('Stock Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    with st.spinner("Training model..."):
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("Forecast data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader("Forecast chart")
    fig2 = m.plot(forecast)
    st.pyplot(fig2)

    st.subheader("Forecast components")
    fig3 = m.plot_components(forecast)
    st.pyplot(fig3)

except Exception as e:
    st.error(f"Error in processing data: {e}")
    st.write("Please ensure your data contains valid date and numeric values.")
