import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf

st.title("Stock Forecast App")

st.subheader("Select dataset for prediction")
stock_symbol = st.selectbox("", ["GOOG", "AAPL", "MSFT", "AMZN"])

st.subheader("Years of prediction:")
years = st.slider("", 1, 4, 1)
period = years * 365

st.text("Loading data... done!")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_symbol)

st.subheader("Raw data")
st.dataframe(data.tail())

df_prophet = pd.DataFrame()
df_prophet['ds'] = data['Date']
df_prophet['y'] = data['Close']

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet['y'] = pd.to_numeric(df_prophet['y'])

st.subheader("Time Series data with Rangeslider")

try:
    m = Prophet()
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("Forecast data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader("Forecast chart")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error during forecasting: {e}")
    st.write("Please try a different stock or time period.")
