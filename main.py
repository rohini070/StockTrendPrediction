import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Streamlit UI
st.title('Stock Trend Prediction')

# User Input
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOG):", 'AAPL')

# Date Range
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if stock:
    try:
        # Downloading Stock Data
        df = yf.download(stock, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Please check the stock ticker and date range.")
            st.stop()

        st.write("### Raw Data", df.tail())

        # Prepare Data for Prophet
        df_train = df.reset_index()[['Date', 'Close']]
        df_train.columns = ['ds', 'y']

        df_train['ds'] = pd.to_datetime(df_train['ds'])

        if df_train.shape[0] < 2:
            st.error("Not enough data to train the model. Please select a wider date range.")
            st.stop()

        # Fit Prophet Model
        m = Prophet()
        m.fit(df_train)

        # Future DataFrame
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        # Display Forecast Data
        st.write("### Forecast Data", forecast.tail())

        # Plot Forecast
        fig, ax = plt.subplots()
        m.plot(forecast, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please enter a stock ticker.")
