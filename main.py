import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Stock Trend Prediction")

# User input for stock ticker
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", "AAPL")

# Download stock data
def load_data(ticker):
    try:
        df = yf.download(ticker)
        if df.empty:
            st.error("No data found for the given ticker. Please check the ticker symbol.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data(ticker)

st.write("### Raw Data")
st.write(df.tail())

# Prepare data for Prophet
if 'Date' not in df.columns:
    df.reset_index(inplace=True)

# Ensure proper columns and drop missing values
df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()

# Ensure 'y' is numeric
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Drop rows where 'y' is NaN
df.dropna(subset=['y'], inplace=True)

if df.shape[0] < 2:
    st.error("Insufficient data for prediction. At least 2 rows are required.")
    st.stop()

# Split data into train and test
train_size = int(len(df) * 0.8)
df_train = df[:train_size]
df_test = df[train_size:]

# Train Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe
future = m.make_future_dataframe(periods=len(df_test))

# Predict
forecast = m.predict(future)

# Display forecast
st.write("### Forecast Data")
st.write(forecast.tail())

# Plot results
fig, ax = plt.subplots()
m.plot(forecast, ax=ax)
ax.scatter(df_test['ds'], df_test['y'], color='red', label='Actual Values')
ax.legend()
st.pyplot(fig)
