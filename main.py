import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error

# Page config
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

# App title
st.title("Stock Trend Prediction")

# Sidebar
st.sidebar.header("Settings")

# Option 1: Allow user to input ticker symbol
option = st.sidebar.selectbox(
    'Select stock ticker:',
    ('AAPL', 'GOOG', 'MSFT', 'AMZN', 'META')
)

# Option 2: Allow user to upload own data
uploaded_file = st.sidebar.file_uploader("Or upload your own stock data (CSV)", type=["csv"])

# Date range for forecasting
forecast_days = st.sidebar.slider("Forecast days ahead:", 1, 365, 30)

# Main content area
if uploaded_file is not None:
    # User uploaded their own data
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader(f"Uploaded Data Preview")
        st.dataframe(df.head())
        
        # Check required columns
        required_cols = ['Date', 'Close']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        else:
            # Prepare data
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            
            # Get the target column for forecasting
            close_data = df['Close']
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
else:
    # Use yfinance to get data
    try:
        # Get data for the past 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        # Download data
        df = yf.download(option, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data found for {option}")
            st.stop()
            
        st.subheader(f"{option} Stock Data")
        st.dataframe(df.head())
        
        # Get the target column for forecasting
        close_data = df['Close']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# Display stock chart
st.subheader("Historical Stock Price")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
                   row_width=[0.2, 0.8])

fig.add_trace(
    go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Candlestick"),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=df.index, y=df['Volume'], name="Volume"),
    row=2, col=1
)

fig.update_layout(
    height=600,
    title_text=f"{option} Stock Data",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# Add forecasting functionality
st.subheader("Stock Price Forecast")

with st.spinner("Calculating forecast..."):
    # Split the data into training and testing sets
    train_size = int(len(close_data) * 0.9)
    train, test = close_data[:train_size], close_data[train_size:]
    
    # Default ARIMA parameters
    p, d, q = 5, 1, 0
    
    try:
        # Fit ARIMA model
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        
        # Make prediction on test data
        predictions = model_fit.forecast(steps=len(test))
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        # Forecast future values
        future_forecast = model_fit.forecast(steps=forecast_days)
        
        # Create future dates for forecasting
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
        forecast_df.set_index('Date', inplace=True)
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(close_data.index[-365:], close_data.values[-365:], label='Historical Data')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
        plt.fill_between(forecast_df.index, 
                         forecast_df['Forecast'] - forecast_df['Forecast'].std(), 
                         forecast_df['Forecast'] + forecast_df['Forecast'].std(), 
                         color='red', alpha=0.2)
        plt.title(f'{option} Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
        # Show forecast data
        st.subheader("Forecast Data")
        st.dataframe(forecast_df)
        
        # Display RMSE
        st.info(f"Model RMSE on test data: {rmse:.2f}")
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")

# Add some metrics
try:
    # Calculate some metrics
    st.subheader("Stock Metrics")
    col1, col2, col3 = st.columns(3)
    
    # Current price
    current_price = close_data.iloc[-1]
    prev_price = close_data.iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
    
    # 50-day moving average
    ma_50 = close_data.rolling(window=50).mean().iloc[-1]
    ma_50_prev = close_data.rolling(window=50).mean().iloc[-2]
    ma_50_change = ma_50 - ma_50_prev
    ma_50_change_pct = (ma_50_change / ma_50_prev) * 100
    
    col2.metric("50-Day MA", f"${ma_50:.2f}", f"{ma_50_change_pct:.2f}%")
    
    # 200-day moving average
    ma_200 = close_data.rolling(window=200).mean().iloc[-1]
    ma_200_prev = close_data.rolling(window=200).mean().iloc[-2]
    ma_200_change = ma_200 - ma_200_prev
    ma_200_change_pct = (ma_200_change / ma_200_prev) * 100
    
    col3.metric("200-Day MA", f"${ma_200:.2f}", f"{ma_200_change_pct:.2f}%")
except Exception as e:
    st.warning(f"Could not calculate metrics: {str(e)}")

# Add disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("Disclaimer: This app is for educational purposes only. Do not use for investment decisions.")
