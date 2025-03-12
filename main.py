import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet

st.title("Stock Prediction Debug App")

# Let's test with a hardcoded stock (Google)
try:
    # Step 1: Get data directly from yfinance
    st.write("Step 1: Loading stock data...")
    stock_data = yf.download("GOOG", period="2y")
    
    # Display the raw data
    st.write("Raw data from yfinance:")
    st.write(stock_data.head())
    st.write(f"Data shape: {stock_data.shape}")
    
    # Step 2: Reset index to make Date a column
    stock_data.reset_index(inplace=True)
    
    st.write("Data after reset_index:")
    st.write(stock_data.head())
    
    # Step 3: Create a completely new and simple dataframe
    st.write("Step 3: Creating Prophet dataframe...")
    
    # Just extract the two columns we need as lists
    dates = stock_data['Date'].tolist()
    close_prices = stock_data['Close'].tolist()
    
    # Create a new empty dataframe
    prophet_data = pd.DataFrame()
    prophet_data['ds'] = dates
    prophet_data['y'] = close_prices
    
    # Display the Prophet dataframe
    st.write("Prophet input dataframe:")
    st.write(prophet_data.head())
    st.write(f"Prophet data shape: {prophet_data.shape}")
    st.write("Data types:", prophet_data.dtypes)
    
    # Step 4: Try to fit the model
    st.write("Step 4: Fitting Prophet model...")
    
    # Create model with very basic settings
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    
    # This is where the error usually happens
    st.write("About to fit model...")
    model.fit(prophet_data)
    st.write("✅ Model fitted successfully!")
    
    # Step 5: Make prediction if fit was successful
    st.write("Step 5: Making prediction...")
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # Show prediction results
    st.write("Forecast result:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Plot if everything worked
    st.write("Forecast plot:")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Error location:")
    
    # Try to provide more context on the error
    import traceback
    st.code(traceback.format_exc())
