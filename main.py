import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet

# Streamlit UI
st.title("Stock Trend Prediction using Prophet")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write("Original Data Preview:")
        st.write(df.head())
        
        # Debugging information
        st.write("Columns in DataFrame:", df.columns.tolist())
        
        # Check for the required columns
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("Error: CSV must contain 'ds' (date) and 'y' (target) columns.")
            st.stop()
        
        # Create a fresh DataFrame with only needed columns
        # This is a key fix - create a brand new DataFrame rather than modifying the existing one
        new_df = pd.DataFrame()
        new_df['ds'] = pd.to_datetime(df['ds'])
        
        # Convert y column explicitly to list then to Series to ensure proper format
        y_values = df['y'].values.tolist() if isinstance(df['y'], pd.Series) else list(df['y'])
        new_df['y'] = pd.Series(y_values, dtype=float)
        
        # Display debugging info
        st.write("New DataFrame structure:")
        st.write(new_df.head())
        st.write("Data types:", new_df.dtypes)
        
        # Check for invalid values
        st.write("NaN count in new df:", new_df.isna().sum())
        
        # Drop any rows with NaN values
        new_df.dropna(inplace=True)
        
        if new_df.empty:
            st.error("Error: No valid data after cleaning!")
            st.stop()
        
        # Final check before fitting Prophet
        st.write("Final DataFrame shape:", new_df.shape)
        
        # Fit the Prophet model
        m = Prophet(daily_seasonality=False)  # Adjust parameters as needed
        m.fit(new_df)
        
        # Forecast future dates
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        # Display Forecast
        st.write("Forecasted Data:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Plot Forecast
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        
        # Add components plot
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Error details:", str(e))
       
        st.write("Let's look at your raw data more closely:")
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("Column names:", raw_df.columns.tolist())
            for col in raw_df.columns:
                st.write(f"Column '{col}' - First 5 values:", raw_df[col].head().tolist())
                st.write(f"Column '{col}' - Type:", type(raw_df[col]))
        except Exception as ex:
            st.write(f"Error examining raw data: {str(ex)}")
