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
        df_original = pd.read_csv(uploaded_file)
        
        st.write("### Original Data Analysis")
        st.write("First 5 rows:")
        st.write(df_original.head())
        
        st.write("Column names:", df_original.columns.tolist())
        st.write("Data types:", df_original.dtypes)
        
        # Create completely new dataframe from scratch
        # This method avoids any hidden attributes or issues with the original dataframe
        
        st.write("### Creating Prophet-compatible data")
        
        # Check if required columns exist
        if 'ds' not in df_original.columns:
            st.error("Column 'ds' not found. Please ensure your CSV has a date column named 'ds'.")
            st.stop()
            
        if 'y' not in df_original.columns:
            st.error("Column 'y' not found. Please ensure your CSV has a target column named 'y'.")
            st.stop()
        
        # Extract the columns as pure Python lists
        dates = df_original['ds'].tolist()
        values = df_original['y'].tolist()
        
        st.write("Sample dates:", dates[:5])
        st.write("Sample values:", values[:5])
        
        # Create a fresh dataframe with these lists
        prophet_df = pd.DataFrame()
        
        # Convert dates to datetime
        try:
            prophet_df['ds'] = pd.to_datetime(dates)
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            st.stop()
            
        # Convert values to float
        try:
            # Try converting each value individually
            float_values = []
            for v in values:
                try:
                    float_values.append(float(v))
                except:
                    float_values.append(np.nan)
                    
            prophet_df['y'] = float_values
        except Exception as e:
            st.error(f"Error converting values: {e}")
            st.stop()
            
        # Drop NaN values
        prophet_df.dropna(inplace=True)
        
        st.write("### Final Prophet Data")
        st.write(prophet_df.head())
        st.write("Shape:", prophet_df.shape)
        
        if prophet_df.empty:
            st.error("No valid data left after processing!")
            st.stop()
            
        # Only proceed if we have data
        if len(prophet_df) > 0:
            st.write("### Fitting Prophet Model")
            
            # Fit model
            m = Prophet()
            m.fit(prophet_df)
            
            # Forecast
            future = m.make_future_dataframe(periods=90)  # Predict 90 days
            forecast = m.predict(future)
            
            # Display results
            st.write("### Forecast Results")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
            # Plot
            fig1 = m.plot(forecast)
            st.pyplot(fig1)
            
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
            
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        
        # Add detailed debugging for investigation
        st.write("### DEBUG: Examining uploaded file")
        try:
            # Just read file as plain text
            uploaded_file.seek(0)
            content = uploaded_file.read().decode()
            st.write("First 500 characters of file:")
            st.write(content[:500])
        except Exception as ex:
            st.write(f"Error reading file content: {ex}")
