import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np

st.title("Stock Trend Prediction using Prophet")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data Preview:")
    st.write(df.head())

    st.write("Column types:")
    st.write(df.dtypes)

    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("Error: CSV must contain 'ds' (date) and 'y' (target) columns.")
        st.stop()

    st.write("Sample y values:", df['y'].head(10).tolist())

    try:
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        if isinstance(df['y'], pd.Series):
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
        else:
            df['y'] = pd.to_numeric(pd.Series(df['y']), errors='coerce')

        st.write("NaN count in y column:", df['y'].isna().sum())

        df.dropna(subset=['ds', 'y'], inplace=True)

        st.write("DataFrame shape after cleaning:", df.shape)

        if df.empty:
            st.error("Error: DataFrame is empty after cleaning. Check your data!")
            st.stop()

        st.write("Cleaned Data Preview:")
        st.write(df.head())

        df_train = df[['ds', 'y']].copy().reset_index(drop=True)

        st.write("Final data types:", df_train.dtypes)
        st.write("Final data head:", df_train.head())

        m = Prophet()
        m.fit(df_train)

        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        st.write("Forecasted Data:")
        st.write(forecast.tail())

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.write("Traceback details:", str(e.__traceback__))
