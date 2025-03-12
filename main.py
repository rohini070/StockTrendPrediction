import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

st.title('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error("⚠️ No data found for the selected stock. Try another stock.")
            st.stop()
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('✅ Data loaded successfully!')

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Stock Price with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Ensure the data is clean
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Handle missing values
df_train.dropna(inplace=True)

if df_train.shape[0] < 2:
    st.error("⚠️ Not enough valid rows for training. Try another stock or check data availability.")
    st.stop()

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
