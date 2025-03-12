import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np

st.title('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('GOOGL', 'AAPL', 'MSFT', 'TSLA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)

if data.empty:
    st.error("Failed to load stock data. Please check the ticker or try again later.")
    st.stop()

data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Stock Price with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

st.write("Missing values before cleaning:", df_train.isna().sum())
df_train.dropna(inplace=True)
st.write("Missing values after cleaning:", df_train.isna().sum())

if df_train.shape[0] < 2:
    st.error("Not enough data points for training. Please select another stock or date range.")
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
