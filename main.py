import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
from datetime import date
import yfinance as yf
from neuralprophet import NeuralProphet
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
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
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

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train.dropna(inplace=True)

m = NeuralProphet()
m.fit(df_train, freq='D')

future = m.make_future_dataframe(df_train, periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name="Forecast"))
fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Actual"))
fig1.layout.update(title_text='Forecast vs Actual', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)"
