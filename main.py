import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np

# App Title
st.title('Stock Forecast App')

# Date Range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Select Stock
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Prediction Period
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load Data Function
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load and Display Data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(data.tail())

# Plot Raw Data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Stock Price with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare Data for Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train the Model
m = Prophet()
m.fit(df_train)

# Make Predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display Forecast Data
st.subheader('Forecast Data')
st.write(forecast.tail())

# Forecast Plots
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
