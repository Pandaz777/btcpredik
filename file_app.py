import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.title("Prediksi Harga Bitcoin Harian (BTC/IDR)")

@st.cache_data
def load_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "idr", "days": "90", "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [item[1] for item in data["prices"]]
    dates = [pd.to_datetime(item[0], unit='ms') for item in data["prices"]]
    df = pd.DataFrame({"Date": dates, "Price": prices})
    return df

df = load_data()
st.line_chart(df.set_index("Date")["Price"])

# Data Preprocessing
data = df["Price"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sequence untuk LSTM
def create_dataset(dataset, seq_len=10):
    x, y = [], []
    for i in range(seq_len, len(dataset)):
        x.append(dataset[i-seq_len:i])
        y.append(dataset[i])
    return np.array(x), np.array(y)

x, y = create_dataset(scaled_data)

# Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(x.shape[1], 1)),
    LSTM(64),
    Dense(1)
])
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=10, batch_size=8, verbose=0)

# Prediksi hari berikutnya
last_seq = scaled_data[-10:]
last_seq = np.expand_dims(last_seq, axis=0)
predicted = model.predict(last_seq)
predicted_price = scaler.inverse_transform(predicted)[0][0]

st.subheader("Prediksi Harga Besok")
st.metric("Harga Prediksi", f"Rp {int(predicted_price):,}")
