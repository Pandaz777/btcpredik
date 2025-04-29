import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Prediksi Bitcoin", layout="wide")
st.title("Prediksi Harga Bitcoin (IDR) - Live & Analisis")

# --- Ambil Data ---
@st.cache_data
def load_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "idr", "days": "7", "interval": "hourly"}
    res = requests.get(url, params=params)
    data = res.json()
    if "prices" not in data:
        return None
    times = [pd.to_datetime(x[0], unit="ms").tz_localize("UTC").tz_convert("Asia/Jakarta") for x in data["prices"]]
    prices = [x[1] for x in data["prices"]]
    return pd.DataFrame({"Datetime": times, "Price": prices})

df = load_data()
if df is None or df.empty:
    st.error("Gagal mengambil data.")
    st.stop()

# --- Tambahkan Indikator ---
df["SMA_20"] = df["Price"].rolling(20).mean()
low_14 = df["Price"].rolling(14).min()
high_14 = df["Price"].rolling(14).max()
df["%K"] = 100 * (df["Price"] - low_14) / (high_14 - low_14)
df["%D"] = df["%K"].rolling(3).mean()

def signal(row):
    if row["%K"] < 20 and row["%D"] < 20:
        return "Buy"
    elif row["%K"] > 80 and row["%D"] > 80:
        return "Sell"
    return "Hold"
df["Signal"] = df.apply(signal, axis=1)

# --- Grafik Harga ---
st.subheader("Grafik Harga Bitcoin")
fig, ax = plt.subplots()
ax.plot(df["Datetime"], df["Price"], label="Harga")
ax.plot(df["Datetime"], df["SMA_20"], label="SMA 20", linestyle="--")
ax.set_ylabel("Harga (IDR)")
ax.legend()
st.pyplot(fig)

# --- Grafik Stochastic ---
st.subheader("Stochastic Oscillator")
fig2, ax2 = plt.subplots()
ax2.plot(df["Datetime"], df["%K"], label="%K")
ax2.plot(df["Datetime"], df["%D"], label="%D")
ax2.axhline(80, color="red", linestyle="--")
ax2.axhline(20, color="green", linestyle="--")
ax2.legend()
st.pyplot(fig2)

# --- Prediksi LSTM ---
st.subheader("Prediksi Harga Berikutnya")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Price"]].dropna())

def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=5, batch_size=16, verbose=0)

last_input = scaled[-24:].reshape(1, 24, 1)
pred = model.predict(last_input)
pred_price = scaler.inverse_transform(pred)[0][0]

st.metric("Prediksi Harga Berikutnya", f"Rp {pred_price:,.0f}")
st.info(f"Sinyal Terbaru: **{df['Signal'].iloc[-1]}**")
