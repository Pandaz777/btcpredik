import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("Prediksi Harga Bitcoin (IDR) + Sinyal & Analisis")

# --- Fungsi Ambil Data ---
@st.cache_data
def load_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "idr", "days": "7", "interval": "hourly"}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "prices" in data and data["prices"]:
            times = [pd.to_datetime(x[0], unit="ms").tz_localize("UTC").tz_convert("Asia/Jakarta") for x in data["prices"]]
            prices = [x[1] for x in data["prices"]]
            return pd.DataFrame({"Datetime": times, "Price": prices})
        st.warning("CoinGecko kosong, mencoba Binance...")
    except:
        st.warning("Gagal dari CoinGecko, mencoba Binance...")

    # Fallback Binance
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 168}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        times = [pd.to_datetime(int(x[0]), unit="ms").tz_localize("UTC").tz_convert("Asia/Jakarta") for x in data]
        prices = [float(x[4]) * 16000 for x in data]  # USD to IDR
        return pd.DataFrame({"Datetime": times, "Price": prices})
    except Exception as e:
        st.error(f"Binance gagal juga: {e}")
        return pd.DataFrame()

# --- Hitung Indikator ---
def compute_indicators(df):
    df["SMA20"] = df["Price"].rolling(window=20).mean()
    df["%K"] = (df["Price"] - df["Price"].rolling(14).min()) / (df["Price"].rolling(14).max() - df["Price"].rolling(14).min()) * 100
    df["%D"] = df["%K"].rolling(3).mean()
    return df

# --- Sinyal Trading ---
def generate_signal(row):
    if row["%K"] < 20 and row["%K"] > row["%D"]:
        return "Buy"
    elif row["%K"] > 80 and row["%K"] < row["%D"]:
        return "Sell"
    else:
        return "Hold"

# --- Prediksi Harga Selanjutnya ---
def predict_price(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["Price"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_60 = scaled[-60:]
    last_60 = last_60.reshape((1, 60, 1))
    pred_scaled = model.predict(last_60, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)
    return pred_price[0][0]

# --- Main Eksekusi ---
df = load_data()

if not df.empty:
    df = compute_indicators(df)
    df["Signal"] = df.apply(generate_signal, axis=1)
    predicted_price = predict_price(df)

    st.subheader("Live Chart Harga Bitcoin")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Datetime"], df["Price"], label="Harga", color="orange")
    ax.plot(df["Datetime"], df["SMA20"], label="SMA 20", color="blue")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Indikator Stochastic Oscillator")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df["Datetime"], df["%K"], label="%K", color="green")
    ax2.plot(df["Datetime"], df["%D"], label="%D", color="red")
    ax2.axhline(80, color='gray', linestyle='--')
    ax2.axhline(20, color='gray', linestyle='--')
    ax2.set_ylabel("Oscillator")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Sinyal Trading")
    st.dataframe(df[["Datetime", "Price", "Signal"]].tail(10), use_container_width=True)

    st.success(f"Prediksi harga Bitcoin selanjutnya: **Rp {int(predicted_price):,}**")
else:
    st.error("Data tidak tersedia.")
