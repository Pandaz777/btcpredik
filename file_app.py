import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(page_title="Prediksi Bitcoin", layout="wide")
st.title("Prediksi Harga Bitcoin (IDR) - Live & Analisis")

# --- Load Data dari CoinGecko ---
def load_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "idr", "days": "7", "interval": "hourly"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error("Gagal mengambil data dari API.")
        st.stop()
    
    data = response.json()
    if "prices" not in data or not data["prices"]:
        st.error("Data harga kosong!")
        st.stop()

    prices = [item[1] for item in data["prices"]]
    dates = [pd.to_datetime(item[0], unit='ms').tz_localize('UTC').tz_convert('Asia/Jakarta') for item in data["prices"]]
    df = pd.DataFrame({"Datetime": dates, "Price": prices})
    return df

df = load_data()

# --- Indikator Teknikal ---
df["SMA_20"] = df["Price"].rolling(window=20).mean()

low_14 = df["Price"].rolling(window=14).min()
high_14 = df["Price"].rolling(window=14).max()
df["%K"] = 100 * ((df["Price"] - low_14) / (high_14 - low_14))
df["%D"] = df["%K"].rolling(window=3).mean()

# --- Sinyal Buy/Sell/Hold ---
def generate_signal(row):
    if row["%K"] < 20 and row["%D"] < 20:
        return "Buy"
    elif row["%K"] > 80 and row["%D"] > 80:
        return "Sell"
    else:
        return "Hold"

df["Signal"] = df.apply(generate_signal, axis=1)

# --- Chart Interaktif dengan Plotly ---
st.subheader("Live Chart Harga Bitcoin + SMA")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Price"], name="Harga", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["SMA_20"], name="SMA 20", line=dict(color='orange', dash='dot')))
fig.update_layout(xaxis_title="Waktu", yaxis_title="Harga (IDR)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Chart Stochastic Oscillator ---
st.subheader("Stochastic Oscillator")
fig_stoch = go.Figure()
fig_stoch.add_trace(go.Scatter(x=df["Datetime"], y=df["%K"], name="%K"))
fig_stoch.add_trace(go.Scatter(x=df["Datetime"], y=df["%D"], name="%D"))
fig_stoch.update_layout(xaxis_title="Waktu", yaxis_title="Nilai", template="plotly_white")
st.plotly_chart(fig_stoch, use_container_width=True)

# --- Prediksi Harga Selanjutnya ---
st.subheader("Prediksi Harga Berikutnya (LSTM)")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Price"]].values)

# Dataset LSTM
def create_dataset(data, look_back=24):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

# Prediksi Harga Berikutnya
last_seq = scaled[-24:].reshape(1, 24, 1)
pred = model.predict(last_seq)
pred_price = scaler.inverse_transform(pred)[0][0]
st.metric("Prediksi Harga Selanjutnya (IDR)", f"{pred_price:,.0f}")

# --- Sinyal Terbaru ---
st.success(f"Sinyal Terbaru: **{df['Signal'].iloc[-1]}**")
