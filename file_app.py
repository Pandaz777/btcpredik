import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Bitcoin", layout="centered")
st.title("Prediksi Harga Bitcoin Harian (BTC/IDR)")

# --- Ambil Data Bitcoin per Jam, 7 Hari ---
@st.cache_data
def load_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "idr", "days": "7", "interval": "hourly"}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [item[1] for item in data["prices"]]
    # Ubah waktu UTC ke WIB (Asia/Jakarta)
    dates = [pd.to_datetime(item[0], unit='ms').tz_localize('UTC').tz_convert('Asia/Jakarta') for item in data["prices"]]
    df = pd.DataFrame({"Datetime": dates, "Price": prices})
    return df

# --- Tampilkan Chart Real-Time dari CoinGecko ---
df = load_data()

# --- Tambahkan SMA ---
df["SMA_7"] = df["Price"].rolling(window=7).mean()
df["SMA_24"] = df["Price"].rolling(window=24).mean()

st.subheader("Harga Bitcoin (7 Hari Terakhir - Per Jam) dengan Moving Average")
st.line_chart(df.set_index("Datetime")[["Price", "SMA_7", "SMA_24"]])

# --- Hitung Stochastic Oscillator ---
low_14 = df["Price"].rolling(window=14).min()
high_14 = df["Price"].rolling(window=14).max()
df["%K"] = (df["Price"] - low_14) / (high_14 - low_14) * 100
df["%D"] = df["%K"].rolling(window=3).mean()

# --- Tampilkan Chart Stochastic Oscillator ---
st.subheader("Stochastic Oscillator (Momentum Indicator)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Datetime"], df["%K"], label="%K (Fast)")
ax.plot(df["Datetime"], df["%D"], label="%D (Slow)", linestyle="--")
ax.axhline(80, color="red", linestyle="--", label="Overbought (80)")
ax.axhline(20, color="green", linestyle="--", label="Oversold (20)")
ax.set_ylim(0, 100)
ax.set_ylabel("Stochastic %")
ax.legend()
st.pyplot(fig)

# --- Normalisasi Data Harga ---
scaler = MinMaxScaler()
data = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

# --- Buat Dataset untuk LSTM ---
def create_sequences(data, seq_len=24):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(x), np.array(y)

SEQ_LEN = 24
x, y = create_sequences(data, SEQ_LEN)

# --- Model LSTM Canggih ---
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, 1)))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=20, batch_size=8, verbose=0)

# --- Prediksi Harga Jam Berikutnya ---
last_seq = data[-SEQ_LEN:]
last_seq = np.expand_dims(last_seq, axis=0)
predicted = model.predict(last_seq)
predicted_price = scaler.inverse_transform(predicted)[0][0]

# --- Sinyal Trading LSTM ---
current_price = df["Price"].values[-1]
price_change = ((predicted_price - current_price) / current_price) * 100

if price_change > 1:
    lstm_signal = "BUY ✅"
    lstm_color = "green"
elif price_change < -1:
    lstm_signal = "SELL ❌"
    lstm_color = "red"
else:
    lstm_signal = "HOLD ⚖️"
    lstm_color = "gray"

# --- Sinyal Trading dari Stochastic Oscillator ---
last_k = df["%K"].iloc[-1]
last_d = df["%D"].iloc[-1]

if last_k > 80 and last_d > 80:
    stoch_signal = "SELL ❌ (Overbought)"
    stoch_color = "red"
elif last_k < 20 and last_d < 20:
    stoch_signal = "BUY ✅ (Oversold)"
    stoch_color = "green"
else:
    stoch_signal = "HOLD ⚖️"
    stoch_color = "gray"

# --- Tampilkan Hasil Prediksi ---
st.subheader("Prediksi Harga Jam Berikutnya")
st.metric("Harga Prediksi", f"Rp {int(predicted_price):,}", delta=f"{price_change:.2f}%")

# --- Sinyal ---
st.markdown("---")
st.subheader("Sinyal Trading Berdasarkan Prediksi LSTM")
st.markdown(f"<h2 style='color:{lstm_color};'>{lstm_signal}</h2>", unsafe_allow_html=True)

st.subheader("Sinyal Trading Berdasarkan Stochastic Oscillator")
st.markdown(f"<h3 style='color:{stoch_color};'>{stoch_signal}</h3>", unsafe_allow_html=True)

st.caption(f"Dibanding harga sekarang: Rp {int(current_price):,} (WIB)")

# --- TradingView Chart Langsung (Live) ---
st.markdown("---")
st.subheader("Chart Langsung BTC/IDR (TradingView - INDODAX)")

tradingview_widget = """
<div class="tradingview-widget-container">
  <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_12345&symbol=INDODAX:BTCIDR&interval=60&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=F1F3F6&studies=[]&theme=light&style=1&timezone=Asia%2FJakarta" width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
</div>
"""
components.html(tradingview_widget, height=520)
