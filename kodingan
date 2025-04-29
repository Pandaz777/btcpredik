import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io
import base64

# Fungsi untuk mengambil data historis dari Binance
def get_binance_historical(symbol="BTCUSDT", interval="1d", limit=365):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    r = requests.get(url).json()
    df = pd.DataFrame(r, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    return df[['close']]

# Fungsi untuk membuat dan melatih model LSTM
def create_lstm_model(data, seq_len=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(32),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    
    return model, scaler

# Fungsi untuk prediksi harga
def predict_price(model, scaler, data, predict_days=7, seq_len=30):
    last_seq = data[-seq_len:].reshape(1, seq_len, 1)
    pred_prices = []
    for _ in range(predict_days):
        pred_scaled = model.predict(last_seq)
        pred_prices.append(pred_scaled[0][0])
        last_seq = np.append(last_seq[:, 1:, :], [[pred_scaled]], axis=1)
    
    pred_prices_actual = scaler.inverse_transform(np.array(pred_prices).reshape(-1, 1))
    return pred_prices_actual

# Streamlit layout
st.title("Prediksi Harga Bitcoin dengan LSTM")
st.markdown("""
    Ini adalah aplikasi prediksi harga Bitcoin berdasarkan data historis menggunakan model LSTM.
    Silakan pilih jumlah hari ke depan untuk memprediksi harga Bitcoin.
""")

# Ambil data historis
st.write("Mengambil data historis Bitcoin...")
historical_data = get_binance_historical()

# Tampilkan harga Bitcoin saat ini
current_price = historical_data['close'].iloc[-1]
st.write(f"Harga Bitcoin saat ini: ${current_price:.2f}")

# Pilih jumlah hari yang ingin diprediksi
predict_days = st.slider("Jumlah Hari untuk Prediksi", min_value=1, max_value=30, value=7)

# Buat dan latih model LSTM
st.write("Membangun dan melatih model LSTM...")
model, scaler = create_lstm_model(historical_data['close'].values.reshape(-1, 1))

# Prediksi harga
predicted_prices = predict_price(model, scaler, historical_data['close'].values, predict_days=predict_days)

# Tampilkan prediksi harga
st.write(f"Prediksi Harga Bitcoin {predict_days} Hari Ke Depan:")
for i, price in enumerate(predicted_prices, start=1):
    st.write(f"Hari ke-{i}: ${price[0]:.2f}")

# Visualisasi Grafik
st.write("Visualisasi Prediksi Harga Bitcoin:")
predicted_data = np.append(historical_data['close'].values, predicted_prices)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(historical_data['close'].values, label='Harga Sebenarnya')
ax.plot(range(len(historical_data), len(predicted_data)), predicted_prices, label=f'Prediksi {predict_days} Hari', linestyle='--', color='orange')
ax.set_title("Prediksi Harga Bitcoin")
ax.set_xlabel('Hari')
ax.set_ylabel('Harga (USD)')
ax.legend()

# Simpan grafik sebagai gambar dan tampilkan di Streamlit
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_data = base64.b64encode(buf.read()).decode("utf-8")
buf.close()
st.image(f"data:image/png;base64,{img_data}", use_column_width=True)
