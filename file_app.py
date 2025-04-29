import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Fungsi untuk mengambil data harga dari CoinMarketCap
@st.cache_data
def load_data():
    try:
        # API Key CoinMarketCap yang kamu dapatkan setelah mendaftar
        api_key = "YOUR_API_KEY"  # Gantilah dengan API key kamu
        
        # URL untuk mengambil data harga Bitcoin (BTC) dalam IDR
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        
        # Headers untuk autentikasi API
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,  # API Key disertakan dalam header
        }
        
        # Parameter untuk mengambil data BTC dalam IDR
        params = {"symbol": "BTC", "convert": "IDR", "limit": 1}
        
        # Mengirim permintaan GET
        response = requests.get(url, headers=headers, params=params)
        
        # Menangani kesalahan jika ada
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Cek apakah data harga tersedia
        if "data" in data and len(data["data"]) > 0:
            prices = [item['quote']['IDR']['price'] for item in data['data']]
            times = pd.to_datetime("now", unit='s')  # Ambil waktu sekarang sebagai timestamp
            return pd.DataFrame({"Datetime": [times] * len(prices), "Price": prices})
        else:
            st.warning("Tidak ada data harga dari CoinMarketCap.")
            return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat mengambil data: {e}")
        return pd.DataFrame()

# --- Main Eksekusi ---
df = load_data()

if not df.empty:
    st.subheader("Live Chart Harga Bitcoin")
    
    # Membuat grafik menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Datetime"], df["Price"], label="Harga Bitcoin (IDR)", color="orange")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("Tidak ada data yang tersedia.")
    
# Tambahkan Sinyal Buy/Sell/Hold (Contoh sederhana)
if not df.empty:
    last_price = df["Price"].iloc[-1]
    if last_price > 600000000:  # Ambil contoh untuk harga di atas 600 juta
        signal = "SELL"
    elif last_price < 500000000:  # Ambil contoh untuk harga di bawah 500 juta
        signal = "BUY"
    else:
        signal = "HOLD"
    
    st.subheader(f"Sinyal: {signal}")
    st.write(f"Terakhir harga Bitcoin: IDR {last_price:,.2f}")
