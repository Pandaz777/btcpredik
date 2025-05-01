import streamlit as st
import trading_bot
import altair as alt
import pandas as pd

st.set_page_config(layout='wide')
st.title("Bot Trading Crypto - Streamlit Dashboard")

symbol = st.selectbox("Pilih Pair", ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
interval = st.selectbox("Interval", ['1h', '4h', '1d'])

with st.spinner("Mengambil data..."):
    df = trading_bot.fetch_ohlcv(symbol, interval)
    df, fib = trading_bot.analyze(df)

st.subheader("Grafik Harga + SMA")
chart = alt.Chart(df).mark_line().encode(
    x='timestamp',
    y='close'
).properties(height=400)

sma50 = alt.Chart(df).mark_line(color='orange').encode(x='timestamp', y='sma50')
sma200 = alt.Chart(df).mark_line(color='red').encode(x='timestamp', y='sma200')

st.altair_chart(chart + sma50 + sma200, use_container_width=True)

st.subheader("Indikator RSI")
st.line_chart(df.set_index('timestamp')['rsi'])

st.subheader("Level Fibonacci")
for k, v in fib.items():
    st.write(f"{k}: {v:.2f}")
