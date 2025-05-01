import ccxt
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime

exchange = ccxt.binance()

def fetch_ohlcv(symbol='BTC/USDT', interval='1h', limit=100):
    bars = exchange.fetch_ohlcv(symbol, interval, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def analyze(df):
    df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

    recent = df[-50:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    fib = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '1.0': low
    }

    return df, fib
