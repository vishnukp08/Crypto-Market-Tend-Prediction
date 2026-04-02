import json
import ssl
import pandas as pd
import pytz
import os
import requests
from websocket import WebSocketApp

IST = pytz.timezone("Asia/Kolkata")
# CANDLE_CSV = "btc_1m_candles.csv"
last_saved_timestamp = None

def on_message(ws, message, callback=None):
    global last_saved_timestamp
    try:
        data = json.loads(message)
        if "topic" in data and "kline.1" in data["topic"]:
            k = data["data"][0]
            if not k["confirm"]:
                return
            
            ts = pd.to_datetime(k["timestamp"], unit="ms", utc=True).tz_convert(IST).floor("min").tz_localize(None)
            
            # if last_saved_timestamp and ts <= last_saved_timestamp:
            #     return
                
            candle = {
                "date": ts, # Keep as Timestamp for internal processing
                "open": float(k["open"]),
                "high": float(k["high"]),
                "low": float(k["low"]),
                "close": float(k["close"]),
                "volume": float(k["volume"]),
            }
            
            # Still save to CSV if desired, but prioritize callback
            # pd.DataFrame([candle]).to_csv(CANDLE_CSV, mode="a", header=False, index=False)
            
            # last_saved_timestamp = ts
            # print(f"[Candle] Ready: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if callback:
                callback(candle)
            
    except Exception as e:
        print("[Candle Error]:", e)

def get_historical_candles(symbol="BTCUSDT", limit=60):
    """
    Fetch historical 1m candles via Bybit REST API to seed the buffer.
    """
    print(f"Fetching last {limit} minutes of historical candles for {symbol}...")
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=1&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["retCode"] == 0:
            candles = []
            # Bybit returns candles in descending order (newest first)
            for k in reversed(data["result"]["list"]):
                ts = pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_convert(IST).floor("min").tz_localize(None)
                candles.append({
                    "date": ts,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles
        else:
            print(f"[Candle Error] API returned code {data['retCode']}: {data['retMsg']}")
    except Exception as e:
        print(f"[Candle Error] Failed to fetch history: {e}")
    return []

def on_open(ws):
    print("🚀 Bybit WebSocket connected. Listening for 1m candles...")
    ws.send(json.dumps({"op": "subscribe", "args": ["kline.1.BTCUSDT"]}))

def start_fetcher(callback=None):
    print("Starting BTC 1m OHLCV Candle Fetcher...")
    from functools import partial
    ws = WebSocketApp(
        "wss://stream.bybit.com/v5/public/spot", 
        on_open=on_open, 
        on_message=partial(on_message, callback=callback)
    )
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

if __name__ == "__main__":
    start_fetcher(lambda c: print(f"Test Callback: {c}"))
