import os
import torch

# ABSOLUTELY FIRST: Mock CUDA to prevent 'Torch not compiled with CUDA enabled'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
except: pass

import threading
import time
import pandas as pd
import numpy as np
from fetch_candles import start_fetcher as start_candle_fetcher, get_historical_candles
from fetch_tweets import get_latest_posts
from sentiment_analysis import analyze_predict
from technical_analysis import get_features
from tft_model_implementation import TFTModule

# Shared state
candle_buffer = []  # Maintain recent candles for technical indicators
sentiment_buffer = {} # timestamp -> {vader, roberta, count}
dashboard_history = [] # For dashboard charts
last_tweet_id = None
tft_engine = TFTModule() # Initialize TFT
lock = threading.Lock()

def candle_callback(candle):
    """
    Called whenever Bybit confirms a 1m candle.
    """
    global candle_buffer
    with lock:
        # Avoid duplicates
        if candle_buffer and candle['date'] <= candle_buffer[-1]['date']:
            return
            
        candle_buffer.append(candle)
        # Keep only last 300 candles for indicator stability and 120-min lookback
        if len(candle_buffer) > 300:
            candle_buffer.pop(0)
            
        print(f"\n[Main] New Candle: {candle['date'].strftime('%Y-%m-%d %H:%M:%S')} | Price: {candle['close']}")
        process_latest_data()

def tweet_worker():
    """
    Background thread to poll Reddit and process sentiment.
    """
    global last_tweet_id, sentiment_buffer
    print("[Main] Tweet worker started...")
    
    while True:
        try:
            posts, last_tweet_id = get_latest_posts(last_tweet_id)
            if posts:
                print(f"[Main] Fetched {len(posts)} new posts. Analyzing sentiment...")
                df_posts = pd.DataFrame(posts)
                
                # Analyze sentiment in-memory
                sentiment_results = analyze_predict(df_posts)
                
                if not sentiment_results.empty:
                    with lock:
                        # Group by date (minute) and average scores
                        # sentiment_results has columns: ['date', 'text', 'compound', 'roberta_score']
                        summary = sentiment_results.groupby('date').agg({
                            'compound': 'mean',
                            'roberta_score': 'mean',
                            'text': 'count'
                        }).rename(columns={'text': 'tweet_count', 'compound': 'vader_score'})
                        
                        for dt_str, row in summary.iterrows():
                            # Convert back to timestamp for matching
                            dt = pd.to_datetime(dt_str)
                            sentiment_buffer[dt] = {
                                'vader_score': row['vader_score'],
                                'roberta_score': row['roberta_score'],
                                'tweet_count': int(row['tweet_count'])
                            }
                            # print(f"[Main] Buffered sentiment for {dt.strftime('%H:%M:%S')} | VADER: {row['vader_score']:.3f} | RoBERTa: {row['roberta_score']:.3f} | Count: {row['tweet_count']}")
                        
                        # Cleanup old sentiment buffer (older than 2 hours)
                        cutoff = pd.Timestamp.now().floor("min") - pd.Timedelta(hours=2)
                        sentiment_buffer = {k: v for k, v in sentiment_buffer.items() if k > cutoff}
            
        except Exception as e:
            print(f"[Main Tweet Error]: {e}")
            
        time.sleep(60) # Poll every minute

def process_latest_data():
    """
    Merges candles with sentiment and runs technical analysis.
    """
    if not candle_buffer:
        return
        
    # Convert buffer to DataFrame
    df = pd.DataFrame(candle_buffer)
    
    # Map sentiment to candles
    vader_scores = []
    roberta_scores = []
    tweet_counts = []
    
    for dt in df['date']:
        # Try exact match first
        s = sentiment_buffer.get(dt)
        if s is None:
            # Look back only 5 minutes to prevent "stuck" sentiment values
            # If no sentiment is found in 5 mins, it should be 0.0 (realistic sparsity)
            recent_s = [v for k, v in sentiment_buffer.items() if dt - pd.Timedelta(minutes=5) <= k <= dt]
            if recent_s:
                # Take the average of the last 5 minutes if no exact hit
                s = {
                    'vader_score': np.mean([x['vader_score'] for x in recent_s]),
                    'roberta_score': np.mean([x['roberta_score'] for x in recent_s]),
                    'tweet_count': sum([x['tweet_count'] for x in recent_s])
                }
            else:
                s = {'vader_score': 0.0, 'roberta_score': 0.0, 'tweet_count': 0}
            
        vader_scores.append(s['vader_score'])
        roberta_scores.append(s['roberta_score'])
        tweet_counts.append(s['tweet_count'])
        # if s['tweet_count'] > 0:
        #      print(f"[Main] Successfully matched sentiment for candle {dt.strftime('%H:%M:%S')}")
        
    df['vader_score'] = vader_scores
    df['roberta_score'] = roberta_scores
    df['tweet_count'] = tweet_counts
    
    try:
        # Run technical analysis feature extraction
        features_df = get_features(df)
        
        # 4. Generate TFT Prediction (Requires at least 120 candles)
        res = None
        if len(features_df) >= 120:
            res = tft_engine.predict(features_df)
            
        # Display the latest row
        latest = features_df.iloc[-1]
        current_price = latest['close']
        
        print("\n" + "═"*50)
        print(f" BTC-USDT LIVE FORECAST")
        print("═"*50)
        print(f" Time:   {latest['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Comprehensive Feature Table (excluding date-related ones)
        print(f" Price:  ${current_price:,.2f} | Volume: {latest['volume']:.4f}")
        print(f" MACD:   {latest['MACD']:.3f} | BB_Width: {latest['BB_Width']:.4f}")
        print(f" ATR:    {latest['ATR_14']:.3f} | EMA_Rev: {latest['EMA_reversion_signal']:.3f}")
        print(f" VADER:  {latest['vader_score']:+.3f} | RoBERTa: {latest['roberta_score']:+.3f}")
        print(f" Tweets: {int(latest['tweet_count'])} | Log_Cnt: {latest['log_tweet_count']:.2f}")
        
        # Boolean / Flags
        flags = []
        if latest['high_atr']: flags.append("HIGH_ATR")
        if latest['tweet_spike']: flags.append("TWEET_SPIKE")
        if latest['MACD_extreme_flag']: flags.append("MACD_EXTREME")
        if latest['sentiment_missing']: flags.append("NO_SENTIMENT")
        if flags:
            print(f" Flags:  {', '.join(flags)}")
        
        if res and 'prediction' in res:
            # We take the first step (next minute) from the prediction horizon
            predicted_return = res['prediction']
            
            target_price = current_price * (1 + predicted_return)
            
            direction = "UP" if target_price > current_price else "DOWN"
            diff = target_price - current_price
            pct_change = (diff / current_price) * 100
            
            print("─"*50)
            print(f" NEXT MINUTE PREDICTION")
            print(f" TARGET PRICE:  ${target_price:,.2f}")
            print(f" TREND:        {direction} ({diff:+,.2f} | {pct_change:+.3f}%)")
            
            if 'top_features' in res:
                print("─"*50)
                print(f" EXPLAINABLE AI (Top Drivers)")
                for feat, weight in res['top_features']:
                    print(f" • {feat:<15}: {(weight*100):.1f}% influence")
                    
            # Export Live State to JSON for Django Dashboard
            dashboard_entry = {
                "timestamp": latest['date'].isoformat(),
                "price": float(current_price),
                "target_price": float(target_price),
                "direction": direction,
                "pct_change": float(pct_change),
                "vader": float(latest['vader_score']),
                "roberta": float(latest['roberta_score']),
                "tweet_count": int(latest['tweet_count']),
                "macd": float(latest['MACD']),
                "atr": float(latest['ATR_14']),
                "volume": float(latest['volume']),
                "flags": flags,
                "top_features": [{"feature": feat, "importance": float(weight)} for feat, weight in res.get('top_features', [])]
            }
            dashboard_history.append(dashboard_entry)
            if len(dashboard_history) > 60:
                dashboard_history.pop(0)
                
            try:
                import json
                with open("dashboard_data.json", "w") as f:
                    json.dump({"live": dashboard_entry, "history": dashboard_history}, f)
            except Exception as e:
                print(f"[Dashboard Export Error]: {e}")
        else:
            print("─"*50)
            print(f" [TFT] Warming up... ({len(features_df)}/120 candles)")
            
        print("═"*50 + "\n")
        
    except Exception as e:
        print(f"[Main Process Error]: {e}")

def main():
    print("Starting Integrated Live BTC Prediction Pipeline...")
    
    # 0. Seed history
    global candle_buffer
    history = get_historical_candles(limit=200)
    if history:
        with lock:
            candle_buffer = history
            print(f"[Main] Seeded buffer with {len(candle_buffer)} historical candles.")
        
        # Fit the live scaler on historical data BEFORE first prediction
        # This fixes the scaling mismatch that causes model collapse
        try:
            seed_df = pd.DataFrame(candle_buffer)
            seed_df['vader_score'] = 0.0
            seed_df['roberta_score'] = 0.0
            seed_df['tweet_count'] = 0
            from technical_analysis import get_features
            seed_features = get_features(seed_df)
            tft_engine.fit_live_scaler(seed_features)
            print("[Main] Live scaler fitted on historical data. Predictions ready.")
        except Exception as e:
            print(f"[Main Warning] Could not fit live scaler: {e}")
    
    # 1. Start sentiment worker in background
    t = threading.Thread(target=tweet_worker, daemon=True)
    t.start()
    
    # 2. Start candle fetcher (blocking)
    try:
        start_candle_fetcher(callback=candle_callback)
    except KeyboardInterrupt:
        print("\nShutdown signal received. Closing pipeline...")

if __name__ == "__main__":
    main()
