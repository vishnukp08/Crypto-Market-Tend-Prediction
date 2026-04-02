import pandas as pd
import pytz
import os
import time
import requests

IST = pytz.timezone("Asia/Kolkata")

QUERIES = [
    # Baseline
    "bitcoin OR BTC",
    # Regulation / Law
    "(bitcoin OR crypto) (ETF OR SEC OR regulation OR ban OR ruling)",
    # Institutional / On-chain
    "(bitcoin OR BTC) (whale OR flows OR inflow OR outflow)",
    # Liquidations
    "(bitcoin OR crypto) (liquidation OR liquidated OR short squeeze OR deleveraging)",
    # Macro / Fed
    "(bitcoin OR crypto) (Fed rate OR Powell OR CPI OR inflation OR macro)",
    # Volatility Event
    "(bitcoin OR crypto) (flash crash OR why crash OR why pump)",
    # Crypto-native Event
    "(bitcoin OR crypto) (halving OR hack OR hacked OR exploit)"
]

def get_latest_posts(last_seen_id=None):
    headers = {"User-Agent": "my-crypto-bot/0.0.1"}
    keywords = [
        "btc", "bitcoin", "crypto", "bull", "bear", "pump", "dump", 
        "market", "price", "trading", "etf", "sec", "regulation", "ban", 
        "ruling", "whale", "inflow", "outflow", "liquidation", "liquidated", 
        "short", "squeeze", "deleveraging", "fed", "powell", "cpi", 
        "inflation", "macro", "crash", "halving", "hack", "exploit"
    ]

    try:
        rows = []
        for query in QUERIES:
            url = f"https://www.reddit.com/search.json?q={query}&sort=new&limit=25"
            resp = requests.get(url, headers=headers)
            time.sleep(1.0) # Rate limit protection

            if resp.status_code != 200:
                continue

            posts = resp.json().get("data", {}).get("children", [])
            
            for child in posts:
                p = child["data"]
                post_timestamp_sec = int(p.get("created_utc", 0))
                pseudo_id = post_timestamp_sec * (10 ** 6) + (abs(hash(p.get("id"))) % (10 ** 6))
                if last_seen_id and pseudo_id <= last_seen_id:
                    continue
                    
                text = (p.get("title", "") + " " + p.get("selftext", "")).lower()
                
                if not any(k in text for k in keywords):
                    continue

                # Use actual post time for better alignment
                ts = pd.to_datetime(post_timestamp_sec, unit='s', utc=True).tz_convert(IST).floor("min").tz_localize(None)

                rows.append({
                    "tweet_id": pseudo_id,
                    "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text.replace("\n", " ").replace("\r", " ")
                })

        # 🔥 2. Subreddit stream (real-time feed bypassing search cache)
        subreddits = "CryptoCurrency+Bitcoin+CryptoMarkets+BitcoinBeginners+SatoshiStreetBets"
        url_sub = f"https://api.reddit.com/r/{subreddits}/new?limit=50"
        resp_sub = requests.get(url_sub, headers=headers)
        time.sleep(1.0)
        
        if resp_sub.status_code == 200:
            posts = resp_sub.json().get("data", {}).get("children", [])
            for child in posts:
                p = child["data"]
                post_timestamp_sec = int(p.get("created_utc", 0))
                pseudo_id = post_timestamp_sec * (10 ** 6) + (abs(hash(p.get("id"))) % (10 ** 6))
                if last_seen_id and pseudo_id <= last_seen_id:
                    continue
                    
                text = (p.get("title", "") + " " + p.get("selftext", "")).lower()
                if not any(k in text for k in keywords):
                    continue

                # Use actual post time for better alignment
                ts = pd.to_datetime(post_timestamp_sec, unit='s', utc=True).tz_convert(IST).floor("min").tz_localize(None)
                rows.append({
                    "tweet_id": pseudo_id,
                    "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text.replace("\n", " ").replace("\r", " ")
                })

        if rows:
            max_id = max(r["tweet_id"] for r in rows)
            if last_seen_id is None or max_id > last_seen_id:
                last_seen_id = max_id
            
        return rows, last_seen_id

    except Exception as e:
        print("[Fetch Error]:", e)
        return [], last_seen_id
