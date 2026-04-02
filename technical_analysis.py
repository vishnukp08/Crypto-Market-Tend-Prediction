import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """
    Calculates basic technical indicators for a DataFrame with OHLCV data.
    """
    df = df.copy()
    
    # Ensure numeric and sort by date
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date')

    close = df['close']

    # 1. EMAs (used for reversion signal)
    df['EMA_7'] = close.ewm(span=7, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    
    # Return / Price Change
    df['return'] = close.pct_change().fillna(0)

    # 2. MACD (Hypothesis 6: Extreme values mean high volatility)
    EMA_12 = close.ewm(span=12, adjust=False).mean()
    EMA_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA_12 - EMA_26

    # 3. Bollinger Bands (BB_Width)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Width'] = (2 * bb_std) / (bb_mid + 1e-9)

    # 4. ATR_14 (Volatility)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - close.shift()).abs()
    low_close = (df['low'] - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    return df


def get_features(df):
    """
    Final feature selection based on the 7 hypotheses from the notebook.
    Input df should have: ['date', 'open', 'high', 'low', 'close', 'volume', 'vader_score', 'roberta_score', 'tweet_count']
    """
    # 1. Base Indicators
    df = calculate_technical_indicators(df)
    
    # 2. Derived Hypothesis-based Features
    # H1: EMA Reversion Signal (Reverse of the 7-21 spread)
    df['EMA_reversion_signal'] = -(df['EMA_7'] - df['EMA_21'])
    
    # H3: High ATR (Volatility Proxy)
    # Using 70th percentile threshold from historical distribution
    if len(df) > 1:
        df['high_atr'] = df['ATR_14'] > df['ATR_14'].quantile(0.7)
    else:
        df['high_atr'] = False

    # H5: Tweet Volume & Spikes
    df['log_tweet_count'] = np.log1p(df.get('tweet_count', 0))
    if len(df) > 1:
        df['tweet_spike'] = df.get('tweet_count', 0) > df.get('tweet_count', 0).quantile(0.9)
    else:
        df['tweet_spike'] = False

    # H6: MACD Extremes
    if len(df) > 1:
        low_thresh = df['MACD'].quantile(0.1)
        high_thresh = df['MACD'].quantile(0.9)
        df['MACD_extreme_flag'] = np.where(
            (df['MACD'] <= low_thresh) | (df['MACD'] >= high_thresh),
            1, 0
        )
    else:
        df['MACD_extreme_flag'] = 0

    # 3. Sentiment & Metadata
    df['asset'] = 'BTC'
    df['return'] = (df['close'].shift(-1) - df['close']) / df['close'] # Next minute's fractional return
    df['sentiment_missing'] = np.where(df.get('tweet_count', 0) > 0, 0, 1)

    # 4. Final selection and ordering
    cols = [
        'date', 'open', 'high', 'low', 'close', 'volume', 'asset', 'return',
        'MACD', 'BB_Width', 'ATR_14', 'vader_score', 'roberta_score',
        'sentiment_missing', 'EMA_reversion_signal', 'high_atr',
        'log_tweet_count', 'tweet_spike', 'MACD_extreme_flag', 'tweet_count'
    ]
    
    # Add year, month, day, hour, minute (from date) - although they are added in tft_model_implementation, 
    # many users expect them in technical_analysis output
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    cols += ['year', 'month', 'day', 'hour', 'minute']
    
    # Ensure all columns exist (fill missing sentiment with 0)
    for col in ['vader_score', 'roberta_score', 'tweet_count']:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[cols]


if __name__ == "__main__":
    # Test with dummy data
    data = {
        'date': pd.date_range(start='2026-03-21 12:00:00', periods=50, freq='min'),
        'open': np.random.uniform(60000, 61000, 50),
        'high': np.random.uniform(61000, 62000, 50),
        'low': np.random.uniform(59000, 60000, 50),
        'close': np.random.uniform(60000, 61000, 50),
        'volume': np.random.uniform(1, 10, 50),
        'vader_score': np.random.uniform(-1, 1, 50),
        'roberta_score': np.random.uniform(-1, 1, 50),
        'tweet_count': np.random.randint(0, 100, 50)
    }
    test_df = pd.DataFrame(data)
    features = get_features(test_df)
    print("Final Output Columns:")
    print(features.columns.tolist())
    print("\nLast 5 rows:")
    print(features.tail())