# Bitcoin Live Prediction Pipeline: Deep-Dive Technical Guide

This document provides a comprehensive technical breakdown of the Live Prediction system, outlining the algorithmic flow, architectural design justifications, and the engineering solutions implemented to handle the complexities of real-time financial forecasting.

---

## ✨ Key Features of the Prediction System
The pipeline integrates several state-of-the-art technologies and custom engineering solutions to deliver institutional-grade Bitcoin forecasts:

*   **Multi-Modal Data Fusion**: Seamlessly integrates high-frequency OHLCV market data with unstructured social sentiment from Reddit to capture both technical and psychological market drivers.
*   **Dual-Model Sentiment Engine**: Combines the lightning speed of **VADER** (lexicon-based) with the deep semantic understanding of **RoBERTa** (Transformer-based) for a bias-resistant sentiment consensus.
*   **State-of-the-Art Forecasting (TFT)**: Utilizes the **Temporal Fusion Transformer**, a model specifically designed for time-series forecasting that excels at capturing long-range temporal dependencies.
*   **Explainable AI (XAI)**: Features built-in interpretability that reveals the "Top 10 Drivers" for every prediction, offering transparency into the model's decision-making process.
*   **Dynamic Live Scaling**: Implements a proprietary **LiveScaler** that fits to current market volatility in real-time, preventing "Model Collapse" during extreme price regimes.
*   **Asynchronous High-Performance Pipeline**: A multithreaded architecture that protects the time-critical market data ingestion from the computational overhead of heavy transformer-based NLP processing.
*   **Historical Seeding (Warm-Start)**: Automatically backfills 200 minutes of data at boot to ensure technical indicators (EMA/MACD) are stable and accurate from the very first prediction.
*   **Strategic Signal Filtering**: Uses a sophisticated volatility-based threshold (`Tch`) to filter out market noise and focus only on high-confidence directional signals.

---

## 🔄 Step-by-Step System Workflow
The following sequence outlines the end-to-end execution of the predictive pipeline, from initialization to live signal delivery:

1.  **System Initialization**: Load the pre-trained Temporal Fusion Transformer (`best_model.ckpt`) and initialize thread-safe data buffers for market and sentiment features.
2.  **Historical Seeding**: Fetch the last 200 minutes of OHLCV data via Bybit REST API to eliminate "Start-up Lag" and ensure technical indicators (EMA/MACD) are mathematically stable.
3.  **Real-Time Data Collection**: Open a low-latency WebSocket stream for 1-minute Bitcoin candles and launch an asynchronous background worker to poll live Reddit discussions.
4.  **Sentiment Processing**: Perform linguistic cleaning (slang expansion, noise removal) and generate consensus sentiment scores using a combined VADER and RoBERTa NLP engine.
5.  **Technical Feature Engineering**: Generate high-precision technical indicators (MACD, Bollinger Bands, ATR) and derive complex features like EMA Reversion to detect market over-extensions.
6.  **Data Alignment**: Synchronize the latest sentiment metrics with the confirmed candle data by timestamp, ensuring the model sees a unified snapshot of the market state.
7.  **Dynamic Live Scaling**: Fit a `StandardScaler` to the current seeding window to normalize features, ensuring the model remains accurate across any price regime (e.g., $100k+ BTC).
8.  **TFT Inference Engine**: Feed the 120-minute normalized sequence into the Transformer to forecast the directional return for the next 1-minute interval.
9.  **XAI & Signal Delivery**: Apply Explainable AI (XAI) to extract variable selection weights and deliver the final prediction (UP/DOWN) along with its "Top 10 Drivers" to the dashboard.

## 1. Data Seeding & Sequence Initialization
**Process**: At startup, the system performs a high-speed fetch of the last 200 minutes of historical OHLCV data from the Bybit V5 REST API.

**Design Justification & Details**:
*   **Indicator Convergence**: Indicators like the Exponential Moving Average (EMA) and MACD are recursive. Starting from a single data point causes "Start-up Lag." A 200-minute seeding ensures these indicators have converged to a high degree of mathematical accuracy before the first prediction is made.
*   **Sequence Integrity**: The Temporal Fusion Transformer (TFT) requires a 120-minute lookback sequence (`MAX_ENCODER_LENGTH`). Seeding ensures that the very first live candle has a 100% complete historical context, eliminating the need for zero-padding or "warming up" during live trading.
*   **Live Scaler Fitting**: This 200-minute window serves as the calibration dataset for the `LiveScaler`, allowing the system to adjust to the current price level (e.g., $100k vs $30k) immediately upon launch.
*   **Reliability**: Using REST API for seeding provides a "snapshot" of the recent past, bridging the gap before the real-time WebSocket stream takes over.

---

## 2. Market Pipeline (Real-Time Ingestion)
**Process**: A persistent, low-latency WebSocket connection subscribes to the `kline.1.BTCUSDT` topic, delivering confirmed 1-minute OHLCV candles.

**Design Justification & Details**:
*   **Confirmed Candle Logic**: The system only processes candles where `confirm=True`. This prevents "repainting"—a common error where a prediction is made based on a mid-minute price that ultimately changes by the time the candle closes.
*   **Multi-Indicator Synergy**:
    *   **MACD & Signal**: Identifies momentum shifts and trend exhaustions.
    *   **Bollinger Band Width**: Quantifies volatility "squeezes" which often precede explosive price movements.
    *   **ATR (Average True Range)**: Acts as a dynamic noise filter, helping the model distinguish between genuine trends and random price oscillations.
    *   **EMA Reversion**: A custom "stretch" indicator that measures the distance between 7-min and 21-min EMAs, signaling potential mean-reversion opportunities.
*   **Fault Tolerance**: The pipeline includes a callback-based architecture that can gracefully handle WebSocket disconnects and resume data flow without crashing the central orchestrator.

---

## 3. Sentiment Pipeline (Advanced NLP)
**Process**: Real-time polling of Reddit using targeted search queries (Regulation, ETF news, Whale movements) followed by a dual-model sentiment scoring engine.

**Design Justification & Details**:
*   **Targeted Information Extraction**: Rather than just "Bitcoin", the system searches for specific volatility drivers like "SEC", "ban", "Fed rate", and "liquidations" across high-signal subreddits.
*   **Dual-Model Consensus Architecture**:
    *   **VADER (Lexicon-Based)**: Optimized for social media shorthand and explicit emotional keywords. It is extremely fast and provides a baseline "polarity" score.
    *   **RoBERTa (Transformer-Based)**: A deep learning model that understands semantic nuance, sarcasm, and complex financial linguistic patterns that simple word counts miss.
*   **Advanced Cleaning & Normalization**:
    *   **Slang Expansion**: Converts "HODL" to "Hold", "REKT" to "Ruined", and "ATH" to "All Time High", bridging the gap between social slang and pre-trained NLP vocabularies.
    *   **Negation Preservation**: Correctly handles phrases like "Not a pump," which simple bag-of-words models often misclassify as positive.
*   **Sparsity Management**: If no new posts are found in a 5-minute window, sentiment is decayed to 0.0, preventing "stuck" sentiment values from misleading the model during quiet market periods.

---

## 4. Forecasting Pipeline (TFT & XAI)
**Process**: Merged market and sentiment data is scaled and fed into the Temporal Fusion Transformer for multi-horizon forecasting.

**Design Justification & Details**:
*   **Architecture (Temporal Fusion Transformer)**:
    *   **Variable Selection Networks (VSN)**: Most models struggle with "noisy" features. VSN allows the TFT to automatically ignore sentiment or specific indicators if they lack predictive power at the current moment.
    *   **Temporal Self-Attention**: Identifies repeating cyclical patterns in Bitcoin's 1-minute price action over the last 2 hours.
*   **Dynamic Live Scaling**:
    *   **Non-Stationarity Management**: Bitcoin prices are not stationary. A static scaler fails when prices move significantly. The Live Scaler ensures the model sees a "normalized" version of the current price regime.
    *   **Sanity Checking**: Every time the scaler fits, it confirms the "Close Price" mean falls within reasonable bounds, preventing erroneous spikes from corrupting the model inputs.
*   **Explainable AI (XAI)**:
    *   **Variable Weights**: The system extracts the attention weights for each of the 21 input features, displaying the "Top Drivers" in the terminal. This provides the user with an "Audit Trail" for every prediction.

---

## 📂 Key Functional Modules & Architectural Components

The system is architected as a set of interconnected functional modules, each responsible for a distinct phase of the live prediction lifecycle:

### 📡 1. Market Data Fetching Module
*   **Purpose**: Ensures a continuous, low-latency stream of Bitcoin price action.
*   **Key Functions**:
    *   **WebSocket Streamer**: Maintains a persistent connection to the Bybit exchange, handling real-time "Confirmed Candle" events to prevent mid-minute data noise.
    *   **Historical Seeder**: Executes REST API calls during startup to backfill the last 200 minutes of data, ensuring all technical indicators are "hot" and accurate immediately.
    *   **Auto-Reconnect Logic**: Monitors heartbeat signals and automatically re-establishes connections during network interruptions.

### 🌐 2. Social Sentiment Harvester
*   **Purpose**: Scrapes the "Mood of the Market" from high-signal social channels.
*   **Key Functions**:
    *   **Reddit Poller**: Targeted scraping of major subreddits (r/Bitcoin, r/CryptoCurrency) and specific keyword searches (e.g., "SEC", "ETF", "Pump").
    *   **Asynchronous Worker**: Runs on a dedicated background thread to ensure that the heavy computational load of data scraping never delays the time-sensitive market data pipeline.
    *   **Keyword Strategy**: Utilizes a diverse set of queries to capture macro-events, regulatory news, and retail FOMO simultaneously.

### 🧠 3. NLP Analysis Engine
*   **Purpose**: Translates raw human text into structured mathematical sentiment scores.
*   **Key Functions**:
    *   **Linguistic Cleaner**: A multi-stage regex pipeline that handles social media noise (URLs, emojis, special characters) while preserving sentiment-critical punctuation.
    *   **Dual-Model Consensus**: Runs every post through both **VADER** (rule-based) and **RoBERTa** (transformer-based) models to generate a balanced, consensus-driven sentiment polarity.
    *   **Slang Normalization**: A custom dictionary expansion that translates "crypto-speak" (HODL, REKT, MOON) into standardized English for the AI models.

### 📈 4. Technical Feature Generator
*   **Purpose**: Converts raw Price/Volume data into predictive technical signals.
*   **Key Functions**:
    *   **Momentum & Volatility Engine**: Calculates high-precision MACD, Bollinger Bands, and ATR values in real-time.
    *   **Hypothesis Builder**: Computes complex derived features like **EMA Reversion** (detecting over-extensions) and **Price-Volume Divergence**.
    *   **State Buffer**: Maintains a rolling 21-period memory window to ensure indicators remain stable and consistent across every 1-minute step.

### 🔮 5. TFT Forecasting Core
*   **Purpose**: The "Brain" of the project; generates the final directional forecasts.
*   **Key Functions**:
    *   **Temporal Fusion Transformer**: A state-of-the-art model that uses self-attention to identify which historical moments in the last 2 hours are most relevant to the next minute's price.
    *   **Dynamic Live Scaler**: Recalibrates the entry data on every launch to ensure the model can handle any price regime (e.g., $100k+ BTC) without "Model Collapse."
    *   **XAI Interpreter**: Extracts internal attention weights to explain the "Top 10 Drivers" behind every prediction, providing transparency to the user.

### 🏗️ 6. Central Execution Pipeline
*   **Purpose**: Orchestrates the entire system and manages the user interface.
*   **Key Functions**:
    *   **Pipeline Synchronizer**: Ensures that sentiment data and market data are perfectly aligned by timestamp before being sent to the model.
    *   **Threading Manager**: Safely handles the parallel execution of data fetching, analysis, and forecasting.
    *   **Terminal Dashboard**: Updates the real-time UI every minute with price levels, sentiment shifts, and model confidence.

---

## 🧪 Experiment Setup & Training Workflow
The project follows a hybrid development cycle to balance real-time execution with heavy deep-learning training requirements.

### 💻 1. Local Development (VS Code)
*   **Role**: Pipeline orchestration, real-time data ingestion, and live dashboard management.
*   **Execution**: Modules like `Main.py` and `fetch_candles.py` are optimized for long-running stability on local hardware.
*   **Output**: The final production environment where live predictions are served.

### ☁️ 2. High-Performance Training (Google Colab)
*   **Role**: Model training and hyperparameter optimization.
*   **Details**: Historically, the **Temporal Fusion Transformer (TFT)** was trained on Google Colab using high-RAM GPU environments (T4/A100).
*   **Generated Artifacts**:
    *   **`best_model.ckpt`**: The optimized weights of the trained model.
    *   **`scaler.pkl`**: The historical data distribution used for initial baseline normalization.

---

## 📊 Performance Metrics & Strategy Logic
We evaluate the model using both standard regression metrics and specialized trading performance indicators.

### 📉 Standard Performance Metrics
*   **MAE (Mean Absolute Error)**: Measures the average magnitude of the prediction errors without considering their direction.
*   **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily, reflecting the model's sensitivity to market volatility.
*   **R2 Score**: Provides an indication of how well the features explain the variance in Bitcoin's 1-minute returns.
*   **Directional Accuracy**: Calculated as `np.mean(direction_true == direction_pred)`, this metric tracks how often the model correctly identifies whether the next candle will be **UP** or **DOWN**.

### 💰 Trading Strategy & Expectancy
The model results are converted into a tradable strategy using a **Volatility Threshold** to filter out low-confidence signals:

*   **Tch (Signal Threshold)**: Set at `0.3368`. A trade signal (1 or -1) is only generated if the predicted return exceeds `std(pred_return) * 0.3368`.
*   **Win/Loss Rate**: Monitors the percentage of profitable signals versus losing signals.
*   **Expectancy Per Trade**: Calculated as `(Win Rate * Avg Win) - (Loss Rate * Avg Loss)`.
    *   **Significance**: A positive expectancy is the ultimate proof of a profitable trading system, representing the average amount you expect to "win" on every signal generated.

---

## 🔍 Results Interpretation (XAI)
The power of the Temporal Fusion Transformer lies in its **Interpretability** through Explainable AI.

### 🤖 Variable Importance using XAI (TFT) & Interactive Explanations
*   **Variable Selection Networks (VSN)**: At every 1-minute step, the system extracts the internal "Attention Weights" to dynamically identify the strongest predictive features.
*   **Visualizing Drivers**: The dashboard displays a responsive ranking of the **Top Features** influencing the current forecast's direction.
*   **Interactive Glossary System**: Every dimension of the time-series model is mapped in an interactive, on-click dashboard glossary. Key definitions established in the tool include:
    *   `roberta_score` / `vader_score`: Captures pure semantic emotions (raw vs complex) mapped directly to panic/hype volatility.
    *   `macd` / `bb_width` / `atr_14`: Momentum trend flags pointing towards convergence, divergence, and potential volatility expansions.
    *   *Time / Meta Features*: `hour`, `day`, and `minute` cyclic variables giving the model baseline structural anchors to expect specific geographic market-open liquidity spikes.
*   **Trust & Audit**: Users can click directly on the XAI horizontal bars in the UI to instantly display a 3-point bulleted breakdown defining exactly what the variable is and why the Transformer architecture is currently focused on it.

---

## ⚡ Technical Challenges & Engineering Solutions

### 📈 Challenge 1: Model Collapse (Near-Zero Predictions)
*   **Cause**: Static scalers becoming obsolete due to Bitcoin's extreme price volatility relative to historical training data.
*   **Solution**: Implemented a **Dynamic Live Scaler** that recalibrates on every system restart using the 200-minute seed window, ensuring model inputs are always "in-range."

### 🧵 Challenge 2: Processing Bottlenecks (Synchronous Lag)
*   **Cause**: The RoBERTa transformer model is computationally intensive, sometimes taking >1 second per batch on CPU, which would cause the WebSocket thread to hang.
*   **Solution**: Moved the entire Sentiment Pipeline to a **Dedicated Background Thread**, using a thread-safe `Lock` and buffer to pass sentiment scores to the main Market loop without blocking.

### 🧩 Challenge 3: Hardware Divergence (CUDA/CPU)
*   **Cause**: `pytorch-forecasting` dependencies frequently default to CUDA, leading to runtime errors on systems without NVIDIA GPUs.
*   **Solution**: Created an **Environment Mocking Module** at the start of `Main.py` that intercepts Torch calls to simulate a CPU-only environment, ensuring the pipeline runs stably on any Windows/Linux machine.
