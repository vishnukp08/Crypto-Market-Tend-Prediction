import os
import torch

# ABSOLUTELY FIRST: Mock CUDA to prevent 'Torch not compiled with CUDA enabled'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
except: pass

import pandas as pd
import numpy as np
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─── Feature Configuration (must match training exactly) ───────────────────────
TARGET = "return"  # fractional return: (next_close - close) / close
TIME_IDX = "time_idx"
GROUP_IDS = ["asset"]

# These are the 16 unknown reals the model was trained on
TIME_VARYING_UNKNOWN_REALS = [
    "open", "high", "low", "close", "volume",
    "MACD", "BB_Width", "ATR_14",
    "vader_score", "roberta_score", "sentiment_missing",
    "EMA_reversion_signal", "high_atr",
    "log_tweet_count", "tweet_spike", "MACD_extreme_flag"
]

# These are the 5 known reals (calendar features)
TIME_VARYING_KNOWN_REALS = ["month", "day", "year", "hour", "minute"]
STATIC_CATEGORICALS = ["asset"]

MAX_ENCODER_LENGTH = 120
MAX_PREDICTION_LENGTH = 5  # model trained with 5-step horizon, we use step 0 (next minute)

# ──────────────────────────────────────────────────────────────────────────────

def preprocess_data(df):
    """
    Preprocess the combined technical and sentiment data for the TFT model.
    Adds time features and time_idx. Does NOT scale — scaling is handled in TFTModule.
    """
    df = df.copy()

    # Fill any NaNs from technical indicators
    df = df.ffill().bfill()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Sort values
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    # Calendar features (always overwrite to avoid stale values)
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["day"]    = df["date"].dt.day
    df["hour"]   = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    # Contiguous time index per asset
    df[TIME_IDX] = df.groupby("asset").cumcount()

    return df


class TFTModule:
    def __init__(self, model_path="best_model.ckpt", scaler_path="scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None            # loaded from disk (may have wrong columns)
        self.live_scaler = None       # fitted on live data — the one actually used
        self.load()

    def load(self):
        """Load the model and attempt to load the saved scaler."""
        if self.model is not None:
            return

        if os.path.exists(self.model_path):
            try:
                self.model = TemporalFusionTransformer.load_from_checkpoint(
                    self.model_path,
                    map_location=torch.device("cpu")
                )
                self.model.eval()
                print("[TFT] Model loaded successfully on CPU")
            except Exception as e:
                print(f"[TFT Error] Failed to load model: {e}")
                self.model = None

        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print("[TFT] Disk scaler loaded")
            except Exception as e:
                print(f"[TFT Warning] Could not load scaler from disk: {e}")
                self.scaler = None

    def fit_live_scaler(self, features_df):
        """
        Fit a fresh StandardScaler on the live historical data.
        Called once after the candle buffer is seeded.
        This replaces the disk scaler if its columns don't match.
        """
        df = preprocess_data(features_df)

        # Ensure target/return column exists
        if TARGET not in df.columns:
            df[TARGET] = 0.0

        cols_present = [c for c in TIME_VARYING_UNKNOWN_REALS if c in df.columns]

        # Check if the disk scaler already matches
        if self.scaler is not None:
            try:
                self.scaler.transform(df[cols_present].iloc[[0]])
                self.live_scaler = self.scaler
                print(f"[TFT] Disk scaler matches live features ({len(cols_present)} cols). Using it.")
                return
            except Exception:
                print("[TFT Warning] Disk scaler column mismatch. Fitting a live scaler on historical data.")

        # Fit a new scaler on the historical data
        scaler = StandardScaler()
        scaler.fit(df[cols_present])
        self.live_scaler = scaler
        print(f"[TFT] Live scaler fitted on {len(df)} rows, {len(cols_present)} features.")
        # Sanity print: close mean should be near training BTC price
        close_idx = cols_present.index("close") if "close" in cols_present else -1
        if close_idx >= 0:
            print(f"[TFT] Scaler close mean: {scaler.mean_[close_idx]:,.2f} (sanity check)")

    def predict(self, df):
        """
        Generate predictions and XAI feature importance.
        Returns dict with 'prediction' (fractional return) and 'top_features'.
        """
        if self.model is None:
            return None

        # Use live scaler if available, else disk scaler, else skip scaling
        active_scaler = self.live_scaler if self.live_scaler is not None else self.scaler

        # Preprocess
        df_processed = preprocess_data(df)

        # Scale the unknown-real features
        cols_to_scale = [c for c in TIME_VARYING_UNKNOWN_REALS if c in df_processed.columns]
        if active_scaler is not None:
            try:
                df_processed[cols_to_scale] = active_scaler.transform(df_processed[cols_to_scale])
                # Scale the return column too (it's the target)
                if TARGET in df_processed.columns and TARGET in cols_to_scale:
                    pass  # already scaled above
            except Exception as e:
                print(f"[TFT Warning] Scaling failed: {e}. Feature values will be unscaled.")
        else:
            print("[TFT Warning] No scaler available — call fit_live_scaler() first!")

        # Ensure target column exists (fill last NaN from shift)
        if TARGET not in df_processed.columns:
            df_processed[TARGET] = 0.0
        else:
            df_processed[TARGET] = df_processed[TARGET].ffill().fillna(0.0)

        try:
            params = self.model.dataset_parameters
            params['max_encoder_length'] = MAX_ENCODER_LENGTH
            params['predict_mode'] = True

            dataset = TimeSeriesDataSet.from_parameters(params, df_processed)
            dataloader = dataset.to_dataloader(train=False, batch_size=1)

            # Predict in raw mode to get both output and interpretability
            with torch.no_grad():
                # predict() with return_x=True returns a namedtuple: (output, x)
                result = self.model.predict(dataloader, mode="raw", return_x=True)
                raw_output = result.output   # the raw model output dict
                x = result.x                # the input features dict

            # ── Get return prediction ────────────────────────────────────────
            # raw_output['prediction'] shape: [batch=1, horizon=5, quantiles=3]
            # Take horizon step 0 (next minute), quantile index 1 (median / 0.5)
            pred_scaled_return = raw_output['prediction'][0, 0, 1].item()

            # Inverse-scale the return using the active scaler (if return is in cols_to_scale)
            if active_scaler is not None and TARGET in cols_to_scale:
                return_idx = cols_to_scale.index(TARGET)
                pred_return = (pred_scaled_return * active_scaler.scale_[return_idx]
                               + active_scaler.mean_[return_idx])
            else:
                # Return is already in return space (standardised near 0)
                # Use transform_output as fallback
                try:
                    transformed = self.model.transform_output(
                        raw_output['prediction'], target_scale=x['target_scale']
                    )
                    pred_return = transformed[0, 0, 1].item()
                except Exception:
                    pred_return = pred_scaled_return

            # ── XAI: Variable Selection Weights ─────────────────────────────
            interpretation = self.model.interpret_output(raw_output, reduction="sum")
            weights = interpretation["encoder_variables"].cpu().numpy()

            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum

            feature_names = self.model.encoder_variables
            importance = {k: float(v) for k, v in zip(feature_names, weights)}
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_importance[:10]

            return {
                'prediction': pred_return,
                'top_features': top_features
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[TFT Prediction Error]: {e}")
            return None


def main():
    print("TFT Model Implementation Script")
    if os.path.exists("best_model.ckpt"):
        tft = TFTModule()
        print("Ready for predictions.")

if __name__ == "__main__":
    main()
