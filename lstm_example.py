"""
LSTM Example, Ensuring We Use All 5000+ Lines
--------------------------------------------
CSV: pro_filtered_output.csv
Columns: timestamp, heart_rate, blood_glucose, insulin_dose

Steps:
  1) Read entire CSV (5000+ lines).
  2) Fill missing insulin_dose with 0.0 (rather than dropping).
  3) Drop rows only if heart_rate or blood_glucose is NaN or empty.
  4) Print how many lines remain after cleaning.
  5) Create sequences with window_size=2 => more samples from short intervals.
  6) Train & plot LSTM prediction with a correct x-axis index.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: if you want PSD
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------------------------------
# 1) LOAD & CLEAN DATA
# -----------------------------------------------------------------------------
def load_data(csv_path="pro_filtered_output.csv"):
    """
    Loads CSV with columns [timestamp, heart_rate, blood_glucose, insulin_dose].
    Fills missing insulin_dose with 0.0, drops rows only if heart_rate or blood_glucose is missing.
    Sorts by timestamp and returns the cleaned DataFrame.
    """
    print(f"[INFO] Reading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"[INFO] Initially read {len(df)} rows.")

    # Fill insulin_dose with 0.0 if blank
    if "insulin_dose" in df.columns:
        df["insulin_dose"] = df["insulin_dose"].fillna(0.0)
    else:
        # If for some reason 'insulin_dose' is missing, create it
        df["insulin_dose"] = 0.0
    
    # Convert columns to float if possible
    df["heart_rate"] = df["heart_rate"].astype(float, errors='ignore')
    df["blood_glucose"] = df["blood_glucose"].astype(float, errors='ignore')
    df["insulin_dose"] = df["insulin_dose"].astype(float, errors='ignore')
    
    # Replace +/- inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Only drop rows missing heart_rate or blood_glucose
    df.dropna(subset=["heart_rate","blood_glucose"], inplace=True)
    
    # Sort by timestamp
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[INFO] After cleaning, {len(df)} rows remain.")
    return df


# -----------------------------------------------------------------------------
# 2) (OPTIONAL) PSD FEATURE EXTRACTION
# -----------------------------------------------------------------------------
def compute_psd_features(hr_segment, fs=1.0):
    """
    Compute PSD for a segment of heart rate data (optional).
    """
    f, pxx = welch(hr_segment, fs=fs, nperseg=len(hr_segment))
    total_power = np.sum(pxx)
    
    # Example frequency bands
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f <= 0.4)
    
    lf_power = np.sum(pxx[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = np.sum(pxx[hf_mask]) if np.any(hf_mask) else 0.0
    peak_frequency = f[np.argmax(pxx)] if len(f) > 0 else 0.0
    
    return {
        'total_power': total_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'peak_frequency': peak_frequency
    }


# -----------------------------------------------------------------------------
# 3) CREATE SEQUENCES
# -----------------------------------------------------------------------------
def create_sequences(df, window_size=2, horizon=1, use_psd=False):
    """
    Build rolling sequences from the DataFrame.
      - Each window is 'window_size' rows of [heart_rate, blood_glucose, insulin_dose].
      - Predict 'blood_glucose' horizon steps ahead.
      - If use_psd=True, also append PSD features for heart_rate in each window.
    """
    hr_array = df['heart_rate'].values
    bg_array = df['blood_glucose'].values
    ins_array = df['insulin_dose'].values
    
    X_list, y_list = [], []

    for i in range(len(df) - window_size - horizon):
        hr_window = hr_array[i : i + window_size]
        bg_window = bg_array[i : i + window_size]
        ins_window = ins_array[i : i + window_size]

        # Target is BG horizon steps ahead
        y_target = bg_array[i + window_size + horizon - 1]

        # Base features => shape (window_size, 3)
        base_feats = np.column_stack([hr_window, bg_window, ins_window])

        if use_psd:
            psd_dict = compute_psd_features(hr_window, fs=1.0)
            psd_vals = np.array([
                psd_dict['total_power'],
                psd_dict['lf_power'],
                psd_dict['hf_power'],
                psd_dict['peak_frequency']
            ])
            # Repeat PSD across each time step => shape (window_size, 4)
            psd_window = np.tile(psd_vals, (window_size, 1))
            window_feats = np.hstack([base_feats, psd_window])
        else:
            window_feats = base_feats

        X_list.append(window_feats)
        y_list.append(y_target)
    
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"[INFO] Created {len(X)} total samples from window_size={window_size}, horizon={horizon}")
    return X, y


# -----------------------------------------------------------------------------
# 4) TRAIN LSTM MODEL
# -----------------------------------------------------------------------------
def train_lstm_model(X, y, epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Builds and trains a 2-layer LSTM. Splits data 80/20 for train/test.
    Plots predictions vs. ground truth on sample index.
    """
    num_samples, timesteps, num_features = X.shape
    
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()

    # Train-test split
    train_size = int(num_samples * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test,  y_test  = X[train_size:], y[train_size:]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    if len(y_test) > 0:
        mse = np.mean((y_test - y_pred.squeeze())**2)
        print(f"[INFO] Test MSE: {mse:.4f}")
    else:
        print("[INFO] No test samples; skipping MSE calculation.")

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(range(len(y_test)), y_test, label='True BG')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted BG')
    plt.title("Blood Glucose Prediction")
    plt.xlabel("Test Sample Index")
    plt.ylabel("BG (Scaled or Original)")
    plt.legend()
    plt.show()

    return model


# -----------------------------------------------------------------------------
# 5) MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "pro_filtered_output.csv"  # Update if needed
    df = load_data(csv_path)

    # Optional: scale [heart_rate, blood_glucose, insulin_dose] => 0..1
    scaler = MinMaxScaler()
    df[['heart_rate','blood_glucose','insulin_dose']] = scaler.fit_transform(
        df[['heart_rate','blood_glucose','insulin_dose']]
    )

    # We use a small window_size=2 to produce more samples from a large dataset
    window_size = 2
    horizon = 1
    use_psd = False

    X, y = create_sequences(df, window_size, horizon, use_psd)
    print("X shape:", X.shape, "y shape:", y.shape)

    # If for any reason you see only a few samples:
    # -> Possibly many rows had missing heart_rate/blood_glucose.
    # -> Print df.isna().sum() to see if these columns were mostly NaN.

    # Train LSTM
    model = train_lstm_model(
        X, y,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4
    )