import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

###############################################################################
# 1) UTILITY: FORWARD-FILL (OR INTERPOLATE) MISSING DATA
###############################################################################
def fill_missing_values(df):
    """
    Keep more rows by forward-filling heart_rate and blood_glucose. 
    Optionally, you could do 'interpolate(method="time")' instead of ffill().
    """
    df = df.sort_values("timestamp").copy()
    df.set_index("timestamp", inplace=True)

    # Forward-fill heart_rate and blood_glucose
    df["heart_rate"] = df["heart_rate"].ffill()
    df["blood_glucose"] = df["blood_glucose"].ffill()

    # If you prefer time interpolation:
    # df["heart_rate"] = df["heart_rate"].interpolate(method="time")
    # df["blood_glucose"] = df["blood_glucose"].interpolate(method="time")

    df.reset_index(inplace=True)
    return df

###############################################################################
# 2) LOAD & CLEAN DATA
###############################################################################
def load_data(csv_path="carbs_and_insulin_included.csv"):
    """
    Reads CSV with columns:
      [creationDate, heart_rate, blood_glucose, insulin_dose, dietary_carbohydrates]
    Fills insulin_dose / dietary_carbohydrates with 0, forward-fills HR/BG,
    and then drops any rows still missing both HR/BG.
    """
    print(f"[INFO] Reading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["creationDate"])
    print(f"[INFO] Initially read {len(df)} rows.")

    # Rename creationDate -> timestamp
    df.rename(columns={"creationDate":"timestamp"}, inplace=True)

    # Ensure columns exist, fill missing insulin/carbs with 0
    for col in ["insulin_dose", "dietary_carbohydrates"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Convert relevant columns to numeric
    numeric_cols = ["heart_rate", "blood_glucose", "insulin_dose", "dietary_carbohydrates"]
    for nc in numeric_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # Replace ±inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward-fill (or interpolate) missing heart_rate / blood_glucose
    df = fill_missing_values(df)

    # Drop rows where heart_rate or blood_glucose is still NaN
    df.dropna(subset=["heart_rate", "blood_glucose"], how='any', inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] After cleaning, {len(df)} rows remain.")
    return df

###############################################################################
# 3) TRANSFORMATIONS: ACTIVE INSULIN & OPERATIVE CARBS
###############################################################################
def compute_active_insulin(df, insulin_col="insulin_dose", time_col="timestamp", decay_rate=0.028):
    """
    Simple demonstration of a 'active_insulin' feature by exponential decay.
    Each new insulin dose adds to the previous 'active_insulin' value.
    """
    df = df.copy()
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    active_insulin = [0.0]
    last_time = df[time_col].iloc[0]
    for i in range(1, len(df)):
        current_time = df[time_col].iloc[i]
        dt_hours = (current_time - last_time).total_seconds() / 3600.0

        # Decay from previous AI
        prev_ai = active_insulin[-1] * np.exp(-decay_rate * dt_hours)

        # Add new insulin dose
        new_ai = prev_ai + df[insulin_col].iloc[i]
        active_insulin.append(new_ai)

        last_time = current_time

    df["active_insulin"] = active_insulin
    return df

def compute_operative_carbs(df, carbs_col="dietary_carbohydrates", time_col="timestamp"):
    """
    Simplistic 'operative_carbs' approach. Each new carb intake 
    adds to the previous operative_carbs, which decays exponentially.
    """
    df = df.copy()
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    operative_carbs = [0.0]
    last_time = df[time_col].iloc[0]
    DECAY_RATE = 0.028  # same or different from insulin if you prefer
    for i in range(1, len(df)):
        current_time = df[time_col].iloc[i]
        dt_hours = (current_time - last_time).total_seconds() / 3600.0

        # Decay from previous
        prev_oc = operative_carbs[-1] * np.exp(-DECAY_RATE * dt_hours)

        # Add new carbs
        new_oc = prev_oc + df[carbs_col].iloc[i]
        operative_carbs.append(new_oc)
        last_time = current_time

    df["operative_carbs"] = operative_carbs
    return df

###############################################################################
# 4) CREATE SEQUENCES
###############################################################################
def create_sequences(df, window_size=8, horizon=1):
    """
    Build input sequences of shape (window_size, features) -> predict BG horizon steps ahead.
    Features: heart_rate, blood_glucose, active_insulin, operative_carbs
    """
    hr_array  = df["heart_rate"].values
    bg_array  = df["blood_glucose"].values
    ai_array  = df["active_insulin"].values
    oc_array  = df["operative_carbs"].values

    X_list, y_list = [], []
    for i in range(len(df) - window_size - horizon):
        hr_window = hr_array[i : i + window_size]
        bg_window = bg_array[i : i + window_size]
        ai_window = ai_array[i : i + window_size]
        oc_window = oc_array[i : i + window_size]

        # Target: BG at (i + window_size + horizon - 1)
        y_target = bg_array[i + window_size + horizon - 1]

        window_feats = np.column_stack([hr_window, bg_window, ai_window, oc_window])
        X_list.append(window_feats)
        y_list.append(y_target)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"[INFO] Created {len(X)} samples (window_size={window_size}, horizon={horizon})")
    return X, y

###############################################################################
# 5) TRAIN BiLSTM MODEL
###############################################################################
def train_lstm_model(X, y, epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Build and train a 2-layer BiLSTM to predict BG. Splits data 80/20 for train/test.
    """
    num_samples, timesteps, num_features = X.shape

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), 
                            input_shape=(timesteps, num_features)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    model.summary()

    # Split
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

    # Prediction
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred.squeeze())**2)
    rmse = np.sqrt(mse)
    print(f"[INFO] Test MSE:  {mse:.4f}")
    print(f"[INFO] Test RMSE: {rmse:.4f}")

    # Plot results
    if len(y_test) > 0:
        plt.figure(figsize=(10,4))
        plt.plot(range(len(y_test)), y_test, label='True BG')
        plt.plot(range(len(y_pred)), y_pred, label='Predicted BG')
        plt.title("Blood Glucose Prediction (Test Set)")
        plt.xlabel("Test Sample Index")
        plt.ylabel("BG (scaled)")
        plt.legend()
        plt.show()

    return model

###############################################################################
# 6) MAIN
###############################################################################
if __name__ == "__main__":
    csv_path = "carbs_and_insulin_included.csv"
    df = load_data(csv_path)  # Step 1: Load & Clean

    # Step 2: Compute transformations
    df = compute_active_insulin(df, insulin_col="insulin_dose", time_col="timestamp", decay_rate=0.028)
    df = compute_operative_carbs(df, carbs_col="dietary_carbohydrates", time_col="timestamp")

    # Step 3: Scale the relevant columns
    columns_to_scale = ["heart_rate","blood_glucose","active_insulin","operative_carbs"]
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Step 4: Create sequences
    window_size = 8   # how many past timesteps per sample
    horizon = 1       # how far ahead to predict
    X, y = create_sequences(df, window_size=window_size, horizon=horizon)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("[WARN] Not enough data after cleaning to form sequences.")
    else:
        # Step 5: Train LSTM
        train_lstm_model(
            X, y,
            epochs=10,
            batch_size=32,
            learning_rate=1e-4
        )