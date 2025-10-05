"""
GRU-based price forecasting script.

Usage (example):
    python GRU.py --ticker BTC-USD --predict_days 30

This script:
- downloads Close prices with yfinance
- builds time-series sequences (seq_len=60)
- fits a GRU model on a time-aware train split
- evaluates on a held-out test set using MAE, RMSE, MAPE
- produces a multi-day recursive forecast and saves it to CSV
"""
import argparse
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf


def fetch_close(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"No data found for ticker: {ticker}")
    return df[['Close']].dropna()


def build_sequences(scaled_all, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled_all)):
        X.append(scaled_all[i-seq_len:i])
        y.append(scaled_all[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def evaluate(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, mape


def train_gru_from_series(data, seq_len=60, epochs=10, batch_size=64, predict_days=30, train_frac=0.8, val_frac=0.1):
    """Train GRU on a pandas Series or DataFrame with one column Close.

    Returns a dict: {"model": model, "metrics": {...}, "future_df": DataFrame, "test_df": DataFrame}
    """
    tf.random.set_seed(42)
    np.random.seed(42)

    if isinstance(data, pd.DataFrame):
        series = data['Close']
    else:
        series = data

    n = len(series)
    if n < 100:
        raise ValueError("Not enough data to train (need at least ~100 rows).")

    # prepare scaler fit on train only
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_series = series.iloc[:train_end]
    scaler = MinMaxScaler()
    scaler.fit(train_series.values.reshape(-1, 1))

    scaled_all = scaler.transform(series.values.reshape(-1, 1))
    X_all, y_all = build_sequences(scaled_all, seq_len)

    sample_target_indices = np.arange(seq_len, n)
    train_mask = sample_target_indices < train_end
    val_mask = (sample_target_indices >= train_end) & (sample_target_indices < val_end)
    test_mask = sample_target_indices >= val_end

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
    X_val = X_val.reshape((X_val.shape[0], seq_len, 1)) if len(X_val) else np.empty((0, seq_len, 1))
    X_test = X_test.reshape((X_test.shape[0], seq_len, 1)) if len(X_test) else np.empty((0, seq_len, 1))

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
        GRU(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val) if len(X_val) else None, verbose=0)

    results = {}
    if len(X_test):
        pred_test = model.predict(X_test)
        pred_test_inv = scaler.inverse_transform(pred_test)
        y_test_inv = scaler.inverse_transform(y_test)
        mae, rmse, mape = evaluate(y_test_inv, pred_test_inv)
        results['metrics'] = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
        # build test dataframe aligned with original dates
        # test target indices correspond to sample_target_indices[test_mask]
        test_target_idx = sample_target_indices[test_mask]
        test_dates = series.index[test_target_idx]
        test_df = pd.DataFrame({'Close': y_test_inv.flatten(), 'Predictions': pred_test_inv.flatten()}, index=test_dates)
        results['test_df'] = test_df
    else:
        results['metrics'] = {'mae': None, 'rmse': None, 'mape': None}
        results['test_df'] = pd.DataFrame()

    # future forecast
    last_seq = scaled_all[-seq_len:]
    seq = last_seq.reshape(1, seq_len, 1)
    future = []
    for _ in range(predict_days):
        nxt = model.predict(seq)[0]
        future.append(nxt)
        seq = np.append(seq[:, 1:, :], [[nxt]], axis=1)

    future = np.array(future)
    future_inv = scaler.inverse_transform(future).flatten()
    last_date = series.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(predict_days)]
    future_df = pd.DataFrame({"Forecast": future_inv}, index=future_dates)
    results['future_df'] = future_df
    results['model'] = model

    return results


def main(args):
    print(f"Fetching {args.ticker} data {args.start} → {args.end} ...")
    data = fetch_close(args.ticker, args.start, args.end)
    res = train_gru_from_series(data, seq_len=60, epochs=args.epochs, batch_size=args.batch_size, predict_days=args.predict_days)
    print("Metrics:", res['metrics'])
    out_csv = f"gru_forecast_{args.ticker.replace(':','_').replace('/','_')}.csv"
    res['future_df'].to_csv(out_csv)
    model_path = f"gru_model_{args.ticker.replace(':','_').replace('/','_')}.h5"
    res['model'].save(model_path)
    print(f"Saved model to {model_path} and forecast to {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRU price forecasting')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Ticker symbol (e.g. BTC-USD or TCS.NS)')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='2025-06-18', help='End date YYYY-MM-DD')
    parser.add_argument('--predict_days', type=int, default=30, help='Days to forecast')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    args = parser.parse_args()
    main(args)
