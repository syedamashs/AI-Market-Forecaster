from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        raise ImportError("prophet package not found. Install with 'pip install prophet'")


def evaluate(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, mape


def train_prophet_from_series(data, predict_days=30, train_frac=0.8, val_frac=0.1):
    if isinstance(data, pd.DataFrame):
        series = data['Close']
    else:
        series = data

    n = len(series)
    if n < 30:
        raise ValueError('Not enough data for Prophet (need at least ~30 rows)')

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # prepare train dataframe for evaluation (trained only on train)
    ds_train = pd.to_datetime(series.index[:train_end]).to_numpy()
    y_train = series.iloc[:train_end].astype(float).to_numpy().ravel()
    train_df = pd.DataFrame({'ds': ds_train, 'y': y_train})

    # model for evaluation
    model_eval = Prophet()
    model_eval.fit(train_df)

    # actual test dates are those after val_end
    test_dates = series.index[val_end:]
    if len(test_dates) > 0:
        df_test = pd.DataFrame({'ds': pd.to_datetime(test_dates).to_numpy()})
        pred_test = model_eval.predict(df_test)
        y_pred = pred_test['yhat'].to_numpy().ravel()
        y_true = series.loc[test_dates].astype(float).to_numpy().ravel()
        mae, rmse, mape = evaluate(y_true, y_pred)
        test_df = pd.DataFrame({'Close': y_true, 'Predictions': y_pred}, index=pd.to_datetime(test_dates))
        metrics = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
    else:
        metrics = {'mae': None, 'rmse': None, 'mape': None}
        test_df = pd.DataFrame()

    # For forecasting, fit on full series for best future forecast
    full_df = pd.DataFrame({'ds': pd.to_datetime(series.index).to_numpy(), 'y': series.astype(float).to_numpy().ravel()})
    model_full = Prophet()
    model_full.fit(full_df)
    future = model_full.make_future_dataframe(periods=predict_days, freq='D')
    forecast = model_full.predict(future)

    # select only the future period after the last data index
    last_date = series.index[-1]
    future_mask = forecast['ds'] > last_date
    future_forecast = forecast.loc[future_mask, ['ds', 'yhat']].copy()
    future_forecast.set_index('ds', inplace=True)
    future_df = pd.DataFrame({'Forecast': future_forecast['yhat']})

    return {'model': model_full, 'metrics': metrics, 'test_df': test_df, 'future_df': future_df}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prophet forecasting')
    parser.add_argument('--ticker', type=str, default='BTC-USD')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2025-06-18')
    parser.add_argument('--predict_days', type=int, default=30)
    args = parser.parse_args()

    # lazy import yfinance to fetch data
    import yfinance as yf
    df = yf.download(args.ticker, start=args.start, end=args.end)
    if df.empty or 'Close' not in df.columns:
        raise SystemExit(f'No data for {args.ticker}')
    data = df[['Close']].dropna()
    res = train_prophet_from_series(data, predict_days=args.predict_days)
    print('Metrics:', res['metrics'])
    res['future_df'].to_csv(f'prophet_forecast_{args.ticker}.csv')
