import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta

# import our model runners
from LSTM import train_lstm_from_series
from GRU import train_gru_from_series
# Prophet module (may raise ImportError if not installed)
try:
    from Prophet import train_prophet_from_serikes
    PROPHET_AVAILABLE = True
except Exception:
    train_prophet_from_series = None
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="Market Forecast (Crypto & Stocks)", layout="centered")
st.title("📈 Price Forecasting (Crypto & Stocks) with LSTM, GRU, PROPHET")

# Step 1: Choose market type
data_type = st.radio("Select Market Type:", ["Cryptocurrency", "Stock (USD/INR)"])

ticker = ""
if data_type == "Cryptocurrency":
    user = st.text_input("Enter Crypto Symbol (e.g. BTC, ETH, DOGE):", "BTC").upper()
    ticker = user + "-USD"
else:
    # Give examples for Indian/NSE stocks
    user = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA or TCS.NS, INFY.NS, ^NSEI):", "TCS.NS")
    ticker = user.upper()

predict_days = st.slider("Forecast Days:", 7, 90, 30)

if st.button("🔍 Predict"):
    try:
        with st.spinner("⏳ Fetching data and training models..."):
            df = yf.download(ticker, start="2015-01-01", end="2025-06-18")
            if df.empty or 'Close' not in df.columns:
                raise ValueError(f"No data found for ticker: {ticker}")
            data = df[['Close']].dropna()

            # Train LSTM and GRU
            lstm_res = train_lstm_from_series(data, seq_len=60, epochs=5, batch_size=64, predict_days=predict_days)
            gru_res = train_gru_from_series(data, seq_len=60, epochs=10, batch_size=64, predict_days=predict_days)

            # Train Prophet if available (graceful fallback)
            prophet_res = None
            if PROPHET_AVAILABLE:
                try:
                    prophet_res = train_prophet_from_series(data, predict_days=predict_days)
                except Exception as pe:
                    prophet_res = None
                    st.warning(f"Prophet model failed: {pe}")
            else:
                st.info("Prophet not installed; skipping Prophet model. Install 'prophet' package to enable it.")

        st.success(f"✅ Prediction complete for: {ticker}")

        # Metrics comparison
        lstm_metrics = lstm_res.get('metrics', {})
        gru_metrics = gru_res.get('metrics', {})
        rows = [
            {
                'Model': 'LSTM',
                'MAE': lstm_metrics.get('mae'),
                'RMSE': lstm_metrics.get('rmse'),
                'MAPE (%)': lstm_metrics.get('mape')
            },
            {
                'Model': 'GRU',
                'MAE': gru_metrics.get('mae'),
                'RMSE': gru_metrics.get('rmse'),
                'MAPE (%)': gru_metrics.get('mape')
            }
        ]
        if prophet_res is not None:
            prop_metrics = prophet_res.get('metrics', {})
            rows.append({
                'Model': 'Prophet',
                'MAE': prop_metrics.get('mae'),
                'RMSE': prop_metrics.get('rmse'),
                'MAPE (%)': prop_metrics.get('mape')
            })
        metrics_df = pd.DataFrame(rows)

        st.subheader("📋 Model Comparison (Test Metrics)")
        st.dataframe(metrics_df.style.format({
            'MAE': '{:,.4f}',
            'RMSE': '{:,.4f}',
            'MAPE (%)': '{:,.2f}'
        }))

        # Prepare future forecasts side-by-side
        future_lstm = lstm_res['future_df'].rename(columns={'Forecast': 'LSTM_Forecast'})
        future_gru = gru_res['future_df'].rename(columns={'Forecast': 'GRU_Forecast'})
        future_combined = future_lstm.join(future_gru, how='outer')
        if prophet_res is not None and 'future_df' in prophet_res:
            future_prophet = prophet_res['future_df'].rename(columns={'Forecast': 'Prophet_Forecast'})
            future_combined = future_combined.join(future_prophet, how='outer')

        st.subheader("📋 Future Predictions (Both Models)")
        currency_fmt = "₹{:,.2f}" if "NS" in ticker or ticker.startswith("^NSE") else "${:,.2f}"
        fmt_map = {'LSTM_Forecast': currency_fmt, 'GRU_Forecast': currency_fmt}
        if 'Prophet_Forecast' in future_combined.columns:
            fmt_map['Prophet_Forecast'] = currency_fmt
        st.dataframe(future_combined.style.format(fmt_map))

        # Plot actual vs predictions vs future forecasts
        st.subheader("📊 Actual vs Predicted vs Future Forecast")
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot test actual + predictions if available
        if 'test_df' in lstm_res and not lstm_res['test_df'].empty:
            ax.plot(lstm_res['test_df']['Close'], label='Actual (Test)', color='black')
            ax.plot(lstm_res['test_df']['Predictions'], label='LSTM Predicted (Test)', color='blue')
        if 'test_df' in gru_res and not gru_res['test_df'].empty:
            ax.plot(gru_res['test_df']['Predictions'], label='GRU Predicted (Test)', color='green')
        if prophet_res is not None and 'test_df' in prophet_res and not prophet_res['test_df'].empty:
            ax.plot(prophet_res['test_df']['Predictions'], label='Prophet Predicted (Test)', color='orange')

        # Plot future forecasts (dashed)
        if 'LSTM_Forecast' in future_combined.columns:
            ax.plot(future_combined['LSTM_Forecast'], '--', color='blue', label='LSTM Forecast')
        if 'GRU_Forecast' in future_combined.columns:
            ax.plot(future_combined['GRU_Forecast'], '--', color='green', label='GRU Forecast')
        if 'Prophet_Forecast' in future_combined.columns:
            ax.plot(future_combined['Prophet_Forecast'], '--', color='orange', label='Prophet Forecast')

        last_date = data.index[-1]
        ax.axvline(last_date, linestyle=':', color='gray', label='Forecast Start')
        ax.set_title(f"{ticker} Forecast — Next {predict_days} Days", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price " + ("₹" if "NS" in ticker or ticker.startswith("^NSE") else "$"))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
