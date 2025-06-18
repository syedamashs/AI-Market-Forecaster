import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

st.set_page_config(page_title="Market Forecast (Crypto & Stocks)", layout="centered")
st.title("📈 Price Forecasting (Crypto & Stocks) with LSTM")

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
        with st.spinner("⏳ Fetching data and training model..."):
            df = yf.download(ticker, start="2015-01-01", end="2025-06-18")
            if df.empty or 'Close' not in df.columns:
                raise ValueError(f"No data found for ticker: {ticker}")
            data = df[['Close']].dropna()
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)

            seq_len = 60
            X, y = [], []
            for i in range(seq_len, len(scaled)):
                X.append(scaled[i-seq_len:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X, y, epochs=5, batch_size=64, verbose=0)

            split = int(len(data) * 0.8)
            valid = data[split:]
            scaled_valid = scaler.transform(valid)
            Xv = [scaled_valid[i-seq_len:i] for i in range(seq_len, len(scaled_valid))]
            Xv = np.array(Xv)

            pred = model.predict(Xv)
            pred = scaler.inverse_transform(pred)
            valid = valid[seq_len:]
            valid["Predictions"] = pred

            seq = scaled[-seq_len:]
            seq = seq.reshape(1, seq_len, 1)
            future = []
            for _ in range(predict_days):
                nxt = model.predict(seq)[0]
                future.append(nxt)
                seq = np.append(seq[:,1:,:], [[nxt]], axis=1)
            future = scaler.inverse_transform(future).flatten()
            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(predict_days)]
            future_df = pd.DataFrame({"Forecast": future}, index=future_dates)

        st.success(f"✅ Prediction complete for: {ticker}")

        st.subheader("📋 Future Predictions")
        st.dataframe(future_df.style.format({"Forecast": "₹{:,.2f}" if "NS" in ticker or ticker.startswith("^NSE") else "${:,.2f}"}))

        st.subheader("📊 Actual vs Predicted vs Future Forecast")
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(valid['Close'], label="Actual (Test)")
        ax.plot(valid['Predictions'], label="Predicted (Test)")
        ax.plot(future_df['Forecast'], '--', c='red', label="Forecast Future")
        ax.axvline(last_date, linestyle=':', color='gray', label="Forecast Start")
        ax.set_title(f"{ticker} Forecast — Next {predict_days} Days", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price " + ("₹" if "NS" in ticker or ticker.startswith("^NSE") else "$"))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
