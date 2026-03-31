import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# Local imports
from data import get_stock_data, prepare_data
from model import build_lstm_model, train_model, predict_future

# Set Streamlit page config for better look
st.set_page_config(page_title="Stock Predictor AI", layout="wide", page_icon="📈")

def main():
    st.title("📈 AI Stock Market Predictor")
    st.markdown("""
        Predict **1 week (5 trading days)** into the future using **Deep Learning (LSTMs)**.
        Adjust the parameters in the sidebar to test across different stocks and time horizons!
    """)
    
    # Sidebar config
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y", "10y"], index=2)
    
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length (Days)", 30, 90, 60)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 10, help="More epochs = longer training time but potentially better accuracy.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64], value=32)
    
    future_days = 5  # Representing 1 week of trading days
    
    # Layout with columns
    col1, col2 = st.columns([2, 1])

    if st.sidebar.button("Train Model & Predict", type="primary"):
        with st.spinner(f"Downloading data for {ticker}..."):
            data = get_stock_data(ticker, period=period)
            
        if data.empty:
            st.error(f"Failed to retrieve data for {ticker}. Please check the ticker symbol.")
            return
            
        st.success(f"Downloaded {len(data)} trading days of data for {ticker}.")
        
        # Display Historical Data
        with col2:
            st.subheader("Recent Data")
            st.dataframe(data.tail(10))
            
        with st.spinner("Preparing data and building model..."):
            X, y, scaler, scaled_data = prepare_data(data, sequence_length, future_days)
            
            # Use all data for training
            model = build_lstm_model((X.shape[1], 1), future_days)
            
        # UI Placeholder for Training
        progress_text = "Training LSTM Model. Check your terminal for progress if running locally..."
        my_bar = st.progress(0, text=progress_text)
        
        with st.spinner(progress_text):
            model, history = train_model(model, X, y, epochs=epochs, batch_size=batch_size)
        
        my_bar.progress(100, text="Training Complete!")
        
        with st.spinner("Generating 1-week forecast..."):
            pred_scaled = predict_future(model, scaled_data, sequence_length)
            pred_prices = scaler.inverse_transform(pred_scaled)[0]
            
        # Create future dates skipping weekends
        last_date = data.index[-1]
        future_dates = []
        current_date = last_date
        while len(future_dates) < future_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
                
        # Prepare plot
        with col1:
            st.subheader("Historical Prices & Prediction")
            
            # Get last 150 days for better visualization context
            plot_data = data.tail(150).copy()
            
            fig = go.Figure()
            
            # Plot historical
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['Close'].values.flatten(),
                mode='lines',
                name='Historical Close',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # The prediction line needs to start from the last historical point to be continuous
            pred_x = [last_date] + future_dates
            # Handle Pandas Series scalar retrieval
            last_price = float(plot_data['Close'].iloc[-1])
            pred_y = [last_price] + list(pred_prices)
            
            # Plot prediction
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='lines+markers',
                name='Predicted (Next 1 Week)',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Predicted Prices 📊")
        pred_df = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
            "Predicted Close Price": np.round(pred_prices, 2)
        })
        st.table(pred_df)
        
        st.info("💡 **Disclaimer:** This tool is for educational purposes only. Machine Learning models cannot perfectly predict stock market trends. Do not use this for actual trading.")

if __name__ == "__main__":
    main()
