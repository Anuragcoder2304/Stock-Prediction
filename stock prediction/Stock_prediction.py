# Imports
import plotly.graph_objects as go
import streamlit as st
from helper import *

# Configure the Streamlit page
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
)

##### Sidebar Start #####

st.sidebar.markdown("## **User Input Features**")


# Fetch and cache stock data
@st.cache_data
def get_stocks():
    return fetch_stocks()

stock_dict = get_stocks()

# Stock selection
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

# Stock exchange selection
st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# Build stock ticker symbol
stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"

# Display ticker
st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input("Stock ticker code", value=stock_ticker, disabled=True)

# Fetch periods and intervals
periods = fetch_periods_intervals()

# Period selection
st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

# Interval selection
st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])

##### Sidebar End #####


##### Title #####
st.markdown("# **Stock Price Prediction**")
st.markdown("##### **Make Smarter Investment Decisions with Data-Driven Forecasting**")

##### Historical Data #####

# Fetch historical stock data
stock_data = fetch_stock_history(stock_ticker, period, interval)

# Show historical chart
st.markdown("## **Historical Data**")
fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
        )
    ]
)
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)


##### Stock Prediction #####
with st.spinner("Generating stock forecast..."):
    train_df, test_df, forecast, predictions = generate_stock_prediction(stock_ticker)

st.markdown("## **Stock Prediction**")

if (
    train_df is not None
    and forecast is not None
    and predictions is not None
    and (forecast >= 0).all()
    and (predictions >= 0).all()
):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=train_df.index,
                y=train_df["Close"],
                name="Train",
                mode="lines",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=test_df.index,
                y=test_df["Close"],
                name="Test",
                mode="lines",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x=test_df.index,
                y=predictions,
                name="Test Predictions",
                mode="lines",
                line=dict(color="green"),
            ),
            go.Scatter(
                x=forecast.index,
                y=forecast,
                name="Forecast (90 days)",
                mode="lines",
                line=dict(color="red"),
            ),
        ]
    )

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Sorry! No forecast data available for the selected stock.")

##### End #####
