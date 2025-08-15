# Stock-Prediction
# 📈 Stock Market Insights & Price Prediction

An interactive **Streamlit web application** that provides in-depth stock information and predicts future stock prices using **Auto Regression**.  
The app fetches real-time market data, historical prices, and visualizes both **past trends** and **future forecasts** to assist in investment decisions.

---

## 🚀 Features
- **Stock Information Dashboard**:
  - Displays basic company details, market data, financial performance, analyst targets, and more.
  - Fetches live stock data from Yahoo Finance.
- **Price Prediction Module**:
  - Uses **Auto Regression (AR)** to forecast stock prices for the next 90 days.
  - Visualizes historical, test, and forecast data in interactive charts.
- **Customizable Inputs**:
  - Choose stock from **BSE/NSE**.
  - Select custom time period and interval for historical data.
- **Interactive Charts** with Plotly:
  - Candlestick chart for historical data.
  - Line charts for training, test, predictions, and future forecasts.

---

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Data Source:** [Yahoo Finance API via yfinance](https://pypi.org/project/yfinance/)  
- **ML Model:** Auto Regression (statsmodels)  
- **Visualization:** Plotly, Pandas  

---

