# Automated Trading Charts

A Streamlit web application that generates professional trading charts for any stock using free Python libraries.

## Features

- **Interactive Charts**: Choose between Plotly (interactive) and mplfinance (static) charts
- **Multiple Chart Types**: Candlestick, OHLC, and Line charts
- **Technical Indicators**: Moving averages with customizable periods
- **Volume Analysis**: Optional volume bars with color coding
- **Real-time Data**: Fetches live data from Yahoo Finance
- **Flexible Time Periods**: 1 month to 5 years of historical data
- **Stock Metrics**: Current price, daily change, volume, and 52-week high

## Installation

1. Clone or download this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Use the sidebar to:
   - Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)
   - Select time period
   - Choose chart type and indicators
   - Generate the chart

## Libraries Used

- **streamlit**: Web app framework
- **yfinance**: Yahoo Finance data fetching
- **mplfinance**: Static candlestick charts
- **plotly**: Interactive charts
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## Example Usage

```python
# Popular stock symbols to try:
# AAPL - Apple Inc.
# GOOGL - Alphabet Inc.
# TSLA - Tesla Inc.
# MSFT - Microsoft Corporation
# AMZN - Amazon.com Inc.
# NVDA - NVIDIA Corporation
```

## Chart Types

1. **Candlestick**: Traditional OHLC representation with filled/hollow candles
2. **OHLC**: Open-High-Low-Close bars
3. **Line**: Simple closing price line chart

## Technical Indicators

- **Moving Averages**: 5, 10, 20, 50, and 200-period options
- **Volume**: Color-coded volume bars (red for down days, green for up days)

## Deployment

To deploy on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repo to Streamlit Cloud
3. Deploy with one click

## License

This project is open source and available under the MIT License.