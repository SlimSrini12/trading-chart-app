"""
Example usage of the enhanced trading chart with market structure analysis
"""

# Popular stocks to test with:
EXAMPLE_SYMBOLS = [
    "AAPL",   # Apple - Tech stock with clear trends
    "TSLA",   # Tesla - Volatile with strong swings
    "SPY",    # S&P 500 ETF - Market index
    "GOOGL",  # Google - Large cap tech
    "NVDA",   # NVIDIA - AI/Chip stock
    "META",   # Meta - Social media
    "AMZN",   # Amazon - E-commerce
    "MSFT",   # Microsoft - Enterprise tech
]

# Recommended settings for different analysis types:

# For swing trading (medium-term analysis):
SWING_TRADING_SETTINGS = {
    "time_period": "6 Months",
    "swing_strength": 5,
    "sr_min_touches": 3,
    "chart_type": "Candlestick"
}

# For day trading (short-term analysis):
DAY_TRADING_SETTINGS = {
    "time_period": "1 Month", 
    "swing_strength": 3,
    "sr_min_touches": 2,
    "chart_type": "Candlestick"
}

# For long-term investment analysis:
INVESTMENT_SETTINGS = {
    "time_period": "2 Years",
    "swing_strength": 10,
    "sr_min_touches": 4,
    "chart_type": "Candlestick"
}

print("Enhanced Trading Charts - Market Structure Analysis")
print("=" * 50)
print("\nFeatures included:")
print("✅ Swing Point Identification (HH, HL, LL, LH)")
print("✅ Support & Resistance Levels")
print("✅ Trend Line Analysis")
print("✅ Market Structure Classification")
print("✅ Interactive Plotly Charts")
print("✅ Technical Indicators")
print("\nRecommended symbols to test:", ", ".join(EXAMPLE_SYMBOLS))