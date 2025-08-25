import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import argrelextrema
import openai
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Trading Chart with Levels",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trading Chart with Support/Resistance Levels")

# Sidebar
st.sidebar.header("Chart Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")

# AI Analysis Settings
st.sidebar.subheader("🤖 AI Analysis")
enable_ai = st.sidebar.checkbox("Enable AI Market Analysis", value=True)

# Try to get API key from environment first, then from user input
env_api_key = os.getenv("OPENAI_API_KEY")
if env_api_key and env_api_key != "your_api_key_here":
    openai_api_key = env_api_key
    st.sidebar.success("✅ API Key loaded from .env file")
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key for AI analysis")

if st.sidebar.button("ℹ️ How to get API Key"):
    st.sidebar.info("""
    **Get OpenAI API Key:**
    1. Go to platform.openai.com
    2. Sign up/Login
    3. Go to API Keys section
    4. Create new secret key
    5. Copy and paste here
    
    **Cost:** ~$0.01-0.05 per analysis
    """)

# Save API key in session state
if openai_api_key:
    st.session_state.openai_api_key = openai_api_key
elif 'openai_api_key' in st.session_state:
    openai_api_key = st.session_state.openai_api_key

# Timeframe selection for scalping
timeframe_options = {
    "1 Minute": {"interval": "1m", "period": "1d"},
    "3 Minutes": {"interval": "3m", "period": "5d"},
    "5 Minutes": {"interval": "5m", "period": "5d"},
    "15 Minutes": {"interval": "15m", "period": "5d"},
    "30 Minutes": {"interval": "30m", "period": "1mo"},
    "1 Hour": {"interval": "1h", "period": "1mo"},
    "4 Hours": {"interval": "4h", "period": "3mo"},
    "1 Day": {"interval": "1d", "period": "1y"},
    "1 Week": {"interval": "1wk", "period": "2y"}
}

selected_timeframe = st.sidebar.selectbox(
    "Timeframe", 
    list(timeframe_options.keys()), 
    index=7  # Default to 1 Day
)

# Professional Chart Settings
st.sidebar.subheader("🏆 Professional Chart Settings")

# Core Professional Levels (Always shown by default)
st.sidebar.write("**📊 Core Professional Levels:**")
st.sidebar.write("✅ Support/Resistance Levels")
st.sidebar.write("✅ Trend Lines & Channels") 
st.sidebar.write("✅ Key Price Levels")

# Optional Technical Indicators
st.sidebar.subheader("📈 Optional Technical Indicators")

# Moving Averages
show_emas = st.sidebar.multiselect(
    "📊 EMAs (Exponential Moving Averages)",
    [9, 21, 50, 200],
    default=[],
    help="Select which EMAs to display on the chart"
)

# Bollinger Bands
show_bollinger_bands = st.sidebar.checkbox(
    "📊 Bollinger Bands", 
    value=False,
    help="Show volatility bands around price"
)

# VWAP
show_vwap = st.sidebar.checkbox(
    "📊 VWAP (Volume Weighted Average Price)", 
    value=False,
    help="Show institutional trading level"
)

# Advanced Professional Levels
st.sidebar.subheader("🎯 Advanced Professional Levels")

show_fibonacci = st.sidebar.checkbox(
    "📐 Fibonacci Retracements", 
    value=True,
    help="Show key Fibonacci retracement levels"
)

show_pivot_points = st.sidebar.checkbox(
    "⚡ Pivot Points", 
    value=True,
    help="Show daily/session pivot levels"
)

show_round_numbers = st.sidebar.checkbox(
    "🎯 Psychological Levels", 
    value=True,
    help="Show round number psychological levels"
)

# Chart Customization
st.sidebar.subheader("⚙️ Chart Customization")
max_levels = st.sidebar.slider(
    "Max S/R Levels", 
    3, 15, 8,
    help="Maximum number of support/resistance levels to show"
)

def get_data(symbol, timeframe_config):
    """Get stock data with specific interval and period"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=timeframe_config["period"],
            interval=timeframe_config["interval"],
            prepost=False,
            auto_adjust=True,
            repair=True
        )
        
        # Ensure we have data and remove any NaN values
        if data.empty:
            st.error(f"No data available for {symbol}")
            return None
            
        # Clean the data
        data = data.dropna()
        
        # Debug info
        st.write(f"📊 **Data Info:** {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def find_support_resistance_levels(data, timeframe):
    """Find horizontal support and resistance levels"""
    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    current_price = closes[-1]
    
    # Adjust swing detection based on timeframe
    if "Minute" in timeframe:
        order = 2  # Very sensitive for scalping
        tolerance = 0.005  # 0.5% tolerance
    elif "Hour" in timeframe:
        order = 3
        tolerance = 0.01  # 1% tolerance
    else:
        order = 4  # Less sensitive for daily/weekly
        tolerance = 0.015  # 1.5% tolerance
    
    # Find swing points
    high_peaks = argrelextrema(highs, np.greater, order=order)[0]
    low_peaks = argrelextrema(lows, np.less, order=order)[0]
    
    # Get all significant price levels
    resistance_levels = highs[high_peaks]
    support_levels = lows[low_peaks]
    
    # Also add recent highs and lows
    recent_period = min(50, len(data) // 4)
    recent_high = data['High'].tail(recent_period).max()
    recent_low = data['Low'].tail(recent_period).min()
    
    # Combine all levels
    all_levels = list(resistance_levels) + list(support_levels) + [recent_high, recent_low]
    
    # Group similar levels
    grouped_levels = []
    for level in all_levels:
        found_group = False
        for group in grouped_levels:
            if abs(level - group['price']) / group['price'] < tolerance:
                group['touches'] += 1
                group['price'] = (group['price'] + level) / 2  # Average
                found_group = True
                break
        
        if not found_group:
            level_type = 'Support' if level < current_price else 'Resistance'
            grouped_levels.append({
                'price': level,
                'touches': 1,
                'type': level_type,
                'strength': 1
            })
    
    # Calculate strength and filter
    for level in grouped_levels:
        level['strength'] = level['touches']
        # Boost strength if level is close to current price
        if abs(level['price'] - current_price) / current_price < 0.02:  # Within 2%
            level['strength'] += 2
    
    # Sort by strength and return strong levels
    grouped_levels.sort(key=lambda x: x['strength'], reverse=True)
    strong_levels = [l for l in grouped_levels if l['touches'] >= 1]  # Include single touches for scalping
    
    return strong_levels[:10]  # Return top 10 levels

def find_trend_lines(data, timeframe):
    """Find trend lines connecting swing points"""
    highs = data['High'].values
    lows = data['Low'].values
    
    # Adjust swing detection based on timeframe
    if "Minute" in timeframe:
        order = 3  # More sensitive for scalping
    elif "Hour" in timeframe:
        order = 4
    else:
        order = 5  # Less sensitive for daily/weekly
    
    # Find swing points
    high_peaks = argrelextrema(highs, np.greater, order=order)[0]
    low_peaks = argrelextrema(lows, np.less, order=order)[0]
    
    trend_lines = []
    
    # Support trend line (connect recent lows)
    if len(low_peaks) >= 2:
        recent_lows = low_peaks[-3:]  # Last 3 lows
        if len(recent_lows) >= 2:
            # Connect first and last low
            x1, y1 = recent_lows[0], lows[recent_lows[0]]
            x2, y2 = recent_lows[-1], lows[recent_lows[-1]]
            
            # Extend line across chart
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            
            # Calculate line points for full chart width
            start_x = 0
            end_x = len(data) - 1
            start_y = y1 + slope * (start_x - x1)
            end_y = y1 + slope * (end_x - x1)
            
            # Only add if line stays within reasonable bounds
            min_price = data['Low'].min() * 0.8
            max_price = data['High'].max() * 1.2
            
            if min_price <= start_y <= max_price and min_price <= end_y <= max_price:
                trend_lines.append({
                    'type': 'Support',
                    'start_x': start_x,
                    'end_x': end_x,
                    'start_y': start_y,
                    'end_y': end_y,
                    'dates': [data.index[start_x], data.index[end_x]]
                })
    
    # Resistance trend line (connect recent highs)
    if len(high_peaks) >= 2:
        recent_highs = high_peaks[-3:]  # Last 3 highs
        if len(recent_highs) >= 2:
            # Connect first and last high
            x1, y1 = recent_highs[0], highs[recent_highs[0]]
            x2, y2 = recent_highs[-1], highs[recent_highs[-1]]
            
            # Extend line across chart
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            
            # Calculate line points for full chart width
            start_x = 0
            end_x = len(data) - 1
            start_y = y1 + slope * (start_x - x1)
            end_y = y1 + slope * (end_x - x1)
            
            # Only add if line stays within reasonable bounds
            min_price = data['Low'].min() * 0.8
            max_price = data['High'].max() * 1.2
            
            if min_price <= start_y <= max_price and min_price <= end_y <= max_price:
                trend_lines.append({
                    'type': 'Resistance',
                    'start_x': start_x,
                    'end_x': end_x,
                    'start_y': start_y,
                    'end_y': end_y,
                    'dates': [data.index[start_x], data.index[end_x]]
                })
    
    return trend_lines

def prepare_market_data_for_ai(data, symbol, timeframe, levels, trend_lines):
    """Prepare market data summary for AI analysis"""
    current_price = data['Close'].iloc[-1]
    
    # Price action summary
    recent_high = data['High'].tail(20).max()
    recent_low = data['Low'].tail(20).min()
    price_range = recent_high - recent_low
    position_in_range = (current_price - recent_low) / price_range * 100
    
    # Volatility
    returns = data['Close'].pct_change().tail(20)
    volatility = returns.std() * 100
    
    # Volume analysis
    avg_volume = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume
    
    # Trend analysis
    ma_short = data['Close'].rolling(5).mean().iloc[-1]
    ma_long = data['Close'].rolling(20).mean().iloc[-1]
    trend_direction = "Bullish" if ma_short > ma_long else "Bearish"
    
    # Momentum
    momentum = ((current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]) * 100
    
    # Support/Resistance analysis
    support_levels = [l for l in levels if l['type'] == 'Support']
    resistance_levels = [l for l in levels if l['type'] == 'Resistance']
    
    # Find nearest levels
    nearest_support = None
    nearest_resistance = None
    
    if support_levels:
        nearest_support = min(support_levels, key=lambda x: abs(x['price'] - current_price))
    if resistance_levels:
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - current_price))
    
    market_summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": round(current_price, 2),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "position_in_range": round(position_in_range, 1),
        "volatility": round(volatility, 2),
        "volume_ratio": round(volume_ratio, 2),
        "trend_direction": trend_direction,
        "momentum": round(momentum, 2),
        "support_levels": [{"price": round(l['price'], 2), "touches": l['touches'], "distance_pct": round(abs(l['price'] - current_price) / current_price * 100, 1)} for l in support_levels[:3]],
        "resistance_levels": [{"price": round(l['price'], 2), "touches": l['touches'], "distance_pct": round(abs(l['price'] - current_price) / current_price * 100, 1)} for l in resistance_levels[:3]],
        "nearest_support": round(nearest_support['price'], 2) if nearest_support else None,
        "nearest_resistance": round(nearest_resistance['price'], 2) if nearest_resistance else None,
        "trend_lines_count": len(trend_lines)
    }
    
    return market_summary

def get_ai_market_analysis(market_data, api_key):
    """Get AI analysis of market conditions and trading opportunities"""
    if not api_key:
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        As a professional trading analyst, analyze this market data and provide actionable insights:

        MARKET DATA:
        Symbol: {market_data['symbol']}
        Timeframe: {market_data['timeframe']}
        Current Price: ${market_data['current_price']}
        Recent High: ${market_data['recent_high']}
        Recent Low: ${market_data['recent_low']}
        Position in Range: {market_data['position_in_range']}%
        Volatility: {market_data['volatility']}%
        Volume Ratio: {market_data['volume_ratio']}x
        Trend Direction: {market_data['trend_direction']}
        Momentum: {market_data['momentum']}%

        SUPPORT LEVELS: {market_data['support_levels']}
        RESISTANCE LEVELS: {market_data['resistance_levels']}
        
        Nearest Support: ${market_data['nearest_support']}
        Nearest Resistance: ${market_data['nearest_resistance']}

        Please provide:
        1. MARKET STRUCTURE ANALYSIS (2-3 sentences)
        2. KEY LEVELS ASSESSMENT (which levels are most important and why)
        3. TRADING OPPORTUNITIES (specific entry/exit points with risk management)
        4. RISK FACTORS (what to watch out for)
        5. TIMEFRAME-SPECIFIC INSIGHTS (scalping vs swing trading approach)

        Keep it concise, actionable, and focused on practical trading decisions.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional trading analyst with expertise in technical analysis, support/resistance levels, and risk management. Provide clear, actionable trading insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return None

def get_multi_timeframe_analysis(symbol, api_key):
    """Get AI analysis across multiple timeframes"""
    if not api_key:
        return None
    
    try:
        # Get data for multiple timeframes
        timeframes = {
            "5 Minutes": {"interval": "5m", "period": "5d"},
            "1 Hour": {"interval": "1h", "period": "1mo"},
            "1 Day": {"interval": "1d", "period": "1y"}
        }
        
        multi_tf_data = {}
        
        for tf_name, tf_config in timeframes.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=tf_config["period"],
                    interval=tf_config["interval"],
                    prepost=False,
                    auto_adjust=True,
                    repair=True
                )
                
                if not data.empty:
                    levels = find_support_resistance_levels(data, tf_name)
                    trend_lines = find_trend_lines(data, tf_name)
                    market_summary = prepare_market_data_for_ai(data, symbol, tf_name, levels, trend_lines)
                    multi_tf_data[tf_name] = market_summary
            except:
                continue
        
        if not multi_tf_data:
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Analyze {symbol} across multiple timeframes and provide a comprehensive trading strategy:

        MULTI-TIMEFRAME DATA:
        {json.dumps(multi_tf_data, indent=2)}

        Provide:
        1. OVERALL MARKET BIAS (bullish/bearish/neutral and why)
        2. TIMEFRAME ALIGNMENT (are the timeframes confirming each other?)
        3. BEST TRADING APPROACH (scalping, day trading, or swing trading)
        4. SPECIFIC ENTRY STRATEGY (exact levels and conditions)
        5. RISK MANAGEMENT (stop loss and take profit levels)
        6. MARKET TIMING (best times to enter/exit)

        Focus on practical, actionable advice for a trader.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert multi-timeframe trading analyst. Provide comprehensive trading strategies based on timeframe analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Multi-timeframe Analysis Error: {str(e)}")
        return None

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100.0%': low
    }
    return levels

def calculate_pivot_points(data):
    """Calculate pivot points and support/resistance levels"""
    # Use last complete day/period for calculation
    high = data['High'].iloc[-1]
    low = data['Low'].iloc[-1]
    close = data['Close'].iloc[-1]
    
    # Standard pivot point calculation
    pivot = (high + low + close) / 3
    
    # Support and resistance levels
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'PP': pivot,
        'R1': r1, 'R2': r2, 'R3': r3,
        'S1': s1, 'S2': s2, 'S3': s3
    }

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(period).mean()
    std = data['Close'].rolling(period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return sma, upper_band, lower_band

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(data):
    """Calculate Volume Weighted Average Price"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period).mean()

def find_key_levels(data):
    """Find psychological and round number levels"""
    current_price = data['Close'].iloc[-1]
    
    # Round numbers (every $5, $10, $25, $50, $100 depending on price)
    if current_price < 10:
        step = 1
    elif current_price < 50:
        step = 5
    elif current_price < 100:
        step = 10
    elif current_price < 500:
        step = 25
    else:
        step = 50
    
    # Find nearest round numbers above and below
    lower_round = int(current_price // step) * step
    upper_round = lower_round + step
    
    levels = []
    for i in range(-2, 3):  # 5 levels around current price
        level = lower_round + (i * step)
        if level > 0:
            levels.append(level)
    
    return levels

def find_price_channels(data, timeframe):
    """Find price channels and trend channels"""
    highs = data['High'].values
    lows = data['Low'].values
    
    # Adjust parameters based on timeframe
    if "Minute" in timeframe:
        lookback = min(50, len(data) // 2)
        channel_width_pct = 0.02  # 2% channel width
    elif "Hour" in timeframe:
        lookback = min(100, len(data) // 2)
        channel_width_pct = 0.03  # 3% channel width
    else:
        lookback = min(200, len(data) // 2)
        channel_width_pct = 0.05  # 5% channel width
    
    channels = []
    
    # Find recent high and low for channel
    recent_data = data.tail(lookback)
    channel_high = recent_data['High'].max()
    channel_low = recent_data['Low'].min()
    channel_mid = (channel_high + channel_low) / 2
    
    # Create price channel
    channels.append({
        'type': 'Price Channel',
        'upper': channel_high,
        'lower': channel_low,
        'middle': channel_mid,
        'strength': len(recent_data[recent_data['High'] >= channel_high * 0.99]) + 
                   len(recent_data[recent_data['Low'] <= channel_low * 1.01])
    })
    
    return channels

def calculate_market_structure(data, timeframe):
    """Calculate market structure levels (Higher Highs, Lower Lows, etc.)"""
    highs = data['High'].values
    lows = data['Low'].values
    
    # Adjust swing detection based on timeframe
    if "Minute" in timeframe:
        order = 3
    elif "Hour" in timeframe:
        order = 5
    else:
        order = 7
    
    # Find swing points
    high_peaks = argrelextrema(highs, np.greater, order=order)[0]
    low_peaks = argrelextrema(lows, np.less, order=order)[0]
    
    structure = {
        'swing_highs': [(data.index[i], highs[i]) for i in high_peaks[-5:]] if len(high_peaks) >= 5 else [],
        'swing_lows': [(data.index[i], lows[i]) for i in low_peaks[-5:]] if len(low_peaks) >= 5 else [],
        'trend': 'Unknown'
    }
    
    # Determine trend based on swing points
    if len(structure['swing_highs']) >= 2 and len(structure['swing_lows']) >= 2:
        recent_highs = [h[1] for h in structure['swing_highs'][-2:]]
        recent_lows = [l[1] for l in structure['swing_lows'][-2:]]
        
        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
            structure['trend'] = 'Uptrend'
        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
            structure['trend'] = 'Downtrend'
        else:
            structure['trend'] = 'Sideways'
    
    return structure

def calculate_volume_profile(data):
    """Calculate basic volume profile levels"""
    # Simple volume-weighted price levels
    price_volume = {}
    
    for i in range(len(data)):
        price_level = round(data['Close'].iloc[i], 2)
        volume = data['Volume'].iloc[i]
        
        if price_level in price_volume:
            price_volume[price_level] += volume
        else:
            price_volume[price_level] = volume
    
    # Get top volume levels
    sorted_levels = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'poc': sorted_levels[0][0] if sorted_levels else None,  # Point of Control
        'high_volume_levels': [level[0] for level in sorted_levels[:5]]
    }

def get_chart_config(timeframe, user_settings):
    """Get chart configuration based on user selections and timeframe"""
    
    # Base configuration optimized for timeframe
    if "Minute" in timeframe:
        base_config = {
            'annotation_size': 9,
            'line_width_multiplier': 0.8,
            'max_support_resistance': user_settings.get('max_levels', 6)
        }
    elif "Hour" in timeframe:
        base_config = {
            'annotation_size': 10,
            'line_width_multiplier': 1.0,
            'max_support_resistance': user_settings.get('max_levels', 8)
        }
    else:  # Daily, Weekly
        base_config = {
            'annotation_size': 11,
            'line_width_multiplier': 1.2,
            'max_support_resistance': user_settings.get('max_levels', 8)
        }
    
    # Add user-selected indicators
    base_config.update({
        'show_fibonacci': user_settings.get('show_fibonacci', True),
        'show_pivot_points': user_settings.get('show_pivot_points', True),
        'show_round_numbers': user_settings.get('show_round_numbers', True),
        'show_bollinger_bands': user_settings.get('show_bollinger_bands', False),
        'show_vwap': user_settings.get('show_vwap', False),
        'show_emas': user_settings.get('show_emas', [])
    })
    
    return base_config

def create_chart(data, symbol, timeframe, user_settings):
    """Create professional-looking chart with user-selected indicators"""
    fig = go.Figure()
    
    # Get chart configuration
    config = get_chart_config(timeframe, user_settings)
    
    # Enhanced candlestick chart with better colors
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color='#26C281',  # Professional green
        decreasing_line_color='#E74C3C',  # Professional red
        increasing_fillcolor='rgba(38, 194, 129, 0.8)',
        decreasing_fillcolor='rgba(231, 76, 60, 0.8)',
        line=dict(width=1)
    ))
    
    # Add volume bars as background
    colors = ['rgba(38, 194, 129, 0.3)' if close >= open else 'rgba(231, 76, 60, 0.3)' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        yaxis='y2',
        opacity=0.3,
        showlegend=False
    ))
    
    # Add Bollinger Bands (if enabled for this timeframe)
    if config['show_bollinger_bands'] and len(data) >= 20:
        bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(data)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index,
            y=bb_upper,
            mode='lines',
            name='BB Upper',
            line=dict(color='#3498DB', width=1, dash='dot'),
            opacity=0.6,
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=bb_lower,
            mode='lines',
            name='BB Lower',
            line=dict(color='#3498DB', width=1, dash='dot'),
            opacity=0.6,
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.08)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=bb_sma,
            mode='lines',
            name='BB Middle',
            line=dict(color='#3498DB', width=int(2 * config['line_width_multiplier'])),
            opacity=0.7
        ))
    
    # Add VWAP (if enabled for this timeframe)
    if config['show_vwap']:
        vwap = calculate_vwap(data)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=vwap,
            mode='lines',
            name='VWAP',
            line=dict(color='#FF6B35', width=int(3 * config['line_width_multiplier']), dash='solid'),
            opacity=0.9
        ))
    
    # Add EMAs based on timeframe configuration
    ema_colors = {9: '#F1C40F', 21: '#E67E22', 50: '#9B59B6', 200: '#8E44AD'}
    ema_widths = {9: 2, 21: 2, 50: 3, 200: 4}
    
    for ema_period in config['show_emas']:
        if len(data) >= ema_period:
            ema = calculate_ema(data, ema_period)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ema,
                mode='lines',
                name=f'EMA{ema_period}',
                line=dict(
                    color=ema_colors[ema_period], 
                    width=int(ema_widths[ema_period] * config['line_width_multiplier'])
                ),
                opacity=0.8
            ))
    
    # Add Fibonacci Retracement Levels (if enabled)
    if config['show_fibonacci']:
        fib_levels = calculate_fibonacci_levels(data)
        # Only show key Fibonacci levels to reduce clutter
        key_fib_levels = ['23.6%', '38.2%', '50.0%', '61.8%']
        
        for level_name, price in fib_levels.items():
            if level_name in key_fib_levels:
                fig.add_hline(
                    y=price,
                    line=dict(
                        color='#FFD700',
                        width=int(1 * config['line_width_multiplier']),
                        dash='dashdot'
                    ),
                    annotation_text=f"📐 {level_name}: ${price:.2f}",
                    annotation_position="left",
                    annotation=dict(
                        bgcolor='rgba(255, 215, 0, 0.7)',
                        bordercolor='#FFD700',
                        font=dict(color='black', size=config['annotation_size']),
                        borderwidth=1,
                        opacity=0.7
                    )
                )
    
    # Add Pivot Points (if enabled)
    if config['show_pivot_points']:
        pivot_points = calculate_pivot_points(data)
        pivot_colors = {
            'PP': '#FFFFFF',
            'R1': '#FF6B6B', 'R2': '#FF5252', 'R3': '#FF1744',
            'S1': '#4CAF50', 'S2': '#388E3C', 'S3': '#1B5E20'
        }
        
        # For minute timeframes, only show PP, R1, S1 to reduce clutter
        if "Minute" in timeframe:
            pivot_levels = ['PP', 'R1', 'S1']
        else:
            pivot_levels = list(pivot_points.keys())
        
        for level_name in pivot_levels:
            if level_name in pivot_points:
                price = pivot_points[level_name]
                color = pivot_colors[level_name]
                fig.add_hline(
                    y=price,
                    line=dict(
                        color=color,
                        width=int((2 if level_name == 'PP' else 1) * config['line_width_multiplier']),
                        dash='solid' if level_name == 'PP' else 'dash'
                    ),
                    annotation_text=f"⚡ {level_name}: ${price:.2f}",
                    annotation_position="right",
                    annotation=dict(
                        bgcolor=color,
                        bordercolor=color,
                        font=dict(color='white' if level_name != 'PP' else 'black', size=config['annotation_size'], family="Arial Black"),
                        borderwidth=1,
                        opacity=0.8
                    )
                )
    
    # Add Psychological/Round Number Levels (if enabled)
    if config['show_round_numbers']:
        round_levels = find_key_levels(data)
        current_price = data['Close'].iloc[-1]
        
        # Only show round numbers close to current price
        nearby_levels = [level for level in round_levels 
                        if abs(level - current_price) / current_price < 0.05]  # Within 5%
        
        for level in nearby_levels:
            fig.add_hline(
                y=level,
                line=dict(
                    color='#95A5A6',
                    width=int(1 * config['line_width_multiplier']),
                    dash='dot'
                ),
                annotation_text=f"🎯 ${level:.0f}",
                annotation_position="left",
                annotation=dict(
                    bgcolor='rgba(149, 165, 166, 0.6)',
                    bordercolor='#95A5A6',
                    font=dict(color='white', size=config['annotation_size']),
                    borderwidth=1,
                    opacity=0.6
                )
            )
    
    # CORE PROFESSIONAL LEVELS (Always shown)
    current_price = data['Close'].iloc[-1]
    
    # 1. Enhanced Support/Resistance Levels
    levels = find_support_resistance_levels(data, timeframe)
    max_levels = config['max_support_resistance']
    levels = levels[:max_levels]
    
    for i, level in enumerate(levels):
        if level['type'] == 'Resistance':
            color = '#E74C3C'
            line_style = 'dash'
            symbol = '🔴'
        else:
            color = '#26C281'
            line_style = 'dash'
            symbol = '🟢'
        
        # Professional line styling
        base_width = 2 if "Minute" in timeframe else 3
        line_width = int(min(6, base_width + level['strength']) * config['line_width_multiplier'])
        
        fig.add_hline(
            y=level['price'],
            line=dict(
                color=color,
                width=line_width,
                dash=line_style
            ),
            annotation_text=f"{symbol} {level['type']}: ${level['price']:.2f} ({level['touches']}x)",
            annotation_position="right",
            annotation=dict(
                bgcolor=color,
                bordercolor=color,
                font=dict(color='white', size=config['annotation_size'] + 1, family="Arial Black"),
                borderwidth=1,
                opacity=0.95
            )
        )
        
        # Professional zone highlighting for strong levels
        distance_pct = abs(level['price'] - current_price) / current_price * 100
        if distance_pct < 3 and level['strength'] >= 2:
            zone_height = level['price'] * 0.004
            fig.add_shape(
                type="rect",
                x0=data.index[0],
                x1=data.index[-1],
                y0=level['price'] - zone_height,
                y1=level['price'] + zone_height,
                fillcolor=color,
                opacity=0.12,
                line_width=0
            )
    
    # 2. Price Channels
    channels = find_price_channels(data, timeframe)
    for channel in channels:
        # Channel upper line
        fig.add_hline(
            y=channel['upper'],
            line=dict(color='#9C88FF', width=2, dash='longdash'),
            annotation_text=f"📊 Channel Top: ${channel['upper']:.2f}",
            annotation_position="left",
            annotation=dict(
                bgcolor='rgba(156, 136, 255, 0.8)',
                font=dict(color='white', size=config['annotation_size']),
                opacity=0.8
            )
        )
        
        # Channel lower line
        fig.add_hline(
            y=channel['lower'],
            line=dict(color='#9C88FF', width=2, dash='longdash'),
            annotation_text=f"📊 Channel Bottom: ${channel['lower']:.2f}",
            annotation_position="left",
            annotation=dict(
                bgcolor='rgba(156, 136, 255, 0.8)',
                font=dict(color='white', size=config['annotation_size']),
                opacity=0.8
            )
        )
        
        # Channel fill
        fig.add_shape(
            type="rect",
            x0=data.index[0],
            x1=data.index[-1],
            y0=channel['lower'],
            y1=channel['upper'],
            fillcolor='rgba(156, 136, 255, 0.05)',
            line_width=0
        )
    
    # 3. Market Structure Levels
    market_structure = calculate_market_structure(data, timeframe)
    
    # Mark swing highs
    for date, price in market_structure['swing_highs']:
        fig.add_annotation(
            x=date,
            y=price,
            text="🔺",
            showarrow=False,
            font=dict(size=16, color='#FF6B6B'),
            yshift=10
        )
    
    # Mark swing lows
    for date, price in market_structure['swing_lows']:
        fig.add_annotation(
            x=date,
            y=price,
            text="🔻",
            showarrow=False,
            font=dict(size=16, color='#4ECDC4'),
            yshift=-10
        )
    
    # 4. Volume Profile (Point of Control)
    volume_profile = calculate_volume_profile(data)
    if volume_profile['poc']:
        fig.add_hline(
            y=volume_profile['poc'],
            line=dict(color='#FFA726', width=3, dash='solid'),
            annotation_text=f"📈 POC (Point of Control): ${volume_profile['poc']:.2f}",
            annotation_position="right",
            annotation=dict(
                bgcolor='#FFA726',
                font=dict(color='black', size=config['annotation_size'] + 1, family="Arial Black"),
                opacity=0.9
            )
        )
    
    # Enhanced trend lines with better styling
    trend_lines = find_trend_lines(data, timeframe)
    for tline in trend_lines:
        if tline['type'] == 'Resistance':
            color = '#E74C3C'
            symbol = '📉'
        else:
            color = '#26C281'
            symbol = '📈'
        
        fig.add_trace(go.Scatter(
            x=tline['dates'],
            y=[tline['start_y'], tline['end_y']],
            mode='lines',
            line=dict(
                color=color,
                width=3,
                dash='dot'
            ),
            name=f"{symbol} {tline['type']} Trend",
            showlegend=True,
            opacity=0.8
        ))
    
    # Add current price line with enhanced styling
    fig.add_hline(
        y=current_price,
        line=dict(
            color='#FFD700',  # Gold color
            width=4,
            dash='solid'
        ),
        annotation_text=f"💰 CURRENT PRICE: ${current_price:.2f}",
        annotation_position="left",
        annotation=dict(
            bgcolor='#FFD700',
            bordercolor='#FFD700',
            font=dict(color='black', size=14, family="Arial Black"),
            borderwidth=3,
            opacity=1.0
        )
    )
    
    # Add price action zones
    recent_high = data['High'].tail(20).max()
    recent_low = data['Low'].tail(20).min()
    
    # Highlight current trading range
    fig.add_shape(
        type="rect",
        x0=data.index[-20] if len(data) >= 20 else data.index[0],
        x1=data.index[-1],
        y0=recent_low,
        y1=recent_high,
        fillcolor='rgba(255, 255, 255, 0.05)',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'),
        opacity=0.3
    )
    
    # Add session highs/lows for intraday timeframes
    if "Minute" in timeframe or "Hour" in timeframe:
        session_high = data['High'].max()
        session_low = data['Low'].min()
        
        fig.add_hline(
            y=session_high,
            line=dict(color='#FF4757', width=2, dash='longdash'),
            annotation_text=f"📈 Session High: ${session_high:.2f}",
            annotation_position="right",
            annotation=dict(
                bgcolor='#FF4757',
                font=dict(color='white', size=11, family="Arial Black"),
                opacity=0.9
            )
        )
        
        fig.add_hline(
            y=session_low,
            line=dict(color='#2ED573', width=2, dash='longdash'),
            annotation_text=f"📉 Session Low: ${session_low:.2f}",
            annotation_position="right",
            annotation=dict(
                bgcolor='#2ED573',
                font=dict(color='white', size=11, family="Arial Black"),
                opacity=0.9
            )
        )
    
    # Professional layout with enhanced styling
    fig.update_layout(
        title=dict(
            text=f"📊 {symbol} - {timeframe} - PROFESSIONAL ANALYST CHART",
            x=0.5,
            font=dict(size=22, family="Arial Black", color='#FFD700')
        ),
        xaxis_title="Time" if "Minute" in timeframe or "Hour" in timeframe else "Date",
        yaxis_title="Price ($)",
        height=900,  # Even taller chart for more levels
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        
        # Enhanced grid and styling
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showspikes=True,
            spikecolor="#FFD700",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2
        ),
        yaxis=dict(
            title="Price ($)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showspikes=True,
            spikecolor="#FFD700",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2,
            side="left"
        ),
        
        # Add secondary y-axis for volume
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, data['Volume'].max() * 4]  # Volume takes bottom 25%
        ),
        
        # Enhanced legend with better positioning
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="rgba(255,215,0,0.5)",
            borderwidth=2,
            font=dict(size=10, color='white')
        ),
        
        # Professional margins
        margin=dict(l=100, r=100, t=120, b=60),
        
        # Hover mode
        hovermode='x unified',
        
        # Enhanced background
        plot_bgcolor='rgba(10, 10, 15, 1)',
        paper_bgcolor='rgba(10, 10, 15, 1)'
    )
    
    # Add professional watermarks and info
    fig.add_annotation(
        text=f"🏆 PROFESSIONAL TRADING CHART - {timeframe} 🏆",
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        showarrow=False,
        font=dict(size=12, color="rgba(255,215,0,0.6)", family="Arial Black"),
        align="center"
    )
    
    # Create dynamic indicator summary
    core_features = ["S/R Levels", "Price Channels", "Market Structure", "Volume Profile"]
    optional_indicators = []
    
    if config['show_bollinger_bands']: optional_indicators.append("Bollinger Bands")
    if config['show_vwap']: optional_indicators.append("VWAP")
    if config['show_emas']: optional_indicators.append(f"EMA({','.join(map(str, config['show_emas']))})")
    if config['show_fibonacci']: optional_indicators.append("Fibonacci")
    if config['show_pivot_points']: optional_indicators.append("Pivot Points")
    if config['show_round_numbers']: optional_indicators.append("Psychological Levels")
    
    all_features = core_features + optional_indicators
    indicator_text = f"📊 Features: {' • '.join(all_features)}"
    
    fig.add_annotation(
        text=indicator_text,
        xref="paper", yref="paper",
        x=0.5, y=0.97,
        showarrow=False,
        font=dict(size=10, color="rgba(255,255,255,0.7)"),
        align="center"
    )
    
    # Add trend indication
    market_structure = calculate_market_structure(data, timeframe)
    trend_color = {'Uptrend': '#26C281', 'Downtrend': '#E74C3C', 'Sideways': '#F39C12', 'Unknown': '#95A5A6'}
    
    fig.add_annotation(
        text=f"📈 Market Structure: {market_structure['trend']}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=11, color=trend_color.get(market_structure['trend'], '#95A5A6'), family="Arial Black"),
        align="left"
    )
    
    return fig

# Main logic
if st.sidebar.button("Generate Chart", type="primary"):
    timeframe_config = timeframe_options[selected_timeframe]
    
    with st.spinner(f"Fetching {selected_timeframe} data for {symbol}..."):
        data = get_data(symbol, timeframe_config)
    
    if data is not None and not data.empty:
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            st.metric("Daily Change", f"${change:.2f}")
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        # Prepare user settings
        user_settings = {
            'show_fibonacci': show_fibonacci,
            'show_pivot_points': show_pivot_points,
            'show_round_numbers': show_round_numbers,
            'show_bollinger_bands': show_bollinger_bands,
            'show_vwap': show_vwap,
            'show_emas': show_emas,
            'max_levels': max_levels
        }
        
        # Create and show professional chart
        fig = create_chart(data, symbol, selected_timeframe, user_settings)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Analysis Section
        if enable_ai and openai_api_key:
            st.subheader("🤖 AI Market Analysis")
            
            with st.spinner("🧠 AI is analyzing market conditions..."):
                # Prepare data for AI
                levels = find_support_resistance_levels(data, selected_timeframe)
                trend_lines = find_trend_lines(data, selected_timeframe)
                market_data = prepare_market_data_for_ai(data, symbol, selected_timeframe, levels, trend_lines)
                
                # Get AI analysis
                ai_analysis = get_ai_market_analysis(market_data, openai_api_key)
                
                if ai_analysis:
                    st.markdown("### 🎯 AI Trading Insights")
                    st.markdown(ai_analysis)
                    
                    # Multi-timeframe analysis
                    with st.expander("📈 Multi-Timeframe Analysis", expanded=False):
                        with st.spinner("Analyzing multiple timeframes..."):
                            mtf_analysis = get_multi_timeframe_analysis(symbol, openai_api_key)
                            if mtf_analysis:
                                st.markdown(mtf_analysis)
                            else:
                                st.warning("Could not perform multi-timeframe analysis")
                else:
                    st.warning("AI analysis unavailable. Check your API key.")
        
        elif enable_ai and not openai_api_key:
            st.info("🔑 Enter your OpenAI API key in the sidebar to enable AI analysis")
        
        # Show professional technical analysis
        st.subheader("🏆 Professional Technical Analysis")
        
        # Get current price and calculate professional indicators
        current_price = data['Close'].iloc[-1]
        market_structure = calculate_market_structure(data, selected_timeframe)
        volume_profile = calculate_volume_profile(data)
        channels = find_price_channels(data, selected_timeframe)
        
        # Core Professional Analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Market Structure:**")
            trend_emoji = {"Uptrend": "📈", "Downtrend": "📉", "Sideways": "↔️", "Unknown": "❓"}
            st.write(f"• **Trend:** {trend_emoji.get(market_structure['trend'], '❓')} {market_structure['trend']}")
            st.write(f"• **Swing Highs:** {len(market_structure['swing_highs'])} identified")
            st.write(f"• **Swing Lows:** {len(market_structure['swing_lows'])} identified")
        
        with col2:
            st.write("**📈 Volume Analysis:**")
            if volume_profile['poc']:
                poc_distance = abs(volume_profile['poc'] - current_price) / current_price * 100
                st.write(f"• **POC:** ${volume_profile['poc']:.2f} ({poc_distance:.1f}% away)")
            st.write(f"• **High Volume Levels:** {len(volume_profile['high_volume_levels'])} found")
            
            # Current volume vs average
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            st.write(f"• **Volume Ratio:** {volume_ratio:.1f}x average")
        
        with col3:
            st.write("**📊 Price Channels:**")
            if channels:
                channel = channels[0]
                channel_position = (current_price - channel['lower']) / (channel['upper'] - channel['lower']) * 100
                st.write(f"• **Channel Range:** ${channel['lower']:.2f} - ${channel['upper']:.2f}")
                st.write(f"• **Position:** {channel_position:.1f}% of channel")
                st.write(f"• **Channel Strength:** {channel['strength']} touches")
        
        # Optional indicators analysis (only if selected)
        if show_fibonacci or show_pivot_points or show_round_numbers:
            st.write("**🎯 Advanced Professional Levels:**")
            
            col1, col2, col3 = st.columns(3)
            
            if show_fibonacci:
                with col1:
                    fib_levels = calculate_fibonacci_levels(data)
                    st.write("**📐 Fibonacci Levels:**")
                    key_fibs = ['23.6%', '38.2%', '50.0%', '61.8%']
                    for level_name in key_fibs:
                        if level_name in fib_levels:
                            price = fib_levels[level_name]
                            distance_pct = abs(price - current_price) / current_price * 100
                            st.write(f"• **{level_name}:** ${price:.2f} ({distance_pct:.1f}% away)")
            
            if show_pivot_points:
                with col2:
                    pivot_points = calculate_pivot_points(data)
                    st.write("**⚡ Pivot Points:**")
                    key_pivots = ['R1', 'PP', 'S1']
                    for level_name in key_pivots:
                        if level_name in pivot_points:
                            price = pivot_points[level_name]
                            distance_pct = abs(price - current_price) / current_price * 100
                            level_type = "Pivot" if level_name == "PP" else ("Resistance" if "R" in level_name else "Support")
                            st.write(f"• **{level_name}:** ${price:.2f} ({level_type})")
            
            if show_round_numbers:
                with col3:
                    round_levels = find_key_levels(data)
                    st.write("**🎯 Psychological Levels:**")
                    nearby_levels = [level for level in round_levels 
                                   if abs(level - current_price) / current_price < 0.05]
                    for level in nearby_levels[:3]:
                        distance_pct = abs(level - current_price) / current_price * 100
                        st.write(f"• **${level:.0f}** ({distance_pct:.1f}% away)")
        
        # Technical Indicators Analysis (only if selected)
        if show_emas or show_bollinger_bands or show_vwap:
            st.write("**📈 Technical Indicators Analysis:**")
            
            col1, col2, col3 = st.columns(3)
            
            if show_vwap:
                with col1:
                    vwap_current = calculate_vwap(data).iloc[-1]
                    vwap_bias = "Above VWAP ✅" if current_price > vwap_current else "Below VWAP ❌"
                    st.write(f"**VWAP Analysis:**")
                    st.write(f"• **VWAP:** ${vwap_current:.2f}")
                    st.write(f"• **Bias:** {vwap_bias}")
            
            if show_bollinger_bands:
                with col2:
                    if len(data) >= 20:
                        bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(data)
                        bb_position = "Upper" if current_price > bb_upper.iloc[-1] else ("Lower" if current_price < bb_lower.iloc[-1] else "Middle")
                        bb_squeeze = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_sma.iloc[-1] * 100
                        st.write(f"**Bollinger Bands:**")
                        st.write(f"• **Position:** {bb_position} Band")
                        st.write(f"• **Squeeze:** {bb_squeeze:.1f}%")
            
            if show_emas:
                with col3:
                    st.write(f"**EMA Analysis:**")
                    ema_trend = "Mixed"
                    if 9 in show_emas and 21 in show_emas and len(data) >= 21:
                        ema9 = calculate_ema(data, 9).iloc[-1]
                        ema21 = calculate_ema(data, 21).iloc[-1]
                        if ema9 > ema21:
                            ema_trend = "Bullish 🚀"
                        else:
                            ema_trend = "Bearish 📉"
                        st.write(f"• **EMA9:** ${ema9:.2f}")
                        st.write(f"• **EMA21:** ${ema21:.2f}")
                    st.write(f"• **Trend:** {ema_trend}")

        
        # Show detailed scalping information
        st.subheader("📊 Detailed Analysis")
        
        # Chart time range info
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📅 Chart Time Range:**")
            start_time = data.index[0].strftime('%Y-%m-%d %H:%M:%S')
            end_time = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"• **Start:** {start_time}")
            st.write(f"• **End:** {end_time}")
            st.write(f"• **Total Candles:** {len(data)}")
        
        with col2:
            st.write("**💹 Price Action:**")
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            price_range = recent_high - recent_low
            current_price = data['Close'].iloc[-1]
            position_in_range = (current_price - recent_low) / price_range * 100
            
            st.write(f"• **Recent High:** ${recent_high:.2f}")
            st.write(f"• **Recent Low:** ${recent_low:.2f}")
            st.write(f"• **Range:** ${price_range:.2f}")
            st.write(f"• **Position:** {position_in_range:.1f}%")
        
        # Support and Resistance Levels
        levels = find_support_resistance_levels(data, selected_timeframe)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🟢 Support Levels:**")
            support_levels = [l for l in levels if l['type'] == 'Support']
            if support_levels:
                for i, level in enumerate(support_levels[:5], 1):
                    distance = abs(current_price - level['price'])
                    distance_pct = (distance / current_price) * 100
                    st.write(f"{i}. **${level['price']:.2f}** ({level['touches']} touches) - {distance_pct:.1f}% away")
            else:
                st.write("No strong support levels found")
        
        with col2:
            st.write("**🔴 Resistance Levels:**")
            resistance_levels = [l for l in levels if l['type'] == 'Resistance']
            if resistance_levels:
                for i, level in enumerate(resistance_levels[:5], 1):
                    distance = abs(current_price - level['price'])
                    distance_pct = (distance / current_price) * 100
                    st.write(f"{i}. **${level['price']:.2f}** ({level['touches']} touches) - {distance_pct:.1f}% away")
            else:
                st.write("No strong resistance levels found")
        
        # Additional scalping info for intraday timeframes
        if "Minute" in selected_timeframe or "Hour" in selected_timeframe:
            st.write("**⚡ Scalping Metrics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                returns = data['Close'].pct_change().tail(20)
                volatility = returns.std() * 100
                st.write(f"• **Volatility:** {volatility:.2f}%")
                
                avg_volume = data['Volume'].tail(20).mean()
                st.write(f"• **Avg Volume:** {avg_volume:,.0f}")
            
            with col2:
                # Trend direction
                ma_short = data['Close'].rolling(5).mean().iloc[-1]
                ma_long = data['Close'].rolling(20).mean().iloc[-1]
                trend = "Bullish" if ma_short > ma_long else "Bearish"
                st.write(f"• **Short-term Trend:** {trend}")
                
                # Price momentum
                momentum = ((current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]) * 100
                st.write(f"• **10-period Momentum:** {momentum:.2f}%")
            
            with col3:
                # Trading signals
                if position_in_range > 80:
                    st.write("🔴 **Signal: Near Resistance**")
                    st.write("Consider short positions")
                elif position_in_range < 20:
                    st.write("🟢 **Signal: Near Support**")
                    st.write("Consider long positions")
                else:
                    st.write("🟡 **Signal: Mid Range**")
                    st.write("Wait for breakout")
        
        # AI-Enhanced Trading Signals
        if enable_ai and openai_api_key:
            st.subheader("🎯 AI Trading Signals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🤖 AI-Powered Insights:**")
                levels = find_support_resistance_levels(data, selected_timeframe)
                current_price = data['Close'].iloc[-1]
                
                # Find the strongest levels
                support_levels = [l for l in levels if l['type'] == 'Support']
                resistance_levels = [l for l in levels if l['type'] == 'Resistance']
                
                if support_levels:
                    strongest_support = max(support_levels, key=lambda x: x['strength'])
                    distance_to_support = abs(current_price - strongest_support['price']) / current_price * 100
                    st.write(f"💪 **Strongest Support:** ${strongest_support['price']:.2f} ({distance_to_support:.1f}% away)")
                
                if resistance_levels:
                    strongest_resistance = max(resistance_levels, key=lambda x: x['strength'])
                    distance_to_resistance = abs(current_price - strongest_resistance['price']) / current_price * 100
                    st.write(f"🛡️ **Strongest Resistance:** ${strongest_resistance['price']:.2f} ({distance_to_resistance:.1f}% away)")
            
            with col2:
                st.write("**📊 Market Structure:**")
                
                # Calculate market structure score
                recent_high = data['High'].tail(20).max()
                recent_low = data['Low'].tail(20).min()
                position = (current_price - recent_low) / (recent_high - recent_low) * 100
                
                if position > 80:
                    st.write("🔴 **Structure:** Topping zone")
                    st.write("⚠️ **Risk:** High (consider shorts)")
                elif position < 20:
                    st.write("🟢 **Structure:** Bottoming zone")
                    st.write("✅ **Opportunity:** High (consider longs)")
                else:
                    st.write("🟡 **Structure:** Consolidation")
                    st.write("⏳ **Action:** Wait for breakout")
        
    else:
        st.error(f"Could not fetch data for {symbol}")

st.markdown("---")
st.markdown("Trading Chart with Support/Resistance Levels and Trend Lines")