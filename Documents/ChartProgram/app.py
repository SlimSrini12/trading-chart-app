import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import argrelextrema

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

def create_chart(data, symbol, timeframe):
    """Create single chart with levels and trend lines"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add support/resistance levels (horizontal dashed lines)
    levels = find_support_resistance_levels(data, timeframe)
    for level in levels:
        color = '#ff4444' if level['type'] == 'Resistance' else '#00ff88'
        
        fig.add_hline(
            y=level['price'],
            line=dict(
                color=color,
                width=2,
                dash='dash'
            ),
            annotation_text=f"{level['type']}: ${level['price']:.2f}",
            annotation_position="right",
            annotation=dict(
                bgcolor=color,
                bordercolor=color,
                font=dict(color='white', size=10)
            )
        )
    
    # Add trend lines (diagonal dotted lines)
    trend_lines = find_trend_lines(data, timeframe)
    for tline in trend_lines:
        color = '#ff4444' if tline['type'] == 'Resistance' else '#00ff88'
        
        fig.add_trace(go.Scatter(
            x=tline['dates'],
            y=[tline['start_y'], tline['end_y']],
            mode='lines',
            line=dict(
                color=color,
                width=2,
                dash='dot'
            ),
            name=f"{tline['type']} Trend",
            showlegend=False
        ))
    
    # Update layout - SINGLE CHART ONLY
    fig.update_layout(
        title=f"{symbol} - {timeframe} - Support/Resistance Levels & Trend Lines",
        xaxis_title="Time" if "Minute" in timeframe or "Hour" in timeframe else "Date",
        yaxis_title="Price ($)",
        height=700,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,  # Remove bottom chart
        # Ensure all data is visible
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
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
        
        # Create and show SINGLE chart
        fig = create_chart(data, symbol, selected_timeframe)
        st.plotly_chart(fig, use_container_width=True)
        
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
        
    else:
        st.error(f"Could not fetch data for {symbol}")

st.markdown("---")
st.markdown("Trading Chart with Support/Resistance Levels and Trend Lines")