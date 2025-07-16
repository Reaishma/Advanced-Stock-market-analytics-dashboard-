import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_fetcher import DataFetcher
from utils.indicators import TechnicalIndicators
from utils.prediction_models import PredictionModels
from utils.visualizations import ChartVisualizer
from utils.export_utils import ExportUtils

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize utility classes
@st.cache_resource
def initialize_utilities():
    return {
        'data_fetcher': DataFetcher(),
        'indicators': TechnicalIndicators(),
        'predictor': PredictionModels(),
        'visualizer': ChartVisualizer(),
        'exporter': ExportUtils()
    }

utils = initialize_utilities()

def main():
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .block-container {
        padding-top: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>📈 Advanced Stock Market Analytics Dashboard</h1><p>Professional-grade financial analysis with ML predictions</p></div>', unsafe_allow_html=True)
    
    # Market overview at the top
    st.subheader("📊 Market Overview")
    with st.spinner("Loading market data..."):
        market_summary = utils['data_fetcher'].get_market_summary()
    
    if market_summary:
        cols = st.columns(len(market_summary))
        for i, (index, data) in enumerate(market_summary.items()):
            with cols[i]:
                change_color = "green" if data['change'] >= 0 else "red"
                st.metric(
                    label=index,
                    value=f"{data['price']:.2f}",
                    delta=f"{data['change']:.2f} ({data['change_pct']:.2f}%)"
                )
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("Configure your analysis parameters below:")
        
        # Stock selection
        st.markdown("#### 📊 Stock Selection")
        
        # Popular stocks suggestion
        popular_stocks = {
            "Tech Giants": "AAPL,GOOGL,MSFT,AMZN",
            "Electric Vehicle": "TSLA,NIO,RIVN,LCID",
            "Financial": "JPM,BAC,WFC,GS",
            "Healthcare": "JNJ,PFE,UNH,MRNA",
            "Energy": "XOM,CVX,COP,EOG"
        }
        
        preset_selection = st.selectbox(
            "Quick Select Portfolio",
            ["Custom"] + list(popular_stocks.keys()),
            help="Choose a predefined portfolio or select Custom to enter your own"
        )
        
        if preset_selection != "Custom":
            stock_symbols = st.text_input(
                "Stock symbols",
                value=popular_stocks[preset_selection],
                help="Modify the selection or use as-is"
            )
        else:
            stock_symbols = st.text_input(
                "Enter stock symbols (comma-separated)",
                value="AAPL,GOOGL,MSFT,TSLA",
                help="Enter stock symbols separated by commas (e.g., AAPL,GOOGL,MSFT)"
            )
        
        symbols_list = [symbol.strip().upper() for symbol in stock_symbols.split(',') if symbol.strip()]
        
        # Time range selection
        st.markdown("#### 📅 Time Range")
        time_periods = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825
        }
        
        selected_period = st.selectbox("Select time period", list(time_periods.keys()), index=3)
        days = time_periods[selected_period]
        
        # Custom date range
        use_custom_range = st.checkbox("Use custom date range")
        if use_custom_range:
            start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End date", datetime.now())
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        
        # Chart type selection
        st.markdown("#### 📈 Visualization Options")
        chart_type = st.selectbox(
            "Chart type",
            ["Candlestick", "Line Chart", "OHLC", "Volume Analysis"],
            help="Choose the visualization style for price data"
        )
        
        # Technical indicators
        st.markdown("#### 📊 Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            show_sma = st.checkbox("SMA (20, 50)", value=True, help="Simple Moving Average")
            show_bollinger = st.checkbox("Bollinger Bands", help="Price volatility bands")
            show_macd = st.checkbox("MACD", help="Moving Average Convergence Divergence")
        with col2:
            show_ema = st.checkbox("EMA (12, 26)", help="Exponential Moving Average")
            show_rsi = st.checkbox("RSI", help="Relative Strength Index")
        
        # Prediction options - Disabled due to timestamp compatibility issues
        st.markdown("#### 🤖 AI Prediction Models")
        st.info("ML predictions temporarily disabled for system stability")
        enable_predictions = False
        prediction_days = 7
        
        # Export options
        st.markdown("#### 💾 Export Options")
        export_format = st.selectbox("Data Format", ["CSV", "Excel", "JSON"], help="Choose export format for your data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export Data", help="Download stock data in selected format"):
                st.session_state.export_data = True
                st.session_state.export_format = export_format
        with col2:
            if st.button("📊 Export Charts", help="Download charts as HTML report"):
                st.session_state.export_charts = True
    
    # Main dashboard area
    if not symbols_list:
        st.warning("Please enter at least one stock symbol.")
        return
    
    # Fetch data for all symbols
    with st.spinner("Fetching stock data..."):
        stock_data = {}
        failed_symbols = []
        
        for symbol in symbols_list:
            try:
                data = utils['data_fetcher'].fetch_stock_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    stock_data[symbol] = data
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                failed_symbols.append(symbol)
                st.error(f"Failed to fetch data for {symbol}: {str(e)}")
        
        if failed_symbols:
            st.warning(f"Failed to fetch data for: {', '.join(failed_symbols)}")
        
        if not stock_data:
            st.error("No valid stock data found. Please check your symbols and try again.")
            return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Analysis", 
        "📊 Technical Indicators", 
        "📋 Portfolio Comparison",
        "📝 Trading Signals"
    ])
    
    with tab1:
        st.header("Price Analysis")
        
        # Individual stock analysis
        selected_stock = st.selectbox("Select stock for detailed analysis", list(stock_data.keys()))
        
        if selected_stock in stock_data:
            data = stock_data[selected_stock]
            
            # Add technical indicators to data
            if show_sma:
                data = utils['indicators'].add_sma(data, [20, 50])
            if show_ema:
                data = utils['indicators'].add_ema(data, [12, 26])
            if show_bollinger:
                data = utils['indicators'].add_bollinger_bands(data)
            if show_rsi:
                data = utils['indicators'].add_rsi(data)
            if show_macd:
                data = utils['indicators'].add_macd(data)
            
            # Create visualization
            fig = utils['visualizer'].create_price_chart(
                data, selected_stock, chart_type,
                show_sma, show_ema, show_bollinger, show_rsi, show_macd
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
            with col2:
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            with col3:
                st.metric("52W High", f"${data['High'].rolling(252).max().iloc[-1]:.2f}")
            with col4:
                st.metric("52W Low", f"${data['Low'].rolling(252).min().iloc[-1]:.2f}")
            with col5:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility (1Y)", f"{volatility:.2f}%")
    
    with tab2:
        st.header("Technical Indicators")
        
        if selected_stock in stock_data:
            data = stock_data[selected_stock]
            
            # Calculate all technical indicators
            data_with_indicators = utils['indicators'].calculate_all_indicators(data)
            
            # Create indicator charts
            fig_indicators = utils['visualizer'].create_indicator_charts(data_with_indicators, selected_stock)
            st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Display indicator values table
            st.subheader("Latest Indicator Values")
            latest_indicators = {}
            
            if 'RSI' in data_with_indicators.columns:
                latest_indicators['RSI'] = f"{data_with_indicators['RSI'].iloc[-1]:.2f}"
            if 'MACD' in data_with_indicators.columns:
                latest_indicators['MACD'] = f"{data_with_indicators['MACD'].iloc[-1]:.4f}"
            if 'MACD_Signal' in data_with_indicators.columns:
                latest_indicators['MACD Signal'] = f"{data_with_indicators['MACD_Signal'].iloc[-1]:.4f}"
            if 'BB_Upper' in data_with_indicators.columns:
                latest_indicators['Bollinger Upper'] = f"${data_with_indicators['BB_Upper'].iloc[-1]:.2f}"
                latest_indicators['Bollinger Lower'] = f"${data_with_indicators['BB_Lower'].iloc[-1]:.2f}"
            
            if latest_indicators:
                indicator_df = pd.DataFrame(list(latest_indicators.items()), columns=['Indicator', 'Value'])
                st.dataframe(indicator_df, use_container_width=True)
    
    with tab3:
        st.header("Stock Comparison")
        
        if len(stock_data) > 1:
            # Normalize prices for comparison
            comparison_data = utils['visualizer'].prepare_comparison_data(stock_data)
            
            # Create comparison chart
            fig_comparison = utils['visualizer'].create_comparison_chart(comparison_data)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Performance metrics table
            st.subheader("Performance Comparison")
            performance_metrics = utils['data_fetcher'].calculate_performance_metrics(stock_data)
            st.dataframe(performance_metrics, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            correlation_data = pd.DataFrame({symbol: data['Close'] for symbol, data in stock_data.items()})
            correlation_matrix = correlation_data.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Stock Price Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Add multiple stocks to view comparison analysis.")
    
    with tab4:
        st.header("Trading Signals & Analysis")
        
        if stock_data:
            selected_signal_stock = st.selectbox("Select stock for signal analysis", list(stock_data.keys()), key="signal_stock")
            
            if selected_signal_stock in stock_data:
                data_with_indicators = utils['indicators'].calculate_all_indicators(stock_data[selected_signal_stock])
                signals = utils['indicators'].get_trading_signals(data_with_indicators)
                
                if signals:
                    st.subheader("Current Trading Signals")
                    
                    # Create signal cards
                    signal_cols = st.columns(len(signals))
                    for i, (indicator, signal_data) in enumerate(signals.items()):
                        with signal_cols[i]:
                            signal_color = {
                                'BUY': 'green',
                                'SELL': 'red',
                                'HOLD': 'orange'
                            }.get(signal_data['signal'], 'gray')
                            
                            st.markdown(f"""
                            <div style="
                                background: {'#d4edda' if signal_data['signal'] == 'BUY' else '#f8d7da' if signal_data['signal'] == 'SELL' else '#fff3cd'};
                                padding: 1rem;
                                border-radius: 8px;
                                border-left: 4px solid {signal_color};
                                margin: 0.5rem 0;
                            ">
                                <h4 style="margin: 0; color: {signal_color};">{indicator}</h4>
                                <p style="margin: 0.5rem 0; font-weight: bold; color: {signal_color};">{signal_data['signal']}</p>
                                <p style="margin: 0; font-size: 0.9rem;">{signal_data['reason']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Risk metrics
                st.subheader("Risk Analysis")
                col1, col2, col3 = st.columns(3)
                
                returns = data_with_indicators['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                max_drawdown = ((data_with_indicators['Close'] / data_with_indicators['Close'].cummax()) - 1).min() * 100
                
                with col1:
                    st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                # Technical analysis summary
                st.subheader("Technical Analysis Summary")
                
                if 'RSI' in data_with_indicators.columns:
                    current_rsi = data_with_indicators['RSI'].iloc[-1]
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    
                    st.write(f"**RSI Analysis:** Current RSI is {current_rsi:.1f} - {rsi_status}")
                
                if all(col in data_with_indicators.columns for col in ['SMA_20', 'SMA_50']):
                    sma_20 = data_with_indicators['SMA_20'].iloc[-1]
                    sma_50 = data_with_indicators['SMA_50'].iloc[-1]
                    current_price = data_with_indicators['Close'].iloc[-1]
                    
                    trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Mixed"
                    st.write(f"**Trend Analysis:** {trend} (Price: ${current_price:.2f}, SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f})")
        else:
            st.info("Load stock data to view trading signals.")
    
    # Handle export requests
    if hasattr(st.session_state, 'export_charts') and st.session_state.export_charts:
        try:
            utils['exporter'].export_charts_to_html(stock_data, selected_stock if 'selected_stock' in locals() else list(stock_data.keys())[0])
            st.success("Charts exported successfully!")
        except Exception as e:
            st.error(f"Error exporting charts: {str(e)}")
        st.session_state.export_charts = False
    
    if hasattr(st.session_state, 'export_data') and st.session_state.export_data:
        try:
            export_format = getattr(st.session_state, 'export_format', 'CSV')
            if export_format == "CSV":
                utils['exporter'].export_data_to_csv(stock_data)
            elif export_format == "Excel":
                utils['exporter'].export_data_to_excel(stock_data)
            elif export_format == "JSON":
                utils['exporter'].export_data_to_json(stock_data)
            st.success("Data exported successfully!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
        st.session_state.export_data = False

if __name__ == "__main__":
    main()
