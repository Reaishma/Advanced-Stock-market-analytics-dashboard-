import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    """Handles stock data fetching and processing"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def fetch_stock_data(_self, symbol, start_date, end_date):
        """
        Fetch stock data for a given symbol and date range
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Create yfinance ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing required columns for {symbol}")
                return None
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def get_stock_info(_self, symbol):
        """
        Get basic stock information
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
    
    def calculate_performance_metrics(self, stock_data):
        """
        Calculate performance metrics for multiple stocks
        
        Args:
            stock_data (dict): Dictionary of stock data
            
        Returns:
            pd.DataFrame: Performance metrics
        """
        metrics = {}
        
        for symbol, data in stock_data.items():
            if data is None or data.empty:
                continue
                
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Performance metrics
            current_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[0]
            total_return = ((current_price - start_price) / start_price) * 100
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            max_drawdown = self._calculate_max_drawdown(data['Close'])
            
            # Recent performance
            if len(data) >= 21:  # 1 month
                month_return = ((current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21]) * 100
            else:
                month_return = total_return
            
            if len(data) >= 5:  # 1 week
                week_return = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100
            else:
                week_return = total_return
            
            metrics[symbol] = {
                'Current Price': f"${current_price:.2f}",
                'Total Return (%)': f"{total_return:.2f}%",
                '1 Month Return (%)': f"{month_return:.2f}%",
                '1 Week Return (%)': f"{week_return:.2f}%",
                'Volatility (%)': f"{volatility:.2f}%",
                'Max Drawdown (%)': f"{max_drawdown:.2f}%",
                'Volume (Avg)': f"{data['Volume'].mean():.0f}"
            }
        
        return pd.DataFrame(metrics).T
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def get_market_summary(self):
        """
        Get market summary data for major indices
        
        Returns:
            dict: Market summary data
        """
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'VIX': '^VIX'
        }
        
        summary = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2d')
                
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100
                    
                    summary[name] = {
                        'price': current,
                        'change': change,
                        'change_pct': change_pct
                    }
            except Exception:
                continue
        
        return summary
