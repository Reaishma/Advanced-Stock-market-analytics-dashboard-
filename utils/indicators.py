import pandas as pd
import numpy as np
import streamlit as st

class TechnicalIndicators:
    """Calculates various technical indicators for stock analysis"""
    
    def __init__(self):
        pass
    
    def add_sma(self, data, periods=[20, 50]):
        """
        Add Simple Moving Average indicators
        
        Args:
            data (pd.DataFrame): Stock data
            periods (list): List of periods for SMA calculation
            
        Returns:
            pd.DataFrame: Data with SMA columns added
        """
        data_copy = data.copy()
        
        for period in periods:
            data_copy[f'SMA_{period}'] = data_copy['Close'].rolling(window=period).mean()
        
        return data_copy
    
    def add_ema(self, data, periods=[12, 26]):
        """
        Add Exponential Moving Average indicators
        
        Args:
            data (pd.DataFrame): Stock data
            periods (list): List of periods for EMA calculation
            
        Returns:
            pd.DataFrame: Data with EMA columns added
        """
        data_copy = data.copy()
        
        for period in periods:
            data_copy[f'EMA_{period}'] = data_copy['Close'].ewm(span=period).mean()
        
        return data_copy
    
    def add_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Add Bollinger Bands indicators
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): Period for moving average
            std_dev (int): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands columns added
        """
        data_copy = data.copy()
        
        # Calculate middle band (SMA)
        data_copy['BB_Middle'] = data_copy['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data_copy['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        data_copy['BB_Upper'] = data_copy['BB_Middle'] + (std * std_dev)
        data_copy['BB_Lower'] = data_copy['BB_Middle'] - (std * std_dev)
        
        # Calculate Bollinger Band Width
        data_copy['BB_Width'] = (data_copy['BB_Upper'] - data_copy['BB_Lower']) / data_copy['BB_Middle']
        
        return data_copy
    
    def add_rsi(self, data, period=14):
        """
        Add Relative Strength Index (RSI)
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): Period for RSI calculation
            
        Returns:
            pd.DataFrame: Data with RSI column added
        """
        data_copy = data.copy()
        
        # Calculate price changes
        delta = data_copy['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        return data_copy
    
    def add_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Add MACD (Moving Average Convergence Divergence) indicators
        
        Args:
            data (pd.DataFrame): Stock data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line EMA period
            
        Returns:
            pd.DataFrame: Data with MACD columns added
        """
        data_copy = data.copy()
        
        # Calculate EMAs
        ema_fast = data_copy['Close'].ewm(span=fast_period).mean()
        ema_slow = data_copy['Close'].ewm(span=slow_period).mean()
        
        # Calculate MACD line
        data_copy['MACD'] = ema_fast - ema_slow
        
        # Calculate signal line
        data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=signal_period).mean()
        
        # Calculate MACD histogram
        data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
        return data_copy
    
    def add_stochastic(self, data, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator
        
        Args:
            data (pd.DataFrame): Stock data
            k_period (int): %K period
            d_period (int): %D period
            
        Returns:
            pd.DataFrame: Data with Stochastic columns added
        """
        data_copy = data.copy()
        
        # Calculate %K
        lowest_low = data_copy['Low'].rolling(window=k_period).min()
        highest_high = data_copy['High'].rolling(window=k_period).max()
        
        data_copy['Stoch_K'] = 100 * ((data_copy['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (moving average of %K)
        data_copy['Stoch_D'] = data_copy['Stoch_K'].rolling(window=d_period).mean()
        
        return data_copy
    
    def add_atr(self, data, period=14):
        """
        Add Average True Range (ATR)
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): Period for ATR calculation
            
        Returns:
            pd.DataFrame: Data with ATR column added
        """
        data_copy = data.copy()
        
        # Calculate True Range
        high_low = data_copy['High'] - data_copy['Low']
        high_close_prev = np.abs(data_copy['High'] - data_copy['Close'].shift(1))
        low_close_prev = np.abs(data_copy['Low'] - data_copy['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR
        data_copy['ATR'] = true_range.rolling(window=period).mean()
        
        return data_copy
    
    def add_volume_indicators(self, data):
        """
        Add volume-based indicators
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with volume indicators added
        """
        data_copy = data.copy()
        
        # Volume Moving Average
        data_copy['Volume_MA'] = data_copy['Volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        data_copy['Volume_ROC'] = data_copy['Volume'].pct_change(periods=1) * 100
        
        # On Balance Volume (OBV)
        price_change = data_copy['Close'].diff()
        volume_direction = np.where(price_change > 0, data_copy['Volume'],
                                  np.where(price_change < 0, -data_copy['Volume'], 0))
        data_copy['OBV'] = volume_direction.cumsum()
        
        return data_copy
    
    def calculate_all_indicators(self, data):
        """
        Calculate all available technical indicators
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with all indicators added
        """
        data_with_indicators = data.copy()
        
        # Add all indicators
        data_with_indicators = self.add_sma(data_with_indicators, [20, 50, 200])
        data_with_indicators = self.add_ema(data_with_indicators, [12, 26])
        data_with_indicators = self.add_bollinger_bands(data_with_indicators)
        data_with_indicators = self.add_rsi(data_with_indicators)
        data_with_indicators = self.add_macd(data_with_indicators)
        data_with_indicators = self.add_stochastic(data_with_indicators)
        data_with_indicators = self.add_atr(data_with_indicators)
        data_with_indicators = self.add_volume_indicators(data_with_indicators)
        
        return data_with_indicators
    
    def get_trading_signals(self, data):
        """
        Generate basic trading signals based on technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with indicators
            
        Returns:
            dict: Trading signals and their explanations
        """
        signals = {}
        latest_data = data.iloc[-1]
        
        # RSI signals
        if 'RSI' in data.columns:
            rsi_value = latest_data['RSI']
            if rsi_value > 70:
                signals['RSI'] = {'signal': 'SELL', 'reason': f'RSI ({rsi_value:.1f}) indicates overbought conditions'}
            elif rsi_value < 30:
                signals['RSI'] = {'signal': 'BUY', 'reason': f'RSI ({rsi_value:.1f}) indicates oversold conditions'}
            else:
                signals['RSI'] = {'signal': 'HOLD', 'reason': f'RSI ({rsi_value:.1f}) is in neutral zone'}
        
        # MACD signals
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = latest_data['MACD']
            macd_signal = latest_data['MACD_Signal']
            
            if macd > macd_signal and len(data) > 1:
                prev_macd = data['MACD'].iloc[-2]
                prev_signal = data['MACD_Signal'].iloc[-2]
                if prev_macd <= prev_signal:  # Bullish crossover
                    signals['MACD'] = {'signal': 'BUY', 'reason': 'MACD bullish crossover detected'}
                else:
                    signals['MACD'] = {'signal': 'HOLD', 'reason': 'MACD above signal line'}
            elif macd < macd_signal and len(data) > 1:
                prev_macd = data['MACD'].iloc[-2]
                prev_signal = data['MACD_Signal'].iloc[-2]
                if prev_macd >= prev_signal:  # Bearish crossover
                    signals['MACD'] = {'signal': 'SELL', 'reason': 'MACD bearish crossover detected'}
                else:
                    signals['MACD'] = {'signal': 'HOLD', 'reason': 'MACD below signal line'}
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            close_price = latest_data['Close']
            bb_upper = latest_data['BB_Upper']
            bb_lower = latest_data['BB_Lower']
            
            if close_price > bb_upper:
                signals['Bollinger'] = {'signal': 'SELL', 'reason': 'Price above upper Bollinger Band'}
            elif close_price < bb_lower:
                signals['Bollinger'] = {'signal': 'BUY', 'reason': 'Price below lower Bollinger Band'}
            else:
                signals['Bollinger'] = {'signal': 'HOLD', 'reason': 'Price within Bollinger Bands'}
        
        return signals
