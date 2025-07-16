import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PredictionModels:
    """Implements various prediction models for stock price forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def prepare_features(self, data, lookback_days=30):
        """
        Prepare features for machine learning models
        
        Args:
            data (pd.DataFrame): Stock data
            lookback_days (int): Number of days to look back for features
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        # Create feature matrix
        features = []
        targets = []
        
        # Technical indicators as features
        data_copy = data.copy()
        
        # Price-based features
        data_copy['Returns'] = data_copy['Close'].pct_change()
        data_copy['Returns_MA5'] = data_copy['Returns'].rolling(5).mean()
        data_copy['Returns_MA10'] = data_copy['Returns'].rolling(10).mean()
        data_copy['Price_MA5'] = data_copy['Close'].rolling(5).mean()
        data_copy['Price_MA10'] = data_copy['Close'].rolling(10).mean()
        data_copy['Price_MA20'] = data_copy['Close'].rolling(20).mean()
        
        # Volume features
        data_copy['Volume_MA5'] = data_copy['Volume'].rolling(5).mean()
        data_copy['Volume_Ratio'] = data_copy['Volume'] / data_copy['Volume_MA5']
        
        # Volatility features
        data_copy['Volatility_5'] = data_copy['Returns'].rolling(5).std()
        data_copy['Volatility_10'] = data_copy['Returns'].rolling(10).std()
        
        # High-Low spread
        data_copy['HL_Spread'] = (data_copy['High'] - data_copy['Low']) / data_copy['Close']
        
        # Price position within the day's range
        data_copy['Price_Position'] = (data_copy['Close'] - data_copy['Low']) / (data_copy['High'] - data_copy['Low'])
        
        # Drop NaN values
        data_copy = data_copy.dropna()
        
        if len(data_copy) < lookback_days + 1:
            return None, None
        
        feature_columns = [
            'Returns', 'Returns_MA5', 'Returns_MA10',
            'Price_MA5', 'Price_MA10', 'Price_MA20',
            'Volume_Ratio', 'Volatility_5', 'Volatility_10',
            'HL_Spread', 'Price_Position'
        ]
        
        # Create sequences for time series prediction
        for i in range(lookback_days, len(data_copy)):
            # Features: use past lookback_days of data
            feature_window = []
            for j in range(i - lookback_days, i):
                feature_values = []
                for col in feature_columns:
                    if col in data_copy.columns:
                        feature_values.append(data_copy[col].iloc[j])
                feature_window.extend(feature_values)
            
            features.append(feature_window)
            
            # Target: next day's closing price
            targets.append(data_copy['Close'].iloc[i])
        
        return np.array(features), np.array(targets)
    
    def linear_regression_prediction(self, data, prediction_days=7):
        """
        Simple linear regression prediction based on time trend
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        # Use simple time-based linear regression
        prices = data['Close'].values
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future prices
        future_X = np.arange(len(prices), len(prices) + prediction_days).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return predictions
    
    def moving_average_prediction(self, data, prediction_days=7, window=20):
        """
        Moving average based prediction
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Number of days to predict
            window (int): Moving average window
            
        Returns:
            np.array: Predicted prices
        """
        # Calculate moving average
        ma = data['Close'].rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        
        # Simple prediction: assume price will revert to moving average
        current_price = data['Close'].iloc[-1]
        reversion_factor = 0.1  # How much to revert each day
        
        predictions = []
        price = current_price
        
        for _ in range(prediction_days):
            # Move towards moving average with some trend
            price = price + (last_ma - price) * reversion_factor
            predictions.append(price)
        
        return np.array(predictions)
    
    def trend_analysis_prediction(self, data, prediction_days=7):
        """
        Trend-based prediction using recent price momentum
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        # Calculate recent trend
        short_term_window = min(5, len(data) // 4)
        medium_term_window = min(20, len(data) // 2)
        
        if len(data) < medium_term_window:
            # Not enough data, use simple momentum
            recent_returns = data['Close'].pct_change().iloc[-3:].mean()
        else:
            # Calculate weighted trend
            short_returns = data['Close'].iloc[-short_term_window:].pct_change().mean()
            medium_returns = data['Close'].iloc[-medium_term_window:].pct_change().mean()
            
            # Weight recent returns more heavily
            recent_returns = 0.7 * short_returns + 0.3 * medium_returns
        
        # Apply trend with diminishing effect
        current_price = data['Close'].iloc[-1]
        predictions = []
        
        for i in range(prediction_days):
            # Diminish trend effect over time
            trend_factor = recent_returns * (0.8 ** i)
            current_price = current_price * (1 + trend_factor)
            predictions.append(current_price)
        
        return np.array(predictions)
    
    def random_forest_prediction(self, data, prediction_days=7):
        """
        Random Forest based prediction using technical features
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        try:
            X, y = self.prepare_features(data, lookback_days=10)
            
            if X is None or len(X) < 20:
                # Fallback to trend analysis
                return self.trend_analysis_prediction(data, prediction_days)
            
            # Split data for training
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # For prediction, we need to simulate future features
            # This is simplified - in practice, you'd need more sophisticated feature engineering
            last_features = X_test_scaled[-1] if len(X_test_scaled) > 0 else X_train_scaled[-1]
            
            predictions = []
            current_features = last_features.copy()
            
            for _ in range(prediction_days):
                pred_price = rf_model.predict(current_features.reshape(1, -1))[0]
                predictions.append(pred_price)
                
                # Update features (simplified approach)
                # In practice, you'd recalculate technical indicators
                current_features = np.roll(current_features, -1)
                current_features[-1] = pred_price
            
            return np.array(predictions)
            
        except Exception as e:
            # Fallback to simpler method
            return self.trend_analysis_prediction(data, prediction_days)
    
    def generate_predictions(self, data, prediction_days=7):
        """
        Generate predictions using multiple models
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Number of days to predict
            
        Returns:
            dict: Dictionary of predictions from different models
        """
        predictions = {}
        
        # Ensure data has proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        
        # Fallback value for any failed predictions
        fallback_value = data['Close'].iloc[-1]
        
        try:
            # Linear regression prediction
            predictions['linear_regression'] = self.linear_regression_prediction(data, prediction_days)
        except Exception as e:
            print(f"Linear regression failed: {e}")
            predictions['linear_regression'] = np.full(prediction_days, fallback_value)
        
        try:
            # Moving average prediction
            predictions['moving_average'] = self.moving_average_prediction(data, prediction_days)
        except Exception as e:
            print(f"Moving average failed: {e}")
            predictions['moving_average'] = np.full(prediction_days, fallback_value)
        
        try:
            # Trend analysis prediction
            predictions['trend_analysis'] = self.trend_analysis_prediction(data, prediction_days)
        except Exception as e:
            print(f"Trend analysis failed: {e}")
            predictions['trend_analysis'] = np.full(prediction_days, fallback_value)
        
        try:
            # Random forest prediction
            predictions['random_forest'] = self.random_forest_prediction(data, prediction_days)
        except Exception as e:
            print(f"Random forest failed: {e}")
            predictions['random_forest'] = np.full(prediction_days, fallback_value)
        
        return predictions
    
    def calculate_prediction_confidence(self, data, model_type='trend_analysis'):
        """
        Calculate confidence score for predictions based on historical accuracy
        
        Args:
            data (pd.DataFrame): Stock data
            model_type (str): Type of model to evaluate
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            if len(data) < 30:
                return 0.5  # Low confidence with limited data
            
            # Backtest the model on recent data
            test_days = min(10, len(data) // 4)
            train_data = data.iloc[:-test_days]
            test_data = data.iloc[-test_days:]
            
            # Generate predictions
            if model_type == 'linear_regression':
                predictions = self.linear_regression_prediction(train_data, test_days)
            elif model_type == 'moving_average':
                predictions = self.moving_average_prediction(train_data, test_days)
            elif model_type == 'trend_analysis':
                predictions = self.trend_analysis_prediction(train_data, test_days)
            else:
                return 0.5
            
            # Calculate accuracy
            actual_prices = test_data['Close'].values
            mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
            
            # Convert MAPE to confidence (inverse relationship)
            confidence = max(0, 1 - (mape / 50))  # 50% MAPE = 0 confidence
            
            return min(1, confidence)
            
        except Exception as e:
            return 0.5  # Default confidence
