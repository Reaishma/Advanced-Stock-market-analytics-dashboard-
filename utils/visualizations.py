import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class ChartVisualizer:
    """Creates interactive charts and visualizations for stock data"""
    
    def __init__(self):
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff0000',
            'neutral': '#ffaa00',
            'volume': '#1f77b4',
            'sma_20': '#ff7f0e',
            'sma_50': '#2ca02c',
            'ema_12': '#d62728',
            'ema_26': '#9467bd',
            'bollinger': '#8c564b'
        }
    
    def create_price_chart(self, data, symbol, chart_type, show_sma=False, show_ema=False, 
                          show_bollinger=False, show_rsi=False, show_macd=False):
        """
        Create comprehensive price chart with technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with indicators
            symbol (str): Stock symbol
            chart_type (str): Type of chart ('Candlestick', 'Line Chart', 'OHLC', 'Volume Analysis')
            show_sma (bool): Show Simple Moving Averages
            show_ema (bool): Show Exponential Moving Averages
            show_bollinger (bool): Show Bollinger Bands
            show_rsi (bool): Show RSI
            show_macd (bool): Show MACD
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        # Determine number of subplots needed
        subplot_count = 1
        subplot_titles = [f"{symbol} - {chart_type}"]
        
        if show_rsi:
            subplot_count += 1
            subplot_titles.append("RSI")
        
        if show_macd:
            subplot_count += 1
            subplot_titles.append("MACD")
        
        if chart_type == "Volume Analysis":
            subplot_count += 1
            subplot_titles.append("Volume")
        
        # Create subplots
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=[0.6] + [0.2] * (subplot_count - 1) if subplot_count > 1 else [1.0]
        )
        
        # Main price chart
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color=self.colors['bullish'],
                    decreasing_line_color=self.colors['bearish']
                ),
                row=1, col=1
            )
        elif chart_type == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol
                ),
                row=1, col=1
            )
        elif chart_type == "Line Chart":
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f"{symbol} Close",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        elif chart_type == "Volume Analysis":
            # Add price line for volume analysis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f"{symbol} Close",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Add moving averages
        if show_sma:
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color=self.colors['sma_20'], width=1)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color=self.colors['sma_50'], width=1)
                    ),
                    row=1, col=1
                )
        
        # Add exponential moving averages
        if show_ema:
            if 'EMA_12' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['EMA_12'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color=self.colors['ema_12'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            if 'EMA_26' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['EMA_26'],
                        mode='lines',
                        name='EMA 26',
                        line=dict(color=self.colors['ema_26'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if show_bollinger and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.colors['bollinger'], width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.colors['bollinger'], width=1),
                    fill='tonexty',
                    fillcolor='rgba(140, 86, 75, 0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color=self.colors['bollinger'], width=1, dash='dot')
                ),
                row=1, col=1
            )
        
        # Add RSI subplot
        current_row = 2
        if show_rsi and 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=current_row, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)
            
            current_row += 1
        
        # Add MACD subplot
        if show_macd and all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red', width=2)
                ),
                row=current_row, col=1
            )
            
            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=current_row, col=1
            )
            
            current_row += 1
        
        # Add Volume subplot for Volume Analysis
        if chart_type == "Volume Analysis":
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=current_row, col=1
            )
            
            # Add volume moving average if available
            if 'Volume_MA' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volume_MA'],
                        mode='lines',
                        name='Volume MA',
                        line=dict(color='orange', width=2)
                    ),
                    row=current_row, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Stock Analysis",
            xaxis_rangeslider_visible=False,
            height=600 if subplot_count == 1 else 800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        
        if show_rsi:
            rsi_row = 2 if not show_macd or show_macd else 2
            fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, range=[0, 100])
        
        if show_macd:
            macd_row = 3 if show_rsi else 2
            fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
        
        if chart_type == "Volume Analysis":
            volume_row = subplot_count
            fig.update_yaxes(title_text="Volume", row=volume_row, col=1)
        
        return fig
    
    def create_prediction_chart(self, historical_data, predictions, symbol, prediction_days):
        """
        Create chart showing historical data and predictions
        
        Args:
            historical_data (pd.DataFrame): Historical stock data
            predictions (dict): Dictionary of predictions from different models
            symbol (str): Stock symbol
            prediction_days (int): Number of prediction days
            
        Returns:
            plotly.graph_objects.Figure: Prediction chart
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Create future dates
        last_date = historical_data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            start_date = last_date + pd.DateOffset(days=1)
        else:
            start_date = pd.Timestamp(last_date) + pd.DateOffset(days=1)
        
        future_dates = pd.date_range(
            start=start_date,
            periods=prediction_days,
            freq='B'  # Business days only
        )
        
        # Color palette for different models
        model_colors = {
            'linear_regression': 'red',
            'moving_average': 'green',
            'trend_analysis': 'orange',
            'random_forest': 'purple'
        }
        
        model_names = {
            'linear_regression': 'Linear Regression',
            'moving_average': 'Moving Average',
            'trend_analysis': 'Trend Analysis',
            'random_forest': 'Random Forest'
        }
        
        # Add predictions
        for model_name, pred_values in predictions.items():
            if model_name in model_colors:
                # Connect last historical point to first prediction
                connection_x = [historical_data.index[-1], future_dates[0]]
                connection_y = [historical_data['Close'].iloc[-1], pred_values[0]]
                
                fig.add_trace(
                    go.Scatter(
                        x=connection_x,
                        y=connection_y,
                        mode='lines',
                        line=dict(color=model_colors[model_name], width=2, dash='dash'),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=pred_values,
                        mode='lines+markers',
                        name=model_names.get(model_name, model_name),
                        line=dict(color=model_colors[model_name], width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )
        
        # Add vertical line to separate historical and predicted data
        fig.add_vline(
            x=historical_data.index[-1],
            line_dash="dot",
            line_color="gray",
            annotation_text="Prediction Start"
        )
        
        fig.update_layout(
            title=f"{symbol} - Price Predictions ({prediction_days} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_indicator_charts(self, data, symbol):
        """
        Create comprehensive technical indicator charts
        
        Args:
            data (pd.DataFrame): Stock data with all indicators
            symbol (str): Stock symbol
            
        Returns:
            plotly.graph_objects.Figure: Indicator charts
        """
        # Create subplots for different indicator groups
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "RSI (Relative Strength Index)",
                "Stochastic Oscillator",
                "MACD",
                "Bollinger Band Width",
                "Volume Analysis",
                "ATR (Average True Range)"
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # Stochastic
        if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_K'], name='%K', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_D'], name='%D', line=dict(color='red')),
                row=1, col=2
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=2)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
                row=2, col=1
            )
            
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', 
                      marker_color=colors, opacity=0.6),
                row=2, col=1
            )
        
        # Bollinger Band Width
        if 'BB_Width' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Width'], name='BB Width', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Volume Analysis
        if 'Volume' in data.columns:
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(data['Close'], data['Open'])]
            fig.add_trace(
                go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                      marker_color=colors, opacity=0.7),
                row=3, col=1
            )
            
            if 'Volume_MA' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['Volume_MA'], name='Volume MA', 
                              line=dict(color='orange')),
                    row=3, col=1
                )
        
        # ATR
        if 'ATR' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['ATR'], name='ATR', line=dict(color='brown')),
                row=3, col=2
            )
        
        fig.update_layout(
            title=f"{symbol} - Technical Indicators Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def prepare_comparison_data(self, stock_data):
        """
        Prepare data for multi-stock comparison (normalize prices)
        
        Args:
            stock_data (dict): Dictionary of stock data
            
        Returns:
            pd.DataFrame: Normalized comparison data
        """
        comparison_df = pd.DataFrame()
        
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                # Normalize to percentage change from first day
                normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                comparison_df[symbol] = normalized
        
        return comparison_df
    
    def create_comparison_chart(self, comparison_data):
        """
        Create multi-stock comparison chart
        
        Args:
            comparison_data (pd.DataFrame): Normalized comparison data
            
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        fig = go.Figure()
        
        for column in comparison_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_data.index,
                    y=comparison_data[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title="Stock Performance Comparison (Normalized)",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        return fig
