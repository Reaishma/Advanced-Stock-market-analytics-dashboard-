import pandas as pd
import numpy as np
import plotly.io as pio
import base64
from io import BytesIO
import streamlit as st
import zipfile
import json
from datetime import datetime

class ExportUtils:
    """Handles data and chart export functionality"""
    
    def __init__(self):
        self.export_formats = ['CSV', 'Excel', 'JSON']
        self.chart_formats = ['HTML', 'PNG', 'PDF']
    
    def export_data_to_csv(self, stock_data):
        """
        Export stock data to CSV format
        
        Args:
            stock_data (dict): Dictionary of stock data
        """
        try:
            for symbol, data in stock_data.items():
                if data is not None and not data.empty:
                    csv_data = data.to_csv()
                    
                    # Create download button
                    st.download_button(
                        label=f"📄 Download {symbol} CSV",
                        data=csv_data,
                        file_name=f"{symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"csv_{symbol}"
                    )
        
        except Exception as e:
            st.error(f"Error exporting CSV data: {str(e)}")
    
    def export_data_to_excel(self, stock_data):
        """
        Export stock data to Excel format with multiple sheets
        
        Args:
            stock_data (dict): Dictionary of stock data
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for symbol, data in stock_data.items():
                    if data is not None and not data.empty:
                        # Clean sheet name (Excel sheet names have restrictions)
                        sheet_name = symbol.replace('/', '_').replace('\\', '_')[:31]
                        data.to_excel(writer, sheet_name=sheet_name, index=True)
                
                # Add summary sheet
                summary_data = self._create_summary_data(stock_data)
                if not summary_data.empty:
                    summary_data.to_excel(writer, sheet_name='Summary', index=True)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📊 Download Excel File",
                data=excel_data,
                file_name=f"stock_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
        
        except Exception as e:
            st.error(f"Error exporting Excel data: {str(e)}")
    
    def export_data_to_json(self, stock_data):
        """
        Export stock data to JSON format
        
        Args:
            stock_data (dict): Dictionary of stock data
        """
        try:
            json_data = {}
            
            for symbol, data in stock_data.items():
                if data is not None and not data.empty:
                    json_data[symbol] = {
                        'data': data.to_dict('records'),
                        'metadata': {
                            'symbol': symbol,
                            'start_date': str(data.index[0]),
                            'end_date': str(data.index[-1]),
                            'total_records': len(data),
                            'export_timestamp': datetime.now().isoformat()
                        }
                    }
            
            json_string = json.dumps(json_data, indent=2, default=str)
            
            st.download_button(
                label="📝 Download JSON File",
                data=json_string,
                file_name=f"stock_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                key="json_download"
            )
        
        except Exception as e:
            st.error(f"Error exporting JSON data: {str(e)}")
    
    def export_charts_to_html(self, stock_data, selected_stock):
        """
        Export charts to HTML format
        
        Args:
            stock_data (dict): Dictionary of stock data
            selected_stock (str): Currently selected stock symbol
        """
        try:
            # This is a simplified version - in a full implementation,
            # you would recreate the charts and export them
            html_content = self._generate_html_report(stock_data, selected_stock)
            
            st.download_button(
                label="📈 Download HTML Report",
                data=html_content,
                file_name=f"stock_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                key="html_download"
            )
        
        except Exception as e:
            st.error(f"Error exporting HTML charts: {str(e)}")
    
    def export_charts_to_png(self, fig, filename_prefix="chart"):
        """
        Export chart to PNG format
        
        Args:
            fig (plotly.graph_objects.Figure): Plotly figure
            filename_prefix (str): Prefix for filename
        """
        try:
            # Convert plotly figure to PNG
            img_bytes = pio.to_image(fig, format="png", width=1200, height=800)
            
            st.download_button(
                label="🖼️ Download PNG Chart",
                data=img_bytes,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png",
                key=f"png_{filename_prefix}"
            )
        
        except Exception as e:
            st.error(f"Error exporting PNG chart: {str(e)}")
    
    def create_portfolio_export(self, stock_data, weights=None):
        """
        Create portfolio analysis export
        
        Args:
            stock_data (dict): Dictionary of stock data
            weights (dict): Portfolio weights for each stock
        """
        try:
            if weights is None:
                # Equal weights
                weights = {symbol: 1/len(stock_data) for symbol in stock_data.keys()}
            
            portfolio_data = self._calculate_portfolio_metrics(stock_data, weights)
            
            # Export portfolio data
            csv_data = portfolio_data.to_csv()
            
            st.download_button(
                label="📊 Download Portfolio Analysis",
                data=csv_data,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="portfolio_csv"
            )
        
        except Exception as e:
            st.error(f"Error creating portfolio export: {str(e)}")
    
    def _create_summary_data(self, stock_data):
        """
        Create summary data for all stocks
        
        Args:
            stock_data (dict): Dictionary of stock data
            
        Returns:
            pd.DataFrame: Summary data
        """
        summary_rows = []
        
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                total_return = ((current_price - start_price) / start_price) * 100
                
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                summary_rows.append({
                    'Symbol': symbol,
                    'Start Price': start_price,
                    'Current Price': current_price,
                    'Total Return (%)': total_return,
                    'Volatility (%)': volatility,
                    'Max Price': data['High'].max(),
                    'Min Price': data['Low'].min(),
                    'Avg Volume': data['Volume'].mean(),
                    'Data Points': len(data)
                })
        
        return pd.DataFrame(summary_rows)
    
    def _generate_html_report(self, stock_data, selected_stock):
        """
        Generate HTML report with basic information
        
        Args:
            stock_data (dict): Dictionary of stock data
            selected_stock (str): Currently selected stock
            
        Returns:
            str: HTML content
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Market Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                .stock-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1 class="header">Stock Market Analysis Report</h1>
            <div class="summary">
                <h2>Report Summary</h2>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Stocks Analyzed:</strong> {', '.join(stock_data.keys())}</p>
                <p><strong>Primary Focus:</strong> {selected_stock}</p>
            </div>
        """
        
        # Add stock sections
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                total_return = ((current_price - start_price) / start_price) * 100
                
                html_content += f"""
                <div class="stock-section">
                    <h3>{symbol}</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Current Price</td><td>${current_price:.2f}</td></tr>
                        <tr><td>Start Price</td><td>${start_price:.2f}</td></tr>
                        <tr><td>Total Return</td><td>{total_return:.2f}%</td></tr>
                        <tr><td>Highest Price</td><td>${data['High'].max():.2f}</td></tr>
                        <tr><td>Lowest Price</td><td>${data['Low'].min():.2f}</td></tr>
                        <tr><td>Average Volume</td><td>{data['Volume'].mean():.0f}</td></tr>
                    </table>
                </div>
                """
        
        html_content += """
            <div class="summary">
                <p><em>This report was generated by the Advanced Stock Market Dashboard</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _calculate_portfolio_metrics(self, stock_data, weights):
        """
        Calculate portfolio-level metrics
        
        Args:
            stock_data (dict): Dictionary of stock data
            weights (dict): Portfolio weights
            
        Returns:
            pd.DataFrame: Portfolio metrics
        """
        import numpy as np
        
        # Align all data to common dates
        all_data = pd.DataFrame()
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                all_data[symbol] = data['Close']
        
        # Calculate returns
        returns = all_data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        
        # Calculate cumulative portfolio value
        portfolio_value = (1 + portfolio_returns).cumprod()
        
        # Create portfolio metrics DataFrame
        portfolio_data = pd.DataFrame({
            'Portfolio_Value': portfolio_value,
            'Portfolio_Returns': portfolio_returns,
            'Cumulative_Return': portfolio_value - 1
        })
        
        return portfolio_data
