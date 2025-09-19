"""
Time Series Analysis Module for Healthcare DSS

This module implements time series analysis for healthcare data to analyze
temporal patterns, disease progression, treatment monitoring, and seasonal trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Time Series Analysis for Healthcare Data
    
    Analyzes temporal patterns in healthcare data including disease progression,
    treatment monitoring, seasonal trends, and predictive forecasting.
    """
    
    def __init__(self, data_manager):
        """
        Initialize Time Series Analyzer
        
        Args:
            data_manager: DataManager instance with loaded datasets
        """
        self.data_manager = data_manager
        self.time_series_data = {}
        self.analysis_results = {}
        
    def prepare_time_series_data(self, dataset_name: str, time_column: str = None, 
                                value_column: str = None) -> pd.DataFrame:
        """
        Prepare time series data for analysis
        
        Args:
            dataset_name: Name of the dataset to analyze
            time_column: Column containing time information
            value_column: Column containing values to analyze
            
        Returns:
            DataFrame with time series data
        """
        if dataset_name not in self.data_manager.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.data_manager.datasets[dataset_name].copy()
        
        # For healthcare expenditure data, create time series
        if dataset_name == 'healthcare_expenditure':
            # Get year columns
            year_cols = [col for col in df.columns if '20' in col and '[' in col]
            
            # Create time series for each country
            time_series_list = []
            for _, row in df.iterrows():
                country = row['Country Name']
                for year_col in year_cols:
                    year = year_col.split('[')[0].strip()
                    value = row[year_col]
                    if pd.notna(value) and value != '..':
                        time_series_list.append({
                            'country': country,
                            'year': int(year),
                            'expenditure': float(value)
                        })
            
            ts_data = pd.DataFrame(time_series_list)
            ts_data = ts_data.sort_values(['country', 'year'])
            
        # For other datasets, create synthetic time series
        elif dataset_name == 'diabetes':
            # Create synthetic time series based on target progression
            ts_data = df.copy()
            ts_data['time_point'] = range(len(ts_data))
            ts_data['progression'] = ts_data['target']
            ts_data = ts_data[['time_point', 'progression']].copy()
            
        elif dataset_name == 'breast_cancer':
            # Create synthetic time series based on tumor characteristics
            ts_data = df.copy()
            ts_data['time_point'] = range(len(ts_data))
            ts_data['tumor_severity'] = ts_data['mean radius'] * ts_data['mean texture']
            ts_data = ts_data[['time_point', 'tumor_severity']].copy()
            
        else:
            raise ValueError(f"Time series analysis not implemented for dataset: {dataset_name}")
        
        self.time_series_data[dataset_name] = ts_data
        logger.info(f"Prepared time series data for {dataset_name}: {len(ts_data)} data points")
        
        return ts_data
    
    def analyze_temporal_patterns(self, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the data
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary with temporal pattern analysis results
        """
        if dataset_name not in self.time_series_data:
            self.prepare_time_series_data(dataset_name)
        
        ts_data = self.time_series_data[dataset_name]
        
        logger.info(f"Analyzing temporal patterns for {dataset_name}")
        
        results = {
            'dataset': dataset_name,
            'total_data_points': len(ts_data),
            'time_range': {},
            'trend_analysis': {},
            'seasonality_analysis': {},
            'volatility_analysis': {}
        }
        
        if dataset_name == 'healthcare_expenditure':
            # Analyze expenditure trends by country
            countries = ts_data['country'].unique()
            results['countries_analyzed'] = len(countries)
            
            # Calculate overall trends
            overall_trend = ts_data.groupby('year')['expenditure'].agg(['mean', 'std', 'min', 'max'])
            results['time_range'] = {
                'start_year': ts_data['year'].min(),
                'end_year': ts_data['year'].max(),
                'years_covered': ts_data['year'].nunique()
            }
            
            # Trend analysis
            years = overall_trend.index.values.reshape(-1, 1)
            mean_expenditure = overall_trend['mean'].values
            
            # Linear trend
            trend_model = LinearRegression()
            trend_model.fit(years, mean_expenditure)
            trend_slope = trend_model.coef_[0]
            trend_r2 = trend_model.score(years, mean_expenditure)
            
            results['trend_analysis'] = {
                'slope': trend_slope,
                'r2_score': trend_r2,
                'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'strength': 'strong' if abs(trend_r2) > 0.7 else 'moderate' if abs(trend_r2) > 0.3 else 'weak'
            }
            
            # Volatility analysis
            expenditure_std = overall_trend['std'].mean()
            expenditure_cv = expenditure_std / overall_trend['mean'].mean()
            
            results['volatility_analysis'] = {
                'average_std': expenditure_std,
                'coefficient_of_variation': expenditure_cv,
                'volatility_level': 'high' if expenditure_cv > 0.3 else 'moderate' if expenditure_cv > 0.1 else 'low'
            }
            
        else:
            # Analyze other datasets
            time_col = 'time_point'
            value_col = ts_data.columns[1]  # Second column is the value
            
            results['time_range'] = {
                'start_point': ts_data[time_col].min(),
                'end_point': ts_data[time_col].max(),
                'total_points': len(ts_data)
            }
            
            # Trend analysis
            time_points = ts_data[time_col].values.reshape(-1, 1)
            values = ts_data[value_col].values
            
            trend_model = LinearRegression()
            trend_model.fit(time_points, values)
            trend_slope = trend_model.coef_[0]
            trend_r2 = trend_model.score(time_points, values)
            
            results['trend_analysis'] = {
                'slope': trend_slope,
                'r2_score': trend_r2,
                'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'strength': 'strong' if abs(trend_r2) > 0.7 else 'moderate' if abs(trend_r2) > 0.3 else 'weak'
            }
            
            # Volatility analysis
            value_std = ts_data[value_col].std()
            value_cv = value_std / ts_data[value_col].mean()
            
            results['volatility_analysis'] = {
                'standard_deviation': value_std,
                'coefficient_of_variation': value_cv,
                'volatility_level': 'high' if value_cv > 0.3 else 'moderate' if value_cv > 0.1 else 'low'
            }
        
        self.analysis_results[dataset_name] = results
        return results
    
    def detect_anomalies(self, dataset_name: str, method: str = 'statistical') -> Dict[str, Any]:
        """
        Detect anomalies in time series data
        
        Args:
            dataset_name: Name of the dataset to analyze
            method: Method for anomaly detection ('statistical', 'zscore', 'iqr')
            
        Returns:
            Dictionary with anomaly detection results
        """
        if dataset_name not in self.time_series_data:
            self.prepare_time_series_data(dataset_name)
        
        ts_data = self.time_series_data[dataset_name]
        
        logger.info(f"Detecting anomalies in {dataset_name} using {method} method")
        
        anomalies = []
        
        if dataset_name == 'healthcare_expenditure':
            # Detect anomalies in expenditure data
            for country in ts_data['country'].unique():
                country_data = ts_data[ts_data['country'] == country].copy()
                
                if method == 'statistical':
                    # Z-score method
                    z_scores = np.abs(stats.zscore(country_data['expenditure']))
                    anomaly_mask = z_scores > 2
                elif method == 'iqr':
                    # IQR method
                    Q1 = country_data['expenditure'].quantile(0.25)
                    Q3 = country_data['expenditure'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    anomaly_mask = (country_data['expenditure'] < lower_bound) | (country_data['expenditure'] > upper_bound)
                else:
                    continue
                
                country_anomalies = country_data[anomaly_mask]
                for _, anomaly in country_anomalies.iterrows():
                    anomalies.append({
                        'country': country,
                        'year': anomaly['year'],
                        'expenditure': anomaly['expenditure'],
                        'method': method
                    })
        
        else:
            # Detect anomalies in other datasets
            value_col = ts_data.columns[1]
            
            if method == 'statistical':
                z_scores = np.abs(stats.zscore(ts_data[value_col]))
                anomaly_mask = z_scores > 2
            elif method == 'iqr':
                Q1 = ts_data[value_col].quantile(0.25)
                Q3 = ts_data[value_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_mask = (ts_data[value_col] < lower_bound) | (ts_data[value_col] > upper_bound)
            
            anomaly_data = ts_data[anomaly_mask]
            for _, anomaly in anomaly_data.iterrows():
                anomalies.append({
                    'time_point': anomaly.iloc[0],
                    'value': anomaly.iloc[1],
                    'method': method
                })
        
        results = {
            'dataset': dataset_name,
            'method': method,
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(ts_data) * 100,
            'anomalies': anomalies
        }
        
        logger.info(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(ts_data)*100:.1f}%)")
        return results
    
    def forecast_values(self, dataset_name: str, periods: int = 5) -> Dict[str, Any]:
        """
        Forecast future values using simple linear regression
        
        Args:
            dataset_name: Name of the dataset to analyze
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        if dataset_name not in self.time_series_data:
            self.prepare_time_series_data(dataset_name)
        
        ts_data = self.time_series_data[dataset_name]
        
        logger.info(f"Forecasting {periods} periods for {dataset_name}")
        
        forecasts = {}
        
        if dataset_name == 'healthcare_expenditure':
            # Forecast for each country
            for country in ts_data['country'].unique():
                country_data = ts_data[ts_data['country'] == country].copy()
                
                if len(country_data) < 3:  # Need at least 3 points for forecasting
                    continue
                
                # Prepare data
                years = country_data['year'].values.reshape(-1, 1)
                expenditures = country_data['expenditure'].values
                
                # Fit linear model
                model = LinearRegression()
                model.fit(years, expenditures)
                
                # Forecast future years
                last_year = country_data['year'].max()
                future_years = np.arange(last_year + 1, last_year + periods + 1).reshape(-1, 1)
                future_expenditures = model.predict(future_years)
                
                # Calculate forecast accuracy (R²)
                r2 = model.score(years, expenditures)
                
                forecasts[country] = {
                    'model_r2': r2,
                    'forecasts': [
                        {'year': int(year), 'expenditure': float(exp)} 
                        for year, exp in zip(future_years.flatten(), future_expenditures)
                    ]
                }
        
        else:
            # Forecast for other datasets
            time_col = 'time_point'
            value_col = ts_data.columns[1]
            
            # Prepare data
            time_points = ts_data[time_col].values.reshape(-1, 1)
            values = ts_data[value_col].values
            
            # Fit linear model
            model = LinearRegression()
            model.fit(time_points, values)
            
            # Forecast future points
            last_point = ts_data[time_col].max()
            future_points = np.arange(last_point + 1, last_point + periods + 1).reshape(-1, 1)
            future_values = model.predict(future_points)
            
            # Calculate forecast accuracy
            r2 = model.score(time_points, values)
            
            forecasts['general'] = {
                'model_r2': r2,
                'forecasts': [
                    {'time_point': int(point), 'value': float(val)} 
                    for point, val in zip(future_points.flatten(), future_values)
                ]
            }
        
        results = {
            'dataset': dataset_name,
            'forecast_periods': periods,
            'forecasts': forecasts,
            'model_performance': {
                'average_r2': np.mean([f['model_r2'] for f in forecasts.values()]) if forecasts else 0
            }
        }
        
        logger.info(f"Forecasting completed. Average R²: {results['model_performance']['average_r2']:.3f}")
        return results
    
    def create_visualization(self, dataset_name: str, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create visualization of time series analysis
        
        Args:
            dataset_name: Name of the dataset to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.time_series_data:
            self.prepare_time_series_data(dataset_name)
        
        ts_data = self.time_series_data[dataset_name]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        if dataset_name == 'healthcare_expenditure':
            # Plot 1: Time series for top countries
            top_countries = ts_data.groupby('country')['expenditure'].mean().nlargest(5).index
            for country in top_countries:
                country_data = ts_data[ts_data['country'] == country]
                ax1.plot(country_data['year'], country_data['expenditure'], 
                        marker='o', label=country, linewidth=2)
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Healthcare Expenditure (USD per capita)')
            ax1.set_title('Healthcare Expenditure Trends (Top 5 Countries)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Average expenditure over time
            yearly_avg = ts_data.groupby('year')['expenditure'].agg(['mean', 'std'])
            ax2.plot(yearly_avg.index, yearly_avg['mean'], marker='o', linewidth=2)
            ax2.fill_between(yearly_avg.index, 
                           yearly_avg['mean'] - yearly_avg['std'],
                           yearly_avg['mean'] + yearly_avg['std'],
                           alpha=0.3)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Average Healthcare Expenditure')
            ax2.set_title('Global Average Healthcare Expenditure Trend')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Expenditure distribution
            ax3.hist(ts_data['expenditure'], bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Healthcare Expenditure (USD per capita)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Healthcare Expenditure')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Country comparison (box plot)
            top_10_countries = ts_data.groupby('country')['expenditure'].mean().nlargest(10).index
            country_data_list = [ts_data[ts_data['country'] == country]['expenditure'].values 
                               for country in top_10_countries]
            ax4.boxplot(country_data_list, labels=top_10_countries)
            ax4.set_ylabel('Healthcare Expenditure (USD per capita)')
            ax4.set_title('Healthcare Expenditure by Country (Top 10)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        else:
            # Plot for other datasets
            time_col = 'time_point'
            value_col = ts_data.columns[1]
            
            # Plot 1: Time series
            ax1.plot(ts_data[time_col], ts_data[value_col], marker='o', linewidth=2)
            ax1.set_xlabel('Time Point')
            ax1.set_ylabel(value_col.replace('_', ' ').title())
            ax1.set_title(f'{value_col.replace("_", " ").title()} Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Trend line
            time_points = ts_data[time_col].values.reshape(-1, 1)
            values = ts_data[value_col].values
            
            model = LinearRegression()
            model.fit(time_points, values)
            trend_line = model.predict(time_points)
            
            ax2.scatter(ts_data[time_col], ts_data[value_col], alpha=0.6)
            ax2.plot(ts_data[time_col], trend_line, color='red', linewidth=2, label='Trend')
            ax2.set_xlabel('Time Point')
            ax2.set_ylabel(value_col.replace('_', ' ').title())
            ax2.set_title(f'{value_col.replace("_", " ").title()} with Trend Line')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Distribution
            ax3.hist(ts_data[value_col], bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel(value_col.replace('_', ' ').title())
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Distribution of {value_col.replace("_", " ").title()}')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Moving average
            window_size = min(10, len(ts_data) // 4)
            if window_size > 1:
                moving_avg = ts_data[value_col].rolling(window=window_size).mean()
                ax4.plot(ts_data[time_col], ts_data[value_col], alpha=0.3, label='Original')
                ax4.plot(ts_data[time_col], moving_avg, linewidth=2, label=f'Moving Average ({window_size})')
                ax4.set_xlabel('Time Point')
                ax4.set_ylabel(value_col.replace('_', ' ').title())
                ax4.set_title(f'{value_col.replace("_", " ").title()} with Moving Average')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor moving average', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Moving Average Analysis')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, dataset_name: str) -> str:
        """
        Generate comprehensive time series analysis report
        
        Args:
            dataset_name: Name of the analyzed dataset
            
        Returns:
            Formatted report string
        """
        if dataset_name not in self.analysis_results:
            self.analyze_temporal_patterns(dataset_name)
        
        results = self.analysis_results[dataset_name]
        
        report = []
        report.append("=" * 60)
        report.append("TIME SERIES ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION")
        report.append("-" * 20)
        report.append(f"Dataset: {results['dataset']}")
        report.append(f"Total data points: {results['total_data_points']}")
        
        if 'countries_analyzed' in results:
            report.append(f"Countries analyzed: {results['countries_analyzed']}")
        
        report.append("")
        
        # Time range
        report.append("TIME RANGE")
        report.append("-" * 15)
        for key, value in results['time_range'].items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Trend analysis
        report.append("TREND ANALYSIS")
        report.append("-" * 15)
        trend = results['trend_analysis']
        report.append(f"Direction: {trend['direction']}")
        report.append(f"Strength: {trend['strength']}")
        report.append(f"R² Score: {trend['r2_score']:.3f}")
        report.append(f"Slope: {trend['slope']:.3f}")
        report.append("")
        
        # Volatility analysis
        report.append("VOLATILITY ANALYSIS")
        report.append("-" * 20)
        volatility = results['volatility_analysis']
        for key, value in volatility.items():
            if isinstance(value, float):
                report.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Anomaly detection
        try:
            anomalies = self.detect_anomalies(dataset_name)
            report.append("ANOMALY DETECTION")
            report.append("-" * 18)
            report.append(f"Total anomalies: {anomalies['total_anomalies']}")
            report.append(f"Anomaly rate: {anomalies['anomaly_rate']:.1f}%")
            report.append("")
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
        
        # Forecasting
        try:
            forecasts = self.forecast_values(dataset_name)
            report.append("FORECASTING ANALYSIS")
            report.append("-" * 20)
            report.append(f"Forecast periods: {forecasts['forecast_periods']}")
            report.append(f"Average model R²: {forecasts['model_performance']['average_r2']:.3f}")
            report.append("")
        except Exception as e:
            logger.warning(f"Forecasting failed: {e}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
