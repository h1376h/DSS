"""
Visualization utilities for Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from healthcare_dss.ui.utils.common import safe_dataframe_display


def create_histogram(data: pd.Series, title: str = "") -> None:
    """Create histogram visualization"""
    st.subheader(f"Histogram of {title}")
    st.bar_chart(data.value_counts().sort_index())


def create_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Create scatter plot visualization"""
    st.subheader(f"Scatter Plot: {x_col} vs {y_col}")
    st.scatter_chart(data[[x_col, y_col]])


def create_bar_chart(data: pd.DataFrame, category_col: str, value_col: str) -> None:
    """Create bar chart visualization"""
    st.subheader(f"Bar Chart: {category_col} by {value_col}")
    bar_data = data.groupby(category_col)[value_col].mean().reset_index()
    st.bar_chart(bar_data.set_index(category_col))


def create_line_chart(data: pd.DataFrame, columns: List[str]) -> None:
    """Create line chart visualization"""
    st.subheader("Line Chart")
    st.line_chart(data[columns])


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    """Create correlation heatmap"""
    st.subheader("Correlation Heatmap")
    safe_dataframe_display(corr_matrix)


def create_cluster_visualization(data: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               feature1: str, 
                               feature2: str,
                               algorithm: str) -> None:
    """Create cluster visualization"""
    st.subheader("Cluster Visualization")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot clusters
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_data = data[cluster_labels == cluster_id]
            ax.scatter(cluster_data[feature1], cluster_data[feature2], 
                     c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            cluster_data = data[cluster_labels == cluster_id]
            ax.scatter(cluster_data[feature1], cluster_data[feature2], 
                     c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7)
    
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f'Clustering Results ({algorithm.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def create_trend_plot(data: pd.DataFrame, 
                     values: np.ndarray, 
                     trend_line: np.ndarray,
                     metric_name: str) -> None:
    """Create trend analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data.index, values, label='Actual', alpha=0.7)
    ax.plot(data.index, trend_line, label='Trend', color='red', linewidth=2)
    
    ax.set_title(f'{metric_name} - Trend Analysis')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def create_seasonal_decomposition(data: pd.DataFrame, 
                                values: np.ndarray,
                                trend: pd.Series,
                                seasonal: np.ndarray,
                                metric_name: str) -> None:
    """Create seasonal decomposition plot"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(data.index, values, label='Original')
    axes[0].set_title(f'{metric_name} - Original Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(data.index, trend, label='Trend', color='red')
    axes[1].set_title('Trend Component')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(data.index, seasonal, label='Seasonal', color='green')
    axes[2].set_title('Seasonal Component')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def create_forecast_plot(data: pd.DataFrame,
                        values: np.ndarray,
                        forecast_values: np.ndarray,
                        forecast_periods: int,
                        metric_name: str) -> None:
    """Create forecasting plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data.index, values, label='Historical', color='blue')
    ax.plot(pd.date_range(start=data.index[-1], periods=forecast_periods+1, freq='D')[1:], 
           forecast_values, label='Forecast', color='red', linestyle='--')
    
    ax.set_title(f'{metric_name} - Forecast ({forecast_periods} periods)')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def create_anomaly_plot(data: pd.DataFrame,
                       values: np.ndarray,
                       anomaly_indices: np.ndarray,
                       metric_name: str,
                       threshold: float) -> None:
    """Create anomaly detection plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data.index, values, label='Normal', color='blue', alpha=0.7)
    if len(anomaly_indices) > 0:
        ax.scatter(data.index[anomaly_indices], values[anomaly_indices], 
                 color='red', s=50, label='Anomalies', zorder=5)
    
    ax.set_title(f'{metric_name} - Anomaly Detection (Threshold: {threshold})')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def create_feature_importance_plot(importance_data: Dict[str, float]) -> None:
    """Create feature importance plot"""
    st.subheader("Feature Importance")
    
    importance_df = pd.DataFrame(
        list(importance_data.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))


def create_performance_metrics_chart(metrics: Dict[str, float]) -> None:
    """Create performance metrics chart"""
    st.subheader("Model Performance")
    
    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=['Metric', 'Value']
    )
    
    safe_dataframe_display(metrics_df)
