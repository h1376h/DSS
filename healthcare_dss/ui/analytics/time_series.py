"""
Time Series Analysis Module
"""

import streamlit as st
import pandas as pd
import numpy as np
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_dataset_info, 
    get_available_datasets,
    display_error_message,
    display_success_message,
    display_warning_message,
    get_time_columns,
    create_analysis_summary,
    safe_dataframe_display
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_time_series_analysis():
    """Show time series analysis interface"""
    st.header("Time Series Analysis")
    st.markdown("**Analyze temporal patterns and trends in healthcare data**")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Time Series Analysis Debug", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Time Series Analysis", {
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(st.session_state.data_manager.datasets) if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager else 0,
                "System Initialized": check_system_initialization()
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                st.markdown("---")
                st.subheader("Available Datasets for Time Series Analysis")
                datasets = get_dataset_names()
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        time_cols = get_time_columns(df)
                        st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Time columns: {len(time_cols)} ({time_cols})")
    
    if not check_system_initialization():
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
        return
    
    try:
        datasets = get_available_datasets()
        
        if not datasets:
            display_warning_message("No datasets available. Please load datasets first.")
            return
        
        # Dataset selection
        st.subheader("1. Select Dataset")
        selected_dataset = st.selectbox("Choose a dataset:", list(datasets.keys()), key="ts_dataset")
        
        if selected_dataset:
            dataset = datasets[selected_dataset]
            display_dataset_info(dataset, selected_dataset)
            
            # Time column selection
            st.subheader("2. Select Time Column")
            time_cols = get_time_columns(dataset)
            
            if time_cols:
                time_column = st.selectbox("Select time column:", time_cols, key="ts_time_col")
            else:
                st.warning("No time columns detected. Using index as time series.")
                time_column = None
            
            # Value column selection
            st.subheader("3. Select Value Column")
            numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                value_column = st.selectbox("Select value column:", numeric_cols, key="ts_value_col")
                
                # Analysis type
                st.subheader("4. Select Analysis Type")
                analysis_types = [
                    "Trend Analysis",
                    "Seasonal Decomposition",
                    "Forecasting",
                    "Anomaly Detection",
                    "Stationarity Test"
                ]
                
                selected_analysis = st.selectbox("Choose analysis type:", analysis_types, key="ts_analysis")
                
                # Run analysis
                if st.button("📈 Run Time Series Analysis", type="primary"):
                    with st.spinner("Running time series analysis..."):
                        try:
                            display_success_message("Time series analysis completed!")
                            
                            if selected_analysis == "Trend Analysis":
                                _show_trend_analysis(dataset, value_column)
                            elif selected_analysis == "Seasonal Decomposition":
                                _show_seasonal_decomposition(dataset, value_column)
                            elif selected_analysis == "Forecasting":
                                _show_forecasting(dataset, value_column)
                            elif selected_analysis == "Anomaly Detection":
                                _show_anomaly_detection(dataset, value_column)
                            elif selected_analysis == "Stationarity Test":
                                _show_stationarity_test(dataset, value_column)
                            
                            # Analysis summary
                            create_analysis_summary(len(dataset), len(dataset.columns), f"Time Series - {selected_analysis}")
                            
                        except Exception as e:
                            display_error_message(e, "in time series analysis")
            else:
                st.warning("No numeric columns available for time series analysis.")
    
    except Exception as e:
        display_error_message(e, "accessing datasets")


def _show_trend_analysis(dataset, value_column):
    """Show trend analysis"""
    st.subheader("Trend Analysis")
    
    values = dataset[value_column].dropna().values
    
    if len(values) > 0:
        # Simple linear trend
        x = np.arange(len(values))
        trend_line = np.polyval(np.polyfit(x, values, 1), x)
        
        # Trend metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(values))
        with col2:
            trend_slope = np.polyfit(x, values, 1)[0]
            st.metric("Trend Slope", f"{trend_slope:.4f}")
        with col3:
            trend_direction = "Increasing" if trend_slope > 0 else "Decreasing" if trend_slope < 0 else "Stable"
            st.metric("Trend Direction", trend_direction)
        
        # Simple trend visualization
        trend_data = pd.DataFrame({
            'Index': x,
            'Values': values,
            'Trend': trend_line
        })
        
        st.line_chart(trend_data.set_index('Index'))


def _show_seasonal_decomposition(dataset, value_column):
    """Show seasonal decomposition"""
    st.subheader("Seasonal Decomposition")
    
    values = dataset[value_column].dropna().values
    
    if len(values) > 0:
        # Simple seasonal decomposition (simplified)
        x = np.arange(len(values))
        
        # Trend component
        trend = np.polyval(np.polyfit(x, values, 1), x)
        
        # Detrended data
        detrended = values - trend
        
        # Simple seasonal component (assuming 12-period seasonality)
        seasonal_period = min(12, len(values) // 2)
        if seasonal_period > 1:
            seasonal = np.tile(np.mean(detrended.reshape(-1, seasonal_period), axis=0), 
                             len(values) // seasonal_period + 1)[:len(values)]
        else:
            seasonal = np.zeros_like(values)
        
        # Residual component
        residual = detrended - seasonal
        
        # Display components
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Trend Component:**")
            st.line_chart(pd.DataFrame({'Trend': trend}))
        with col2:
            st.write("**Seasonal Component:**")
            st.line_chart(pd.DataFrame({'Seasonal': seasonal}))


def _show_forecasting(dataset, value_column):
    """Show forecasting"""
    st.subheader("Forecasting")
    
    values = dataset[value_column].dropna().values
    
    if len(values) > 0:
        # Simple linear forecasting
        x = np.arange(len(values))
        poly_coeffs = np.polyfit(x, values, 1)
        
        # Forecast next 5 periods
        forecast_periods = 5
        future_x = np.arange(len(values), len(values) + forecast_periods)
        forecast_values = np.polyval(poly_coeffs, future_x)
        
        # Display forecast
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Forecast Periods", forecast_periods)
        with col2:
            st.metric("Last Value", f"{values[-1]:.2f}")
        
        st.write("**Forecast Values:**")
        forecast_df = pd.DataFrame({
            'Period': range(1, forecast_periods + 1),
            'Forecast': forecast_values
        })
        safe_dataframe_display(forecast_df)


def _show_anomaly_detection(dataset, value_column):
    """Show anomaly detection"""
    st.subheader("Anomaly Detection")
    
    values = dataset[value_column].dropna().values
    
    if len(values) > 0:
        # Simple outlier detection using IQR
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = (values < lower_bound) | (values > upper_bound)
        anomaly_indices = np.where(anomalies)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(values))
        with col2:
            st.metric("Anomalies Detected", len(anomaly_indices))
        with col3:
            anomaly_percentage = (len(anomaly_indices) / len(values)) * 100
            st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")
        
        if len(anomaly_indices) > 0:
            st.write("**Anomaly Details:**")
            anomaly_data = pd.DataFrame({
                'Index': anomaly_indices,
                'Value': values[anomaly_indices],
                'Deviation': np.abs(values[anomaly_indices] - np.median(values))
            })
            safe_dataframe_display(anomaly_data)


def _show_stationarity_test(dataset, value_column):
    """Show stationarity test"""
    st.subheader("Stationarity Test")
    
    values = dataset[value_column].dropna().values
    
    if len(values) > 0:
        # Simple stationarity check using rolling statistics
        window_size = min(10, len(values) // 4)
        
        if window_size > 1:
            rolling_mean = pd.Series(values).rolling(window=window_size).mean()
            rolling_std = pd.Series(values).rolling(window=window_size).std()
            
            # Stationarity metrics
            mean_stability = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else 0
            std_stability = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Stability", f"{mean_stability:.4f}")
            with col2:
                st.metric("Std Stability", f"{std_stability:.4f}")
            with col3:
                is_stationary = mean_stability < 0.1 and std_stability < 0.1
                st.metric("Stationary", "Yes" if is_stationary else "No")
            
            # Rolling statistics plot
            st.write("**Rolling Statistics:**")
            rolling_data = pd.DataFrame({
                'Rolling Mean': rolling_mean,
                'Rolling Std': rolling_std
            })
            st.line_chart(rolling_data)
        else:
            st.warning("Not enough data points for stationarity test.")
