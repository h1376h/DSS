"""
Common utilities and helper functions for Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List


def check_system_initialization() -> bool:
    """Check if the DSS system is properly initialized"""
    return st.session_state.get('initialized', False) and st.session_state.data_manager is not None


def display_dataset_info(dataset: pd.DataFrame, dataset_name: str) -> None:
    """Display standard dataset information"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", dataset.shape[0])
    with col2:
        st.metric("Columns", dataset.shape[1])
    with col3:
        st.metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Show dataset preview
    with st.expander("Dataset Preview"):
        st.dataframe(dataset.head(10))


def get_available_datasets() -> Dict[str, pd.DataFrame]:
    """Get available datasets from both data managers"""
    if not check_system_initialization():
        return {}
    
    all_datasets = {}
    
    # Get datasets from main data manager
    if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
        all_datasets.update(st.session_state.data_manager.datasets)
    
    # Get datasets from dataset manager (additional synthetic datasets)
    if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
        all_datasets.update(st.session_state.dataset_manager.datasets)
    
    return all_datasets


def get_dataset_names() -> List[str]:
    """Get list of available dataset names from both managers"""
    if not check_system_initialization():
        return []
    
    dataset_names = []
    
    # Get dataset names from main data manager
    if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
        dataset_names.extend(list(st.session_state.data_manager.datasets.keys()))
    
    # Get dataset names from dataset manager (additional synthetic datasets)
    if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
        dataset_names.extend(list(st.session_state.dataset_manager.datasets.keys()))
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(dataset_names))


def get_dataset_from_managers(dataset_name: str) -> Optional[pd.DataFrame]:
    """Get a specific dataset from either data manager"""
    if not check_system_initialization():
        return None
    
    # Try main data manager first
    if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
        if dataset_name in st.session_state.data_manager.datasets:
            return st.session_state.data_manager.datasets[dataset_name]
    
    # Try dataset manager (synthetic datasets)
    if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
        if dataset_name in st.session_state.dataset_manager.datasets:
            return st.session_state.dataset_manager.datasets[dataset_name]
    
    return None


def safe_dataset_selection(key: str, help_text: str = "Choose a dataset") -> Optional[str]:
    """Safely get dataset selection with proper error handling"""
    if not check_system_initialization():
        st.error("System not initialized. Please refresh the page.")
        return None
    
    available_datasets = get_dataset_names()
    
    if not available_datasets:
        st.warning("No datasets available. Please load datasets first.")
        return None
    
    return st.selectbox(
        "Select Dataset",
        available_datasets,
        key=key,
        help=help_text
    )


def display_error_message(error: Exception, context: str = "") -> None:
    """Display standardized error messages"""
    st.error(f"Error {context}: {str(error)}")
    if st.session_state.get('debug_mode', False):
        st.code(f"Exception: {type(error).__name__}")

def display_success_message(message: str) -> None:
    """Display standardized success messages"""
    st.success(f"✅ {message}")

def display_warning_message(message: str) -> None:
    """Display standardized warning messages"""
    st.warning(f"⚠️ {message}")

def create_metric_columns(metrics: Dict[str, Any], columns: int = 3) -> None:
    """Display metrics in columns"""
    cols = st.columns(columns)
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            st.metric(key, value)


def safe_dataframe_display(df: pd.DataFrame, max_rows: int = 10, width: str = None, 
                          hide_index: bool = True) -> None:
    """Safely display dataframe with Arrow compatibility and enhanced features"""
    try:
        if df.empty:
            st.info("No data to display")
            return
            
        # Create a copy for display
        display_df = df.head(max_rows).copy()
        
        # Clean the dataframe for Arrow compatibility
        display_df = _clean_dataframe_for_display(display_df)
        
        # Display the cleaned dataframe with enhanced options
        st.dataframe(
            display_df, 
            width=width if width is not None else 'stretch',
            hide_index=hide_index
        )
        
        # Show additional info if debug mode is enabled
        if st.session_state.get('debug_mode', False):
            with st.expander("Debug Info"):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                st.write(f"**Data Types:**")
                st.code(df.dtypes.to_dict())
        
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.code(f"DataFrame shape: {df.shape}")
            st.code(f"DataFrame dtypes: {df.dtypes.to_dict()}")
            # Show raw data as text as fallback
            st.text("Raw data preview:")
            st.text(str(df.head(max_rows)))


def _clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe to ensure Arrow compatibility with enhanced handling"""
    cleaned_df = df.copy()
    
    for col in cleaned_df.columns:
        try:
            # Handle different data types
            if cleaned_df[col].dtype == 'object':
                # Convert object columns to string, handling NaN values and complex objects
                def safe_str_convert(x):
                    try:
                        # Check if it's None or NaN
                        if x is None or (hasattr(x, '__len__') and len(str(x)) == 0):
                            return ''
                        return str(x)
                    except:
                        try:
                            return repr(x)
                        except:
                            return ''
                cleaned_df[col] = cleaned_df[col].apply(safe_str_convert)
            elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Ensure numeric columns don't have mixed types
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                # Replace inf values with NaN, then convert to string for display
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                # Convert to string for PyArrow compatibility
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x) if pd.notna(x) else '')
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                # Convert datetime to string for display
                cleaned_df[col] = cleaned_df[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
            elif cleaned_df[col].dtype.name == 'category':
                # Convert category to string
                cleaned_df[col] = cleaned_df[col].astype(str)
            elif cleaned_df[col].dtype.name == 'bool':
                # Convert boolean to string safely
                cleaned_df[col] = cleaned_df[col].apply(lambda x: 'True' if x is True else 'False' if x is False else '')
            else:
                # Fallback: convert to string
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x) if pd.notna(x) else '')
        except Exception as e:
            # If any column fails, convert entire column to string
            cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x) if pd.notna(x) else '')
    
    # Final cleanup - replace any remaining NaN values
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.fillna('')
    
    return cleaned_df


def get_numeric_columns(dataset: pd.DataFrame) -> List[str]:
    """Get numeric columns from dataset"""
    return dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()


def get_categorical_columns(dataset: pd.DataFrame) -> List[str]:
    """Get categorical columns from dataset"""
    return dataset.select_dtypes(include=['object', 'category']).columns.tolist()


def get_time_columns(dataset: pd.DataFrame) -> List[str]:
    """Get potential time columns from dataset"""
    time_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp']
    return [col for col in dataset.columns 
            if any(keyword in col.lower() for keyword in time_keywords)]


def validate_dataset_selection(datasets: Dict[str, pd.DataFrame]) -> bool:
    """Validate that datasets are available"""
    if not datasets:
        display_warning_message("No datasets available. Please load datasets first.")
        return False
    return True


def create_analysis_summary(data_points: int, features: int, analysis_type: str) -> None:
    """Create standardized analysis summary"""
    st.subheader("Analysis Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analysis Type", analysis_type)
    with col2:
        st.metric("Data Points", data_points)
    with col3:
        st.metric("Features", features)


def safe_display_model_results(results: Dict[str, Any]) -> None:
    """Safely display model training results with Arrow compatibility"""
    try:
        # Convert results to a displayable format
        if isinstance(results, dict):
            # Create a simple results table
            results_df = pd.DataFrame([
                {"Metric": k, "Value": str(v) if not isinstance(v, (int, float)) else f"{v:.4f}"}
                for k, v in results.items()
            ])
            safe_dataframe_display(results_df, max_rows=20)
        else:
            st.write("Results:", results)
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.code(f"Results type: {type(results)}")
            st.code(f"Results content: {str(results)[:500]}...")


def safe_display_feature_importance(importance_data: Dict[str, float]) -> None:
    """Safely display feature importance with Arrow compatibility"""
    try:
        if importance_data:
            # Convert to DataFrame for display
            importance_df = pd.DataFrame([
                {"Feature": feature, "Importance": importance}
                for feature, importance in importance_data.items()
            ])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            safe_dataframe_display(importance_df, max_rows=20)
        else:
            st.info("No feature importance data available")
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.code(f"Importance data: {str(importance_data)[:200]}...")
