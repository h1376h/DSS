"""
Data Management Dashboard Module
"""

import streamlit as st
import pandas as pd
import numpy as np
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message,
    safe_dataframe_display,
    get_numeric_columns,
    get_dataset_names,
    get_dataset_from_managers
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_data_management():
    """Show data management interface"""
    st.header("Data Management")
    st.markdown("**Manage healthcare datasets and data quality**")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Data Management Debug", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Data Management", {
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(get_dataset_names()) if check_system_initialization() else 0,
                "System Initialized": check_system_initialization()
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                st.markdown("---")
                st.subheader("Data Manager Status")
                st.success("✅ Data Manager Available")
                datasets = get_dataset_names()
                st.write(f"**Total datasets:** {len(datasets)}")
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        numeric_cols = get_numeric_columns(df)
                        st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Numeric columns: {len(numeric_cols)}")
            else:
                st.error("❌ Data Manager Not Available")
    
    # Get datasets from the DSS system
    if check_system_initialization():
        try:
            datasets = get_dataset_names()
            
            st.subheader("Available Datasets")
            for dataset_name in datasets:
                dataset = get_dataset_from_managers(dataset_name)
                if dataset is not None:
                    with st.expander(f"{dataset_name} ({dataset.shape[0]} rows, {dataset.shape[1]} columns)"):
                        st.write(f"**Shape:** {dataset.shape}")
                        st.write(f"**Columns:** {list(dataset.columns)}")
                        st.write(f"**Data Types:**")
                        st.write(dataset.dtypes)
                    
                    if st.button(f"View Sample Data", key=f"view_{dataset_name}"):
                        safe_dataframe_display(dataset, 10)
                        
                    if st.button(f"View Statistics", key=f"stats_{dataset_name}"):
                        # Get numeric columns only for statistics
                        numeric_data = dataset.select_dtypes(include=[np.number])
                        if not numeric_data.empty:
                            safe_dataframe_display(numeric_data.describe(), max_rows=20)
                        else:
                            st.info("No numeric columns available for statistics")
                            
                    # Data quality metrics
                    if st.button(f"Data Quality Report", key=f"quality_{dataset_name}"):
                        st.subheader(f"Data Quality Report for {dataset_name}")
                        
                        # Basic quality metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Rows", dataset.shape[0])
                        with col2:
                            st.metric("Total Columns", dataset.shape[1])
                        with col3:
                            st.metric("Missing Values", dataset.isnull().sum().sum())
                        with col4:
                            st.metric("Duplicate Rows", dataset.duplicated().sum())
                        
                        # Column-wise analysis
                        st.subheader("Column Analysis")
                        column_analysis = []
                        for col in dataset.columns:
                            column_analysis.append({
                                'Column': col,
                                'Data Type': str(dataset[col].dtype),
                                'Missing Values': dataset[col].isnull().sum(),
                                'Missing %': (dataset[col].isnull().sum() / len(dataset)) * 100,
                                'Unique Values': dataset[col].nunique(),
                                'Unique %': (dataset[col].nunique() / len(dataset)) * 100
                            })
                        
                        analysis_df = pd.DataFrame(column_analysis)
                        safe_dataframe_display(analysis_df, max_rows=20)
                        
                        # Data distribution for numeric columns
                        numeric_cols = get_numeric_columns(dataset)
                        if numeric_cols:
                            st.subheader("Numeric Column Distributions")
                            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                                st.write(f"**{col} Distribution:**")
                                st.bar_chart(dataset[col].value_counts().head(10))
                        
        except Exception as e:
            display_error_message(e, "accessing datasets")
    else:
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
