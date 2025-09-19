"""
Association Rules Analysis Module
"""

import streamlit as st
import pandas as pd
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_dataset_info, 
    get_available_datasets,
    display_error_message,
    display_success_message,
    display_warning_message,
    get_categorical_columns,
    create_analysis_summary,
    get_dataset_names,
    get_dataset_from_managers,
    safe_dataframe_display
)
from healthcare_dss.ui.utils.data_helpers import (
    create_binary_matrix,
    find_frequent_itemsets,
    generate_association_rules
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_association_rules():
    """Show association rules analysis"""
    st.header("Association Rules Analysis")
    st.markdown("**Discover patterns and relationships in healthcare data**")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Association Rules Debug", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Association Rules", {
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(st.session_state.data_manager.datasets) if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager else 0,
                "System Initialized": check_system_initialization()
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if check_system_initialization():
                st.markdown("---")
                st.subheader("Available Datasets for Association Rules")
                datasets = get_dataset_names()
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        categorical_cols = get_categorical_columns(df)
                        st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Categorical columns: {len(categorical_cols)} ({categorical_cols})")
    
    if not check_system_initialization():
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
        return
    
    try:
        # Get available datasets
        datasets = get_dataset_names()
        
        if not datasets:
            display_warning_message("No datasets available. Please load datasets first.")
            return
        
        # Dataset selection
        st.subheader("1. Select Dataset")
        selected_dataset = st.selectbox("Choose a dataset:", datasets, key="ar_dataset")
        
        if selected_dataset:
            dataset = get_dataset_from_managers(selected_dataset)
            if dataset is not None:
                display_dataset_info(dataset, selected_dataset)
            
            # Select columns for analysis
            st.subheader("2. Select Columns for Analysis")
            categorical_cols = get_categorical_columns(dataset)
            
            if categorical_cols:
                selected_columns = st.multiselect(
                    "Select categorical columns:",
                    categorical_cols,
                    default=categorical_cols[:3] if len(categorical_cols) >= 3 else categorical_cols,
                    key="ar_columns"
                )
            else:
                display_warning_message("No categorical columns found. Association rules work best with categorical data.")
                selected_columns = []
            
            # Parameters
            st.subheader("3. Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01, key="ar_support")
            with col2:
                min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05, key="ar_confidence")
            
            # Run analysis
            if st.button("🔍 Find Association Rules", type="primary"):
                if not selected_columns:
                    st.error("Please select at least one column for analysis.")
                else:
                    with st.spinner("Analyzing association rules..."):
                        try:
                            # Prepare data
                            analysis_data = dataset[selected_columns].copy()
                            
                            # Convert to binary matrix
                            binary_matrix = create_binary_matrix(analysis_data)
                            
                            # Find frequent itemsets
                            frequent_items = find_frequent_itemsets(binary_matrix, min_support)
                            
                            if len(frequent_items) < 2:
                                display_warning_message(f"No frequent itemsets found with minimum support of {min_support}.")
                                return
                            
                            display_success_message(f"Found {len(frequent_items)} frequent items!")
                            
                            # Display frequent items
                            st.subheader("Frequent Items")
                            frequent_df = pd.DataFrame({
                                'Item': frequent_items,
                                'Support': binary_matrix[frequent_items].sum() / len(binary_matrix)
                            }).sort_values('Support', ascending=False)
                            
                            safe_dataframe_display(frequent_df, max_rows=10)
                            
                            # Generate association rules
                            st.subheader("Association Rules")
                            rules = generate_association_rules(
                                binary_matrix, frequent_items, min_confidence, 1.0, 10
                            )
                            
                            if not rules:
                                display_warning_message("No association rules found with the current parameters.")
                            else:
                                display_success_message(f"Found {len(rules)} association rules!")
                                
                                for i, rule in enumerate(rules, 1):
                                    st.write(f"**Rule {i}:** {rule['rule']}")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Support", f"{rule['support']:.3f}")
                                    with col2:
                                        st.metric("Confidence", f"{rule['confidence']:.3f}")
                                    with col3:
                                        st.metric("Lift", f"{rule['lift']:.3f}")
                            
                            # Analysis summary
                            create_analysis_summary(len(dataset), len(selected_columns), "Association Rules")
                        
                        except Exception as e:
                            display_error_message(e, "in association rules analysis")
    
    except Exception as e:
        display_error_message(e, "accessing datasets")
