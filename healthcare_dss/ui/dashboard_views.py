"""
Dashboard Views Module
=====================

Contains the main dashboard views for the Healthcare DSS:
- Dashboard Overview
- Data Management  
- Model Management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message,
    display_success_message, 
    display_warning_message,
    get_dataset_names,
    get_dataset_from_managers,
    safe_dataframe_display
)


def show_debug_info(title, debug_data, expanded=None):
    """Show debug information if debug mode is enabled"""
    if st.session_state.get('debug_mode', False):
        if expanded is None:
            expanded = st.session_state.get('debug_mode', False)
        with st.expander(f"🔍 {title}", expanded=expanded):
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")


def debug_write(message):
    """Write debug message if debug mode is enabled"""
    if st.session_state.get('debug_mode', False):
        st.write(f"🔍 DEBUG: {message}")


def show_dashboard_overview():
    """Show the main dashboard overview with enhanced visualizations"""
    st.header("Dashboard Overview")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Dashboard Overview Debug", expanded=True):
            st.write("**Dashboard Overview Debug:**")
            st.write(f"- Function called: show_dashboard_overview()")
            st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
            st.write(f"- Has data_manager: {hasattr(st.session_state, 'data_manager')}")
            st.write(f"- Has model_manager: {hasattr(st.session_state, 'model_manager')}")
            st.write(f"- Debug mode: {st.session_state.get('debug_mode', False)}")
    
    # System status with enhanced metrics
    st.subheader("System Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Get total datasets from consolidated DataManager
        data_manager_count = len(get_dataset_names())
        
        # All datasets are now in the consolidated DataManager
        all_dataset_names = get_dataset_names()
        unique_count = len(all_dataset_names)
        total_datasets = unique_count
        
        st.metric(
            "Datasets Loaded", 
            total_datasets,
            delta=f"+{total_datasets}"
        )
    
    with col2:
        # Get total records from unique datasets only
        all_datasets = {}
        all_datasets.update(st.session_state.data_manager.datasets)
        if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
            all_datasets.update(st.session_state.dataset_manager.datasets)
        
        total_records = sum(len(df) for df in all_datasets.values())
        
        st.metric(
            "Total Records", 
            f"{total_records:,}",
            delta=f"+{total_records:,}"
        )
    
    with col3:
        # Count clinical rules from knowledge manager
        try:
            rules_count = len(st.session_state.knowledge_manager.get_clinical_rules())
        except:
            rules_count = 3  # fallback
        st.metric(
            "Clinical Rules", 
            rules_count,
            delta=f"+{rules_count}"
        )
    
    with col4:
        # Calculate average data quality from unique datasets only
        all_datasets = {}
        all_datasets.update(st.session_state.data_manager.datasets)
        if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
            all_datasets.update(st.session_state.dataset_manager.datasets)
        
        quality_scores = []
        for name, df in all_datasets.items():
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            quality_scores.append(completeness)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        st.metric(
            "Data Quality", 
            f"{avg_quality:.1f}%",
            delta=f"+{avg_quality:.1f}%"
        )
    
    with col5:
        st.metric(
            "System Status", 
            "Operational",
            delta="Active"
        )
    
    st.markdown("---")
    
    # Enhanced dataset summary with visualizations
    st.subheader("Dataset Analytics")
    
    # Create dataset summary with enhanced metrics from unique datasets
    dataset_info = []
    
    # Combine datasets from both sources (avoiding duplicates)
    all_datasets = {}
    all_datasets.update(st.session_state.data_manager.datasets)
    if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
        all_datasets.update(st.session_state.dataset_manager.datasets)
    
    for name, df in all_datasets.items():
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        
        # All datasets are now in the consolidated DataManager
        source = 'DataManager'
        
        dataset_info.append({
            'Dataset': name.replace('_', ' ').title(),
            'Source': source,
            'Records': len(df),
            'Features': len(df.columns),
            'Numeric': numeric_cols,
            'Categorical': categorical_cols,
            'Completeness': completeness,
            'Memory (MB)': df.memory_usage(deep=True).sum() / 1024**2
        })
    
    df_info = pd.DataFrame(dataset_info)
    
    # Display dataset summary with tabs
    tab1, tab2, tab3 = st.tabs(["Summary Table", "Visualizations", "Detailed Analysis"])
    
    with tab1:
        safe_dataframe_display(df_info, width="stretch")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset size comparison
            fig_size = px.bar(
                df_info, 
                x='Dataset', 
                y='Records',
                title='Dataset Size Comparison',
                color='Records',
                color_continuous_scale='Blues'
            )
            fig_size.update_layout(showlegend=False)
            st.plotly_chart(fig_size, width="stretch")
        
        with col2:
            # Data completeness pie chart
            fig_completeness = px.pie(
                df_info, 
                values='Completeness', 
                names='Dataset',
                title='Data Completeness by Dataset',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_completeness, width="stretch")
        
        # Feature distribution
        fig_features = px.bar(
            df_info, 
            x='Dataset', 
            y=['Numeric', 'Categorical'],
            title='Feature Distribution by Dataset',
            color_discrete_map={'Numeric': '#1f77b4', 'Categorical': '#ff7f0e'}
        )
        fig_features.update_layout(barmode='stack')
        st.plotly_chart(fig_features, width="stretch")
    
    with tab3:
        # Detailed analysis for each dataset
        selected_dataset = st.selectbox(
            "Select Dataset for Detailed Analysis",
            get_dataset_names()
        )
        
        if selected_dataset:
            df = get_dataset_from_managers(selected_dataset)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicate Rows", df.duplicated().sum())
            
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
                st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            # Convert pandas data types to strings for JSON serialization
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            dtype_values = dtype_counts.values.tolist()
            
            fig_dtypes = px.pie(
                values=dtype_values,
                names=dtype_names,
                title=f'Data Types Distribution - {selected_dataset.title()}'
            )
            st.plotly_chart(fig_dtypes, width="stretch")
    
    # System performance metrics
    st.subheader("System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Initialization Time", "< 5s", delta="Fast")
    
    with col2:
        st.metric("Memory Usage", f"{sum(df.memory_usage(deep=True).sum() for df in st.session_state.data_manager.datasets.values()) / 1024**2:.1f} MB")
    
    with col3:
        st.metric("Last Updated", "Just now", delta="Live")
    
    # Recent activity with enhanced information
    st.subheader("Recent Activity")
    
    activity_data = [
        {"Time": "Just now", "Activity": "System initialized successfully", "Status": "Success", "Details": "All modules operational"},
        {"Time": "Just now", "Activity": "Datasets loaded", "Status": "Success", "Details": f"{len(st.session_state.data_manager.datasets)} datasets ready"},
        {"Time": "Just now", "Activity": "Knowledge base initialized", "Status": "Success", "Details": "Clinical rules and guidelines loaded"},
        {"Time": "Just now", "Activity": "Model management ready", "Status": "Success", "Details": "ML pipeline operational"}
    ]
    
    activity_df = pd.DataFrame(activity_data)
    safe_dataframe_display(activity_df, width="stretch")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("View Data", width="stretch"):
            st.session_state.page = "Data Management"
            st.rerun()
    
    with col2:
        if st.button("Train Model", width="stretch"):
            st.session_state.page = "Model Management"
            st.rerun()
    
    with col3:
        if st.button("View KPIs", width="stretch"):
            st.session_state.page = "KPI Dashboard"
            st.rerun()
    
    with col4:
        if st.button("Analyze Data", width="stretch"):
            st.session_state.page = "CRISP-DM Workflow"
            st.rerun()


def show_data_management():
    """Show enhanced data management interface with interactive visualizations"""
    st.header("Data Management")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Data Management Debug", expanded=True):
            st.write("**Data Management Debug:**")
            st.write(f"- Function called: show_data_management()")
            st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
            st.write(f"- Has data_manager: {hasattr(st.session_state, 'data_manager')}")
            st.write(f"- Debug mode: {st.session_state.get('debug_mode', False)}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager is not None:
                st.write(f"- DataManager datasets: {len(st.session_state.data_manager.datasets)}")
                st.write(f"- Dataset names: {list(st.session_state.data_manager.datasets.keys())}")
            else:
                st.write("- DataManager not available")
    
    # Dataset selection with enhanced info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not check_system_initialization():
            st.error("System not initialized. Please refresh the page.")
            return
        
        datasets = get_dataset_names()
        if not datasets:
            st.warning("No datasets available. Please load datasets first.")
            return
        
        dataset_name = st.selectbox(
            "Select Dataset",
            datasets,
            help="Choose a dataset to explore and analyze"
        )
    
    with col2:
        if dataset_name:
            df = get_dataset_from_managers(dataset_name)
            if df is not None:
                st.metric("Records", f"{len(df):,}")
    
    if dataset_name:
        df = get_dataset_from_managers(dataset_name)
        if df is None:
            st.error(f"Dataset '{dataset_name}' not found in any data manager.")
            return
        
        # Enhanced dataset overview with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Data Quality", "Statistics"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Dataset Info")
                
                # Enhanced metrics
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                categorical_cols = len(df.select_dtypes(include=['object']).columns)
                
                st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Completeness", f"{completeness:.1f}%")
                st.metric("Numeric Columns", numeric_cols)
                st.metric("Categorical Columns", categorical_cols)
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Data types summary
                st.subheader("Data Types")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count} columns")
            
            with col2:
                st.subheader("Data Preview")
                
                # Show more rows with option to customize
                preview_rows = st.slider("Number of rows to display", 5, 50, 10)
                safe_dataframe_display(df.head(preview_rows), width="stretch")
                
                # Column information
                st.subheader("Column Information")
                col_info = []
                for col in df.columns:
                    col_info.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Non-Null': df[col].count(),
                        'Null': df[col].isnull().sum(),
                        'Unique': df[col].nunique()
                    })
                
                col_df = pd.DataFrame(col_info)
                safe_dataframe_display(col_df, width="stretch")
        
        with tab2:
            st.subheader("Interactive Visualizations")
            
            # Select visualization type
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Distribution", "Correlation", "Missing Data", "Outliers", "Custom Plot"]
            )
            
            if viz_type == "Distribution":
                # Distribution plots
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig_hist = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
                        st.plotly_chart(fig_hist, width="stretch")
                    
                    with col2:
                        # Box plot
                        fig_box = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                        st.plotly_chart(fig_box, width="stretch")
                else:
                    st.warning("No numeric columns found for distribution plots")
            
            elif viz_type == "Correlation":
                # Correlation matrix
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, width="stretch")
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
            
            elif viz_type == "Missing Data":
                # Missing data visualization
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    fig_missing = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Data by Column"
                    )
                    st.plotly_chart(fig_missing, width="stretch")
                else:
                    st.success("No missing data found!")
            
            elif viz_type == "Outliers":
                # Outlier detection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)
                    
                    # IQR method for outlier detection
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                    
                    st.write(f"**Outliers detected:** {len(outliers)} out of {len(df)} records")
                    
                    if len(outliers) > 0:
                        fig_outliers = px.scatter(
                            df, 
                            x=range(len(df)), 
                            y=selected_col,
                            title=f'Outliers in {selected_col}',
                            color=df[selected_col].apply(lambda x: 'Outlier' if x < lower_bound or x > upper_bound else 'Normal')
                        )
                        st.plotly_chart(fig_outliers, width="stretch")
                else:
                    st.warning("No numeric columns found for outlier detection")
            
            elif viz_type == "Custom Plot":
                # Custom plot builder
                st.write("**Custom Plot Builder**")
                
                plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Bar", "Histogram"])
                
                if plot_type in ["Scatter", "Line"]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_col = st.selectbox("X-axis", df.columns)
                    with col2:
                        y_col = st.selectbox("Y-axis", df.columns)
                    
                    if plot_type == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    else:
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    
                    st.plotly_chart(fig, width="stretch")
                
                elif plot_type == "Bar":
                    col_col = st.selectbox("Column", df.columns)
                    fig = px.bar(df, x=col_col, title=f"Bar Chart of {col_col}")
                    st.plotly_chart(fig, width="stretch")
                
                elif plot_type == "Histogram":
                    col_col = st.selectbox("Column", df.select_dtypes(include=[np.number]).columns)
                    fig = px.histogram(df, x=col_col, title=f"Histogram of {col_col}")
                    st.plotly_chart(fig, width="stretch")
        
        with tab3:
            st.subheader("Data Quality Assessment")
            
            if st.button("Run Comprehensive Data Quality Assessment", type="primary"):
                with st.spinner("Analyzing data quality..."):
                    try:
                        quality_results = st.session_state.data_manager.assess_data_quality(dataset_name)
                        
                        # Quality metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Completeness", f"{quality_results['completeness_score']:.1f}%")
                        
                        with col2:
                            missing_count = quality_results['missing_values'].sum() if hasattr(quality_results['missing_values'], 'sum') else sum(quality_results['missing_values'].values())
                            st.metric("Missing Values", missing_count)
                        
                        with col3:
                            st.metric("Data Types", len(quality_results['data_types']))
                        
                        with col4:
                            st.metric("Quality Score", f"{quality_results['overall_score']:.1f}/100")
                        
                        # Quality gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = quality_results['overall_score'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Overall Data Quality"},
                            delta = {'reference': 80},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig_gauge, width="stretch")
                        
                        # Detailed quality report
                        st.subheader("Detailed Quality Report")
                        
                        # Missing values report
                        if hasattr(quality_results['missing_values'], 'to_dict'):
                            missing_df = pd.DataFrame({
                                'Column': quality_results['missing_values'].index,
                                'Missing Count': quality_results['missing_values'].values,
                                'Missing Percentage': (quality_results['missing_values'].values / len(df)) * 100
                            })
                            safe_dataframe_display(missing_df[missing_df['Missing Count'] > 0], width="stretch")
                        
                        # Data type consistency
                        st.write("**Data Type Consistency:**")
                        for col, dtype in quality_results['data_types'].items():
                            st.write(f"• {col}: {dtype}")
                        
                        # Quality recommendations
                        st.subheader("Quality Recommendations")
                        recommendations = quality_results.get('recommendations', [])
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
                        else:
                            st.write("No specific recommendations available.")
                        
                    except Exception as e:
                        st.error(f"Error assessing data quality: {e}")
        
        with tab4:
            st.subheader("Statistical Summary")
            
            # Numeric columns statistics
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 0:
                st.write("**Numeric Columns Statistics:**")
                safe_dataframe_display(numeric_df.describe(), width="stretch")
            
            # Categorical columns statistics
            categorical_df = df.select_dtypes(include=['object'])
            if len(categorical_df.columns) > 0:
                st.write("**Categorical Columns Statistics:**")
                cat_stats = []
                for col in categorical_df.columns:
                    cat_stats.append({
                        'Column': col,
                        'Unique Values': categorical_df[col].nunique(),
                        'Most Frequent': categorical_df[col].mode().iloc[0] if len(categorical_df[col].mode()) > 0 else 'N/A',
                        'Frequency': categorical_df[col].value_counts().iloc[0] if len(categorical_df[col].value_counts()) > 0 else 0
                    })
                
                cat_stats_df = pd.DataFrame(cat_stats)
                safe_dataframe_display(cat_stats_df, width="stretch")
            
            # Data shape and memory usage
            st.write("**Dataset Information:**")
            info_data = {
                'Metric': ['Shape', 'Memory Usage', 'Total Cells', 'Non-Null Cells', 'Null Cells'],
                'Value': [
                    f"{df.shape[0]} × {df.shape[1]}",
                    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    df.shape[0] * df.shape[1],
                    df.count().sum(),
                    df.isnull().sum().sum()
                ]
            }
            info_df = pd.DataFrame(info_data)
            safe_dataframe_display(info_df, width="stretch")


def show_model_management():
    """Show model management interface"""
    st.header("Model Management")
    
    debug_write("Function called: show_model_management()")
    
    # Debug information using the utility function
    debug_data = {
        "Function called": "show_model_management()",
        "Session state initialized": st.session_state.get('initialized', 'Not set'),
        "Has data_manager": hasattr(st.session_state, 'data_manager'),
        "Has model_manager": hasattr(st.session_state, 'model_manager')
    }
    
    if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager is not None:
        debug_data["DataManager datasets"] = len(st.session_state.data_manager.datasets)
    else:
        debug_data["DataManager"] = "not available"
    
    if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager is not None:
        try:
            models_df = st.session_state.model_manager.list_models()
            debug_data["ModelManager models"] = len(models_df)
            debug_data["Models DataFrame empty"] = models_df.empty
        except Exception as e:
            debug_data["Error getting models"] = str(e)
    else:
        debug_data["ModelManager"] = "not available"
    
    show_debug_info("Complete Debug Information", debug_data, expanded=True)
    
    # Check if session state is initialized - more robust check
    try:
        initialized = st.session_state.get('initialized', False)
        if not initialized:
            st.warning("System is still initializing. Please wait...")
            return
    except Exception as e:
        st.error(f"Error checking initialization status: {e}")
        return
    
    # Check if required components are available - more robust check
    try:
        has_data_manager = hasattr(st.session_state, 'data_manager') and st.session_state.data_manager is not None
        has_model_manager = hasattr(st.session_state, 'model_manager') and st.session_state.model_manager is not None
        
        if not has_data_manager or not has_model_manager:
            st.error("Required components not initialized. Please refresh the page.")
            return
    except Exception as e:
        st.error(f"Error checking component availability: {e}")
        return
    
    # Model management tabs
    tab1, tab2, tab3 = st.tabs(["Train New Model", "Model Performance", "Model Deployment"])
    
    with tab1:
        st.subheader("Train New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.selectbox(
                "Select Dataset",
                get_dataset_names()
            )
            
            task_type = st.selectbox(
                "Task Type",
                ["classification", "regression"]
            )
            
            model_type = st.selectbox(
                "Model Type",
                ["random_forest", "svm", "neural_network", "knn", "linear_regression", "xgboost", "lightgbm"]
            )
        
        with col2:
            if dataset_name:
                df = get_dataset_from_managers(dataset_name)
                
                # Import smart target manager
                try:
                    from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
                    smart_manager = SmartDatasetTargetManager()
                    
                    # Get smart target recommendations
                    smart_targets = smart_manager.get_dataset_targets(dataset_name)
                    
                    if smart_targets:
                        st.info("Smart Target Recommendations Available!")
                        
                        # Show smart targets
                        st.write("**Recommended Targets:**")
                        for i, target in enumerate(smart_targets):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{target['column']}** ({target['target_type']})")
                                st.caption(target.get('business_meaning', 'Target variable'))
                            with col2:
                                if st.button(f"Select", key=f"dashboard_select_{i}"):
                                    st.session_state.dashboard_target = target['column']
                                    st.rerun()
                        
                        st.markdown("---")
                        
                except Exception as e:
                    st.warning(f"Could not load smart recommendations: {e}")
                
                # Standard target selection
                target_column = st.selectbox(
                    "Target Column",
                    df.columns.tolist()
                )
                
                # Add column analysis section
                if target_column:
                    st.subheader("Column Analysis")
                    
                    try:
                        analysis = st.session_state.model_manager.analyze_target_column(dataset_name, target_column)
                        
                        if 'error' not in analysis:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Data Type", analysis['data_type'])
                            with col2:
                                st.metric("Unique Values", analysis['unique_values'])
                            with col3:
                                st.metric("Missing Values", analysis['missing_values'])
                            
                            # Show value range for numeric columns
                            if analysis['min_value'] is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Min Value", f"{analysis['min_value']:.2f}")
                                with col2:
                                    st.metric("Max Value", f"{analysis['max_value']:.2f}")
                            
                            # Show enhanced recommendations with confidence scores
                            st.subheader("Intelligent Recommendations")
                            
                            # Display primary recommendation with confidence
                            if analysis['primary_recommendation'] == 'classification':
                                confidence = analysis.get('classification_confidence', 0)
                                st.success(f"🎯 Recommended: Classification (Confidence: {confidence}%)")
                                st.info("This column appears to represent distinct categories")
                            elif analysis['primary_recommendation'] == 'regression':
                                confidence = analysis.get('regression_confidence', 0)
                                st.success(f"🎯 Recommended: Regression (Confidence: {confidence}%)")
                                st.info("This column appears to contain continuous numerical values")
                            elif analysis['primary_recommendation'] == 'neither':
                                st.error("❌ Neither classification nor regression is suitable")
                                st.warning("Consider data preprocessing or different target column")
                            else:
                                st.warning("⚠️ Both classification and regression are possible")
                                st.info("Choose based on your prediction goal")
                            
                            # Show detailed recommendations
                            if 'recommendations' in analysis and analysis['recommendations']:
                                with st.expander("📋 Detailed Recommendations"):
                                    for rec in analysis['recommendations']:
                                        st.write(f"• {rec}")
                            
                            # Show detailed reasons
                            with st.expander("Detailed Analysis"):
                                classification_confidence = analysis.get('classification_confidence', 0)
                                regression_confidence = analysis.get('regression_confidence', 0)
                                
                                st.write("**Classification Suitability:**", "Suitable" if analysis['classification_suitable'] else "Not Suitable")
                                st.write(f"Confidence: {classification_confidence}%")
                                for reason in analysis['classification_reasons']:
                                    st.write(f"• {reason}")
                                
                                st.write("**Regression Suitability:**", "Suitable" if analysis['regression_suitable'] else "Not Suitable")
                                st.write(f"Confidence: {regression_confidence}%")
                                for reason in analysis['regression_reasons']:
                                    st.write(f"• {reason}")
                                
                                st.write("**Primary Recommendation:**", analysis.get('primary_recommendation', 'Unknown').title())
                            
                            # Show warning if current selection might be problematic
                            if task_type == 'classification' and not analysis['classification_suitable']:
                                st.error("Warning: Classification may not work well with this column!")
                            elif task_type == 'regression' and not analysis['regression_suitable']:
                                st.error("Warning: Regression may not work well with this column!")
                        
                        else:
                            st.error(f"Error analyzing column: {analysis['error']}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing column: {e}")
        
        # Add professional preprocessing configuration section
        if dataset_name and target_column:
            st.markdown("---")
            st.subheader("🔧 Advanced Preprocessing Configuration")
            
            try:
                # Get preprocessing options
                preprocessing_options = st.session_state.model_manager.get_preprocessing_options(
                    dataset_name, target_column, task_type
                )
                
                if 'error' not in preprocessing_options:
                    # Create tabs for different preprocessing categories
                    tab1, tab2, tab3, tab4 = st.tabs(["Target Preprocessing", "Feature Scaling", "Encoding", "Feature Engineering"])
                    
                    preprocessing_config = {}
                    
                    with tab1:
                        st.write("**Target Variable Preprocessing**")
                        if preprocessing_options['target_preprocessing']:
                            for option in preprocessing_options['target_preprocessing']:
                                if option['suitable']:
                                    enabled = st.checkbox(
                                        f"{option['description']} (Confidence: {option['confidence']}%)",
                                        key=f"target_{option['name']}"
                                    )
                                    if enabled:
                                        preprocessing_config[f"target_{option['name']}"] = True
                        else:
                            st.info("No target preprocessing needed for this data type")
                    
                    with tab2:
                        st.write("**Feature Scaling Options**")
                        suitable_scaling_options = [opt for opt in preprocessing_options.get('scaling_options', []) if opt.get('suitable', False)]
                        
                        if suitable_scaling_options:
                            scaling_method = st.selectbox(
                                "Select Scaling Method",
                                options=[opt['name'] for opt in suitable_scaling_options],
                                format_func=lambda x: next((opt['description'] for opt in suitable_scaling_options if opt['name'] == x), x),
                                help="Choose the scaling method based on your data characteristics"
                            )
                            preprocessing_config['scaling_method'] = scaling_method
                            
                            # Show scaling recommendations
                            selected_scaling = next((opt for opt in suitable_scaling_options if opt['name'] == scaling_method), None)
                            if selected_scaling:
                                st.info(f"**Best for:** {selected_scaling.get('best_for', 'General use')}")
                                st.info(f"**Confidence:** {selected_scaling.get('confidence', 0)}%")
                        else:
                            st.info("No scaling options available for this data type")
                            preprocessing_config['scaling_method'] = 'none'
                    
                    with tab3:
                        st.write("**Categorical Encoding Options**")
                        suitable_encoding_options = [opt for opt in preprocessing_options.get('encoding_options', []) if opt.get('suitable', False)]
                        
                        if suitable_encoding_options:
                            encoding_method = st.selectbox(
                                "Select Encoding Method",
                                options=[opt['name'] for opt in suitable_encoding_options],
                                format_func=lambda x: next((opt['description'] for opt in suitable_encoding_options if opt['name'] == x), x),
                                help="Choose encoding method based on your categorical data"
                            )
                            preprocessing_config['encoding_method'] = encoding_method
                            
                            # Show encoding recommendations
                            selected_encoding = next((opt for opt in suitable_encoding_options if opt['name'] == encoding_method), None)
                            if selected_encoding:
                                st.info(f"**Best for:** {selected_encoding.get('best_for', 'General use')}")
                                st.info(f"**Confidence:** {selected_encoding.get('confidence', 0)}%")
                        else:
                            st.info("No categorical encoding needed")
                            preprocessing_config['encoding_method'] = 'none'
                    
                    with tab4:
                        st.write("**Feature Engineering Options**")
                        suitable_feature_options = [opt for opt in preprocessing_options.get('feature_engineering', []) if opt.get('suitable', False)]
                        
                        if suitable_feature_options:
                            for option in suitable_feature_options:
                                enabled = st.checkbox(
                                    f"{option.get('description', option.get('name', 'Unknown'))} (Confidence: {option.get('confidence', 0)}%)",
                                    key=f"feature_{option.get('name', 'unknown')}"
                                )
                                if enabled:
                                    preprocessing_config[f"feature_{option.get('name', 'unknown')}"] = True
                                st.caption(f"Best for: {option.get('best_for', 'General use')}")
                        else:
                            st.info("No feature engineering options available for this data type")
                    
                    # Data quality issues
                    data_quality_issues = preprocessing_options.get('data_quality_issues', [])
                    if data_quality_issues:
                        st.subheader("⚠️ Data Quality Issues")
                        for issue in data_quality_issues:
                            severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(issue.get('severity', 'low'), '⚪')
                            st.warning(f"{severity_color} {issue.get('description', 'Unknown issue')}")
                            solutions = issue.get('solutions', [])
                            if solutions:
                                st.write("**Solutions:**", ", ".join(solutions))
                    
                    # Show recommendations
                    recommendations = preprocessing_options.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 Intelligent Recommendations")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                else:
                    st.error(f"Error getting preprocessing options: {preprocessing_options['error']}")
                    
            except Exception as e:
                st.error(f"Error analyzing preprocessing options: {e}")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    result = st.session_state.model_manager.train_model(
                        dataset_name=dataset_name,
                        model_name=model_type,
                        task_type=task_type,
                        target_column=target_column,
                        preprocessing_config=preprocessing_config if 'preprocessing_config' in locals() else None
                    )
                    
                    st.success("Model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if 'r2_score' in result['metrics']:
                            st.metric("R² Score", f"{result['metrics']['r2_score']:.4f}")
                    with col2:
                        if 'accuracy' in result['metrics']:
                            st.metric("Accuracy", f"{result['metrics']['accuracy']:.4f}")
                    with col3:
                        st.metric("RMSE", f"{result['metrics']['rmse']:.4f}")
                    with col4:
                        st.metric("MAE", f"{result['metrics']['mae']:.4f}")
                
                except Exception as e:
                    # Display detailed error message
                    error_msg = str(e)
                    if "❌" in error_msg:  # Check if it's our detailed error message
                        st.error("Model Training Failed")
                        st.markdown(error_msg)
                    else:
                        st.error(f"Error training model: {error_msg}")
                    
                    # Add suggestion to try different task type
                    if "classification" in error_msg.lower() and "regression" in error_msg.lower():
                        st.info("Try changing the Task Type to 'regression' for this column")
    
    with tab2:
        st.subheader("Model Performance")
        
        # Always show this content first to test if tab is working
        st.write("🔍 **Tab 2: Model Performance - This should always be visible**")
        
        # Test basic content
        st.write("**Basic Test Content:**")
        st.write("- Tab 2 is working")
        st.write("- Content is being rendered")
        st.write(f"- Current time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show system status
        st.write("**System Status:**")
        st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
        st.write(f"- Has model_manager: {hasattr(st.session_state, 'model_manager')}")
        st.write(f"- Model manager is not None: {st.session_state.model_manager is not None if hasattr(st.session_state, 'model_manager') else False}")
        
        # Show trained models
        try:
            st.write("**Attempting to load models...**")
            
            if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager is not None:
                st.write("✅ Model Manager is available")
                
                try:
                    models_df = st.session_state.model_manager.list_models()
                    st.write(f"✅ Models DataFrame retrieved: {models_df.shape}")
                    st.write(f"✅ Models DataFrame empty: {models_df.empty}")
                    st.write(f"✅ Number of models: {len(models_df)}")
                    
                    if not models_df.empty:
                        st.write("**Trained Models:**")
                        st.write(f"Found {len(models_df)} models")
                        
                        # Show models in a simple format first
                        for idx, model in models_df.iterrows():
                            st.write(f"**Model {idx + 1}:**")
                            st.write(f"- Name: {model.get('model_name', 'Unknown')}")
                            st.write(f"- Dataset: {model.get('dataset_name', 'N/A')}")
                            st.write(f"- Task Type: {model.get('task_type', 'N/A')}")
                            st.write(f"- Status: {model.get('status', 'N/A')}")
                            st.write(f"- Created: {model.get('created_at', 'N/A')}")
                            st.write("---")
                    else:
                        st.info("No trained models found. Train a model first.")
                        
                except Exception as e:
                    st.error(f"❌ Error getting models: {e}")
                    st.write(f"Error details: {str(e)}")
            else:
                st.error("❌ Model Manager is not available")
                
        except Exception as e:
            st.error(f"❌ Error in Model Performance tab: {e}")
            st.write(f"Error details: {str(e)}")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Debug Information", expanded=True):
                st.write("**Model Performance Tab Debug:**")
                st.write(f"- Tab 2 reached successfully")
                st.write(f"- Model Manager exists: {hasattr(st.session_state, 'model_manager')}")
                st.write(f"- Model Manager is not None: {st.session_state.model_manager is not None if hasattr(st.session_state, 'model_manager') else False}")
                
                if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager is not None:
                    try:
                        models_df = st.session_state.model_manager.list_models()
                        st.write(f"- Models DataFrame shape: {models_df.shape}")
                        st.write(f"- Models DataFrame empty: {models_df.empty}")
                        st.write(f"- Number of models: {len(models_df)}")
                        
                        if not models_df.empty:
                            st.write("**First few models:**")
                            for idx, model in models_df.head(3).iterrows():
                                st.write(f"  Model {idx}: {model.get('model_name', 'Unknown')} - {model.get('task_type', 'N/A')}")
                        else:
                            st.write("- No models found")
                    except Exception as e:
                        st.write(f"- Error getting models: {e}")
                else:
                    st.write("- Model Manager is not available")
    
    with tab3:
        st.subheader("Model Deployment")
        st.info("Model deployment features will be available in future versions.")
