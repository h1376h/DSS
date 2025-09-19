"""
Model Training Data Selection Module
====================================

This module handles data selection and analysis for model training:
- Dataset selection and validation
- Target column analysis with smart recommendations
- Intelligent task type detection
- Advanced preprocessing options
- Smart model recommendations
- Data quality analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message,
    display_success_message,
    display_warning_message,
    safe_dataframe_display,
    safe_dataset_selection,
    get_dataset_from_managers
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_data_selection_tab():
    """Show comprehensive data selection and analysis tab with smart features"""
    st.subheader("Data Selection & Analysis")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=st.session_state.get('debug_mode', False), key="debug_mode_training")
    st.session_state.debug_mode = debug_mode
    
    # Dataset selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_name = safe_dataset_selection(key="model_training_dataset", help_text="Select dataset for model training")
    
    with col2:
        if dataset_name:
            df = get_dataset_from_managers(dataset_name)
            if df is not None:
                st.metric("Records", f"{len(df):,}")
                st.metric("Features", len(df.columns))
    
    if not dataset_name:
        st.warning("Please select a dataset to continue.")
        return None
    
    # Get dataset
    df = get_dataset_from_managers(dataset_name)
    if df is None:
        st.error(f"Could not load dataset: {dataset_name}")
        return None
    
    st.success(f"Loaded dataset: **{dataset_name}**")
    
    # Store selected dataset in session state
    st.session_state.selected_dataset = dataset_name
    
    # Comprehensive Dataset Analysis and Recommendations
    st.subheader("Dataset Analysis & Smart Recommendations")
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Target Selection", "Dataset Overview", "Feature Analysis", "Smart Recommendations"])
        
    with tab1:
        st.subheader("Target Column Selection")
        
        # Import smart target manager
        try:
            from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
            smart_manager = SmartDatasetTargetManager()
            
            # Get smart target recommendations
            smart_targets = smart_manager.get_dataset_targets(dataset_name)
            smart_functionalities = smart_manager.get_smart_functionalities(dataset_name)
            
            # Create enhanced target selection
            if smart_targets:
                st.info("Smart Target Recommendations Available!")
                
                # Show smart targets with descriptions
                st.write("**Recommended Target Variables:**")
                for i, target in enumerate(smart_targets):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        # Check if this target is currently selected
                        is_selected = st.session_state.get('selected_target') == target['column']
                        if is_selected:
                            st.write(f"**{target['column']}** (Selected)")
                            st.caption("Currently Selected")
                        else:
                            st.write(f"**{target['column']}**")
                        st.caption(f"Type: {target['target_type']}")
                        st.caption(target.get('business_meaning', 'Target variable'))
                    with col2:
                        # Convert values to int for display
                        unique_values = int(target['unique_values']) if isinstance(target['unique_values'], str) else target['unique_values']
                        st.write(f"Unique: {unique_values}")
                    with col3:
                        # Convert missing values to int for display
                        missing_values = int(target['missing_values']) if isinstance(target['missing_values'], str) else target['missing_values']
                        if missing_values > 0:
                            st.warning(f"Missing: {missing_values}")
                        else:
                            st.success("No missing")
                    with col4:
                        # Change button text based on selection status
                        button_text = "Selected" if is_selected else "Select"
                        button_type = "secondary" if is_selected else "primary"
                        
                        if st.button(button_text, key=f"select_target_{i}", type=button_type):
                            st.session_state.selected_target = target['column']
                            st.success(f"Selected '{target['column']}' as target variable")
                            st.rerun()
                
                st.markdown("---")
            
            # Show smart functionalities
            if smart_functionalities:
                st.write("**Available Smart Features:**")
                cols = st.columns(2)
                for i, feature in enumerate(smart_functionalities):
                    with cols[i % 2]:
                        st.write(f"• {feature}")
                st.markdown("---")
                
        except ImportError:
            st.warning("Smart target manager not available. Using basic target selection.")
        except Exception as e:
            st.warning(f"Could not load smart recommendations: {e}")
        
        # Standard target selection
        # Get the index of the currently selected target
        selected_target = st.session_state.get('selected_target', None)
        target_index = 0
        if selected_target and selected_target in df.columns.tolist():
            target_index = df.columns.tolist().index(selected_target)
        
        target_column = st.selectbox(
            "Select Target Column",
            df.columns.tolist(),
            index=target_index,
            help="Choose the column you want to predict"
        )
    
    with tab2:
        st.subheader("Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Total Features", len(df.columns))
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric_cols)
        
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Features", categorical_cols)
        
        # Data quality metrics
        st.subheader("Data Quality")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Completeness", f"{completeness:.1f}%")
        
        with col2:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        with col3:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)
    
    with tab3:
        st.subheader("Feature Analysis")
        
        # Show all features with their characteristics
        st.write("**All Features in Dataset:**")
        
        # Create a comprehensive feature analysis
        feature_analysis = []
        for col in df.columns:
            col_data = df[col]
            analysis = {
                'feature': col,
                'type': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'missing_values': col_data.isnull().sum(),
                'missing_ratio': (col_data.isnull().sum() / len(col_data)) * 100,
                'is_numeric': pd.api.types.is_numeric_dtype(col_data)
            }
            
            if analysis['is_numeric']:
                analysis['mean'] = col_data.mean()
                analysis['std'] = col_data.std()
                analysis['min'] = col_data.min()
                analysis['max'] = col_data.max()
            else:
                analysis['most_common'] = col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'
            
            feature_analysis.append(analysis)
        
        # Display feature analysis in a nice format
        for i, analysis in enumerate(feature_analysis):
            with st.expander(f"{analysis['feature']} ({analysis['type']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Unique Values:** {analysis['unique_values']}")
                    st.write(f"**Missing Values:** {analysis['missing_values']} ({analysis['missing_ratio']:.1f}%)")
                
                with col2:
                    if analysis['is_numeric']:
                        st.write(f"**Mean:** {analysis['mean']:.2f}")
                        st.write(f"**Std:** {analysis['std']:.2f}")
                    else:
                        st.write(f"**Most Common:** {analysis['most_common']}")
                
                with col3:
                    if analysis['is_numeric']:
                        st.write(f"**Min:** {analysis['min']:.2f}")
                        st.write(f"**Max:** {analysis['max']:.2f}")
                    else:
                        st.write(f"**Type:** Categorical")
    
    with tab4:
        st.subheader("Smart Recommendations")
        
        try:
            from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
            smart_manager = SmartDatasetTargetManager()
            
            # Get comprehensive smart insights
            smart_insights = smart_manager.get_smart_insights(dataset_name)
            
            if smart_insights:
                # Business Value
                if smart_insights.get('business_value'):
                    st.write("**Business Value:**")
                    for value in smart_insights['business_value']:
                        st.write(f"• {value}")
                
                # Recommended Use Cases
                if smart_insights.get('recommended_use_cases'):
                    st.write("**Recommended Use Cases:**")
                    for use_case in smart_insights['recommended_use_cases']:
                        st.write(f"• {use_case}")
                
                # Integration Suggestions
                if smart_insights.get('integration_suggestions'):
                    st.write("**Integration Suggestions:**")
                    for suggestion in smart_insights['integration_suggestions']:
                        st.write(f"• {suggestion}")
                
                # Smart Functionalities
                smart_functionalities = smart_manager.get_smart_functionalities(dataset_name)
                if smart_functionalities:
                    st.write("**Smart Functionalities Available:**")
                    cols = st.columns(2)
                    for i, functionality in enumerate(smart_functionalities):
                        with cols[i % 2]:
                            st.write(f"• {functionality}")
                
                # Model Recommendations
                recommended_models = smart_manager.get_recommended_models(dataset_name)
                if recommended_models:
                    st.write("**Recommended Models:**")
                    cols = st.columns(3)
                    for i, model in enumerate(recommended_models):
                        with cols[i % 3]:
                            st.write(f"• {model}")
            
            else:
                st.info("No smart insights available for this dataset.")
                
        except Exception as e:
            st.warning(f"Could not load smart recommendations: {e}")
    
    # Store selected target
    if target_column:
        # Check if target has changed to trigger re-analysis
        previous_target = st.session_state.get('selected_target', None)
        target_changed = previous_target != target_column
        
        st.session_state.selected_target = target_column
        
        # Debug: Show target column selection details
        if st.session_state.get('debug_mode', False):
            st.write("Debug - Target Column Selection:")
            st.write(f"Selected target column: '{target_column}'")
            st.write(f"Previous target column: '{previous_target}'")
            st.write(f"Target changed: {target_changed}")
            st.write(f"Target column data type: {df[target_column].dtype}")
            st.write(f"Target column unique values: {df[target_column].unique()[:5]}")
            
            # Check if there are other target-related columns
            target_related_cols = [col for col in df.columns if 'target' in col.lower()]
            st.write(f"All target-related columns: {target_related_cols}")
            for col in target_related_cols:
                st.write(f"  {col}: {df[col].dtype} - {df[col].unique()[:3]}")
        
        # Always analyze target column and update recommendation
        _analyze_target_column(df, target_column)
        
        # Task type recommendation - always update when target changes
        _show_task_type_recommendation(df, target_column)
        
        # Clear any existing model config when target changes
        if target_changed and 'model_config' in st.session_state:
            # Reset model config to force re-selection
            del st.session_state.model_config
            debug_manager.log_debug(f"Target changed from '{previous_target}' to '{target_column}', cleared model config")
        
        return {
            'dataset_name': dataset_name,
            'dataframe': df,
            'target_column': target_column
        }
    
    return None


def _detect_potential_targets(df: pd.DataFrame) -> List[str]:
    """Detect potential target columns based on naming patterns and data types"""
    potential_targets = []
    
    # Common target column names
    target_patterns = [
        'target', 'label', 'class', 'outcome', 'result', 'prediction',
        'diagnosis', 'disease', 'condition', 'status', 'type', 'category',
        'score', 'rating', 'grade', 'level', 'risk', 'severity'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for exact matches
        if col_lower in target_patterns:
            potential_targets.append(col)
            continue
        
        # Check for partial matches
        for pattern in target_patterns:
            if pattern in col_lower:
                potential_targets.append(col)
                break
        
        # Check for binary columns (good for classification)
        if df[col].nunique() == 2:
            potential_targets.append(col)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(potential_targets))


def _analyze_target_column(df: pd.DataFrame, target_column: str):
    """Analyze target column and provide comprehensive insights"""
    st.subheader("Target Column Analysis")
    
    target_data = df[target_column]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Type", str(target_data.dtype))
        st.metric("Unique Values", target_data.nunique())
    
    with col2:
        st.metric("Missing Values", target_data.isnull().sum())
        st.metric("Non-Null Values", target_data.count())
    
    with col3:
        if pd.api.types.is_numeric_dtype(target_data):
            st.metric("Min Value", f"{target_data.min():.2f}")
            st.metric("Max Value", f"{target_data.max():.2f}")
        else:
            st.metric("Most Frequent", target_data.mode().iloc[0] if len(target_data.mode()) > 0 else "N/A")
            st.metric("Frequency", target_data.value_counts().iloc[0] if len(target_data.value_counts()) > 0 else 0)
    
    # Visual analysis
    if pd.api.types.is_numeric_dtype(target_data):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(df, x=target_column, title=f'Distribution of {target_column}')
            st.plotly_chart(fig_hist, width="stretch")
        
        with col2:
            fig_box = px.box(df, y=target_column, title=f'Box Plot of {target_column}')
            st.plotly_chart(fig_box, width="stretch")
    else:
        # Categorical data visualization
        value_counts = target_data.value_counts().head(10)
        fig_bar = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=f'Top 10 Values in {target_column}'
        )
        st.plotly_chart(fig_bar, width="stretch")
    
    # Missing Value Analysis
    missing_count = target_data.isnull().sum()
    if missing_count > 0:
        st.subheader("Missing Value Analysis")
        
        try:
            from healthcare_dss.utils.intelligent_data_analyzer import IntelligentDataAnalyzer
            
            # Get missing value strategies
            missing_strategies = IntelligentDataAnalyzer.get_missing_value_strategies(target_data)
            
            if missing_strategies['has_missing']:
                st.warning(f"Found {missing_strategies['missing_count']} missing values ({missing_strategies['missing_ratio']:.1%})")
                
                # Show recommended strategies
                st.write("**Recommended Strategies:**")
                for strategy in missing_strategies['recommended_strategies']:
                    st.write(f"• {strategy}")
                
                # Show strategy details
                if missing_strategies['strategy_details']:
                    with st.expander("Strategy Details"):
                        for strategy_name, details in missing_strategies['strategy_details'].items():
                            st.write(f"**{strategy_name.title()}:**")
                            st.write(f"Description: {details['description']}")
                            st.write(f"Suitable for: {details['suitable_for']}")
                            st.write(f"Confidence: {details['confidence']}%")
                            st.code(details['implementation'], language='python')
                            st.markdown("---")
            
        except Exception as e:
            st.warning(f"Could not analyze missing values: {e}")
            # Basic missing value info
            missing_ratio = missing_count / len(target_data)
            st.write(f"Missing values: {missing_count} ({missing_ratio:.1%})")
            if missing_ratio > 0.1:
                st.warning("High missing data ratio. Consider data cleaning.")


def _show_task_type_recommendation(df: pd.DataFrame, target_column: str):
    """Show intelligent task type recommendation using smart analysis"""
    st.subheader("Intelligent Task Type Recommendation")
    
    target_data = df[target_column]
    
    # Use intelligent data analyzer
    try:
        from healthcare_dss.utils.intelligent_data_analyzer import IntelligentDataAnalyzer
        
        # Get dataset name from session state
        dataset_name = st.session_state.get('selected_dataset', None)
        
        # Perform intelligent analysis
        analysis = IntelligentDataAnalyzer.detect_task_type(target_data, dataset_name)
        
        # Display analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Characteristics")
            characteristics = analysis['data_characteristics']
            st.write(f"**Data Type:** {characteristics['data_type']}")
            st.write(f"**Unique Values:** {characteristics['unique_values']}")
            st.write(f"**Unique Ratio:** {characteristics['unique_ratio']:.1%}")
            st.write(f"**Missing Values:** {characteristics['missing_count']} ({characteristics['missing_ratio']:.1%})")
        
        with col2:
            st.subheader("Recommendation")
            target_type = analysis['target_type']
            confidence = analysis['confidence']
            
            if target_type == 'classification':
                st.success(f"**Recommended: Classification** (Confidence: {confidence}%)")
                st.info("This target represents distinct categories or classes")
            elif target_type == 'regression':
                st.success(f"**Recommended: Regression** (Confidence: {confidence}%)")
                st.info("This target contains continuous numerical values")
            else:
                st.warning(f"**Both Suitable** (Confidence: {confidence}%)")
                st.info("Choose based on your prediction goal")
            
            # Show reasons
            if analysis['reasons']:
                st.write("**Reasons:**")
                for reason in analysis['reasons']:
                    st.write(f"• {reason}")
        
        # Show smart insights if available
        if 'smart_insights' in analysis:
            st.subheader("Smart Insights")
            insights = analysis['smart_insights']
            if insights.get('business_meaning'):
                st.write(f"**Business Meaning:** {insights['business_meaning']}")
            if insights.get('smart_features'):
                st.write("**Smart Features:**")
                for feature in insights['smart_features'][:3]:  # Show first 3
                    st.write(f"• {feature}")
        
        # Store recommendation in session state
        st.session_state.task_type_recommendation = {
            'primary': target_type if target_type != 'mixed' else 'classification',
            'confidence': confidence,
            'analysis': analysis
        }
        
        # Also store the recommended task type for easy access
        st.session_state.recommended_task_type = target_type if target_type != 'mixed' else 'classification'
        st.session_state.task_type_confidence = 'high' if confidence > 80 else 'medium'
        
        # Update model config if it exists
        if 'model_config' in st.session_state:
            st.session_state.model_config['task_type'] = target_type if target_type != 'mixed' else 'classification'
        
    except ImportError:
        # Fallback to original logic if intelligent analyzer not available
        st.warning("Intelligent analyzer not available. Using basic analysis.")
        _show_basic_task_type_recommendation(df, target_column)
    except Exception as e:
        st.error(f"Error in intelligent analysis: {e}")
        _show_basic_task_type_recommendation(df, target_column)


def _show_basic_task_type_recommendation(df: pd.DataFrame, target_column: str):
    """Fallback basic task type recommendation"""
    target_data = df[target_column]
    
    # Analyze for classification vs regression
    is_numeric = pd.api.types.is_numeric_dtype(target_data)
    unique_values = target_data.nunique()
    total_values = len(target_data)
    unique_ratio = unique_values / total_values
    
    # Classification suitability
    classification_suitable = False
    classification_confidence = 0
    classification_reasons = []
    
    if not is_numeric:
        classification_suitable = True
        classification_confidence = 95
        classification_reasons.append("Non-numeric data type")
    elif unique_ratio < 0.1 and unique_values < 20:
        classification_suitable = True
        classification_confidence = 90
        classification_reasons.append(f"Low cardinality ({unique_values} unique values)")
        classification_reasons.append(f"Low unique ratio ({unique_ratio:.1%})")
    elif unique_values == 2:
        classification_suitable = True
        classification_confidence = 100
        classification_reasons.append("Binary classification (2 unique values)")
    
    # Regression suitability
    regression_suitable = False
    regression_confidence = 0
    regression_reasons = []
    
    if is_numeric and unique_ratio > 0.1:
        regression_suitable = True
        regression_confidence = 90
        regression_reasons.append("Numeric data type")
        regression_reasons.append(f"High cardinality ({unique_values} unique values)")
        regression_reasons.append(f"High unique ratio ({unique_ratio:.1%})")
    elif is_numeric and unique_values > 20:
        regression_suitable = True
        regression_confidence = 80
        regression_reasons.append("Numeric data type")
        regression_reasons.append(f"Many unique values ({unique_values})")
    
    # Display recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Suitability")
        if classification_suitable:
            st.success(f"Suitable (Confidence: {classification_confidence}%)")
            for reason in classification_reasons:
                st.write(f"• {reason}")
        else:
            st.error("Not Suitable")
            st.write("• High cardinality numeric data")
    
    with col2:
        st.subheader("Regression Suitability")
        if regression_suitable:
            st.success(f"Suitable (Confidence: {regression_confidence}%)")
            for reason in regression_reasons:
                st.write(f"• {reason}")
        else:
            st.error("Not Suitable")
            st.write("• Low cardinality or non-numeric data")
    
    # Primary recommendation
    if classification_confidence > regression_confidence:
        primary_recommendation = "classification"
        primary_confidence = classification_confidence
    elif regression_confidence > classification_confidence:
        primary_recommendation = "regression"
        primary_confidence = regression_confidence
    else:
        primary_recommendation = "both"
        primary_confidence = max(classification_confidence, regression_confidence)
    
    st.subheader("Primary Recommendation")
    
    if primary_recommendation == "classification":
        st.success(f"**Recommended: Classification** (Confidence: {primary_confidence}%)")
        st.info("This column appears to represent distinct categories or classes")
    elif primary_recommendation == "regression":
        st.success(f"**Recommended: Regression** (Confidence: {primary_confidence}%)")
        st.info("This column appears to contain continuous numerical values")
    else:
        st.warning(f"**Both are possible** (Confidence: {primary_confidence}%)")
        st.info("Choose based on your prediction goal - categories or continuous values")
    
    # Store recommendation in session state
    st.session_state.task_type_recommendation = {
        'primary': primary_recommendation,
        'confidence': primary_confidence,
        'classification_suitable': classification_suitable,
        'regression_suitable': regression_suitable
    }


def show_preprocessing_recommendations(df: pd.DataFrame, target_column: str, task_type: str):
    """Show intelligent preprocessing recommendations"""
    st.subheader("Preprocessing Recommendations")
    
    try:
        from healthcare_dss.utils.intelligent_data_analyzer import IntelligentDataAnalyzer
        
        target_data = df[target_column]
        dataset_name = st.session_state.get('selected_dataset', None)
        
        # Get preprocessing recommendations
        preprocessing_recs = IntelligentDataAnalyzer.get_preprocessing_recommendations(target_data, task_type, dataset_name)
        
        if preprocessing_recs['preprocessing_steps']:
            st.write("**Recommended Preprocessing Steps:**")
            for step in preprocessing_recs['preprocessing_steps']:
                st.write(f"• {step}")
            
            # Show details for each step
            if preprocessing_recs['details']:
                st.write("**Implementation Details:**")
                for step_name, details in preprocessing_recs['details'].items():
                    with st.expander(f"{step_name.replace('_', ' ').title()}"):
                        st.write(f"**Description:** {details['description']}")
                        if 'implementation' in details:
                            st.code(details['implementation'], language='python')
                        if 'imbalance_ratio' in details:
                            st.write(f"**Imbalance Ratio:** {details['imbalance_ratio']:.2f}")
                        if 'outlier_count' in details:
                            st.write(f"**Outlier Count:** {details['outlier_count']}")
        
        # Show smart preprocessing steps if available
        if 'smart_steps' in preprocessing_recs:
            st.write("**Smart Preprocessing Steps:**")
            for step in preprocessing_recs['smart_steps']:
                st.write(f"• {step}")
        
        # Store preprocessing config in session state
        st.session_state.preprocessing_config = {
            'steps': preprocessing_recs['preprocessing_steps'],
            'details': preprocessing_recs['details'],
            'smart_steps': preprocessing_recs.get('smart_steps', [])
        }
        
    except Exception as e:
        st.warning(f"Could not get preprocessing recommendations: {e}")
        st.session_state.preprocessing_config = {}


def show_smart_model_recommendations(dataset_name: str, target_column: str):
    """Show smart model recommendations"""
    st.subheader("Smart Model Recommendations")
    
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        
        # Get model recommendations
        model_recommendations = smart_manager.get_model_recommendations(dataset_name, target_column)
        recommended_models = model_recommendations.get('recommended_models', [])
        recommended_metrics = model_recommendations.get('metrics', [])
        recommended_visualizations = model_recommendations.get('visualizations', [])
        
        if recommended_models:
            st.info("Smart Model Recommendations Available!")
            
            # Show recommended models
            st.write("**Recommended Models:**")
            for model in recommended_models:
                st.write(f"• {model}")
            
            # Show recommended metrics
            if recommended_metrics:
                st.write("**Recommended Metrics:**")
                for metric in recommended_metrics:
                    st.write(f"• {metric}")
            
            # Show recommended visualizations
            if recommended_visualizations:
                st.write("**Recommended Visualizations:**")
                for viz in recommended_visualizations:
                    st.write(f"• {viz}")
            
            st.markdown("---")
            
    except Exception as e:
        st.warning(f"Could not load smart model recommendations: {e}")


def show_data_quality_analysis(df: pd.DataFrame):
    """Show comprehensive data quality analysis"""
    st.subheader("Data Quality Analysis")
    
    # Basic quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    with col2:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    with col3:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    # Detailed missing value analysis
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        st.subheader("Missing Value Analysis by Column")
        
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
        
        # Show missing data table
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df)) * 100
        })
        st.dataframe(missing_df, width="stretch")
    else:
        st.success("No missing values found in the dataset!")
    
    # Data type analysis
    st.subheader("Data Type Analysis")
    dtype_counts = df.dtypes.value_counts()
    
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Data Types Distribution"
    )
    st.plotly_chart(fig, width="stretch")
    
    # Outlier detection for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("Outlier Detection")
        
        outlier_summary = []
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Outlier Count': outlier_count,
                'Outlier %': outlier_percentage
            })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df, width="stretch")
