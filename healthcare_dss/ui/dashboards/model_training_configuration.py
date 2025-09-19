"""
Model Training Configuration Module
===================================

This module handles model configuration and parameter management:
- Model selection and compatibility
- Hyperparameter configuration
- Advanced training settings
- Model-specific parameter management
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from healthcare_dss.ui.utils.common import (
    display_error_message,
    display_success_message,
    display_warning_message
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_model_configuration_tab():
    """Show model configuration tab"""
    st.subheader("Model Configuration")
    
    # Check if data is selected
    if 'selected_data' not in st.session_state or st.session_state.selected_data is None:
        st.warning("Please select data first in the Data Selection tab.")
        return None
    
    selected_data = st.session_state.selected_data
    
    # Task type selection (allow manual override)
    st.subheader("Task Type")
    
    # Get recommended task type from intelligent analysis
    recommended_task_type = st.session_state.get('recommended_task_type', 'classification')
    task_type_confidence = st.session_state.get('task_type_confidence', 'medium')
    
    # If we have a task type recommendation from data selection, use it
    if 'task_type_recommendation' in st.session_state:
        recommendation = st.session_state.task_type_recommendation
        recommended_task_type = recommendation['primary']
        confidence_score = recommendation['confidence']
        task_type_confidence = 'high' if confidence_score > 80 else 'medium'
        
        # Update session state for consistency
        st.session_state.recommended_task_type = recommended_task_type
        st.session_state.task_type_confidence = task_type_confidence
    
    if recommended_task_type:
        if task_type_confidence == 'high':
            st.success(f"**Recommended:** {recommended_task_type.title()} (High confidence)")
        else:
            st.warning(f"**Recommended:** {recommended_task_type.title()} (Medium confidence)")
    
    # Allow manual override with correct default
    task_type_options = ["classification", "regression"]
    default_index = 0 if recommended_task_type == "classification" else 1
    
    task_type = st.selectbox(
        "Select Task Type:",
        options=task_type_options,
        index=default_index,
        help="Override the automatically detected task type if needed"
    )
    
    # Show if user is overriding the recommendation
    if task_type != recommended_task_type:
        st.warning(f"You are overriding the recommendation ({recommended_task_type}) with {task_type}")
        
        # Check if binning is needed for classification on continuous data
        if task_type == 'classification' and recommended_task_type == 'regression':
            _show_binning_options(selected_data['dataframe'], selected_data['target_column'])
    else:
        st.success(f"Using recommended task type: {task_type}")
    
    st.info(f"**Selected Task Type:** {task_type.title()}")
    
    # Model selection
    st.subheader("Model Selection")
    
    # Get compatible models for the task
    compatible_models = _get_compatible_models_for_task(task_type)
    
    if not compatible_models:
        st.error(f"No compatible models found for task type: {task_type}")
        return None
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type:",
        options=compatible_models,
        help=f"Models compatible with {task_type} task"
    )
    
    if not model_type:
        st.warning("Please select a model type.")
        return None
    
    st.success(f"Selected model: **{model_type}**")
    
    # Clear irrelevant parameters when model changes
    if st.session_state.get('last_model_type') != model_type:
        _clear_irrelevant_model_parameters(model_type)
        st.session_state.last_model_type = model_type
    
    # Model-specific configuration
    model_config = _show_model_specific_configuration(model_type, task_type)
    
    if model_config is None:
        return None
    
    # Store configuration
    st.session_state.model_config = {
        'model_type': model_type,
        'task_type': task_type,
        'parameters': model_config
    }
    
    return {
        'model_type': model_type,
        'task_type': task_type,
        'parameters': model_config
    }


def _get_compatible_models_for_task(task_type: str) -> List[str]:
    """Get compatible models for the given task type using dynamic configuration"""
    
    # Import the dynamic configuration function
    from healthcare_dss.ui.dashboards.model_training_model_creation import get_model_configurations
    
    configurations = get_model_configurations()
    compatible_models = []
    
    for model_name, model_config in configurations.items():
        if task_type in model_config:
            compatible_models.append(model_name)
    
    return compatible_models


def _clear_irrelevant_model_parameters(model_type: str):
    """Clear parameters that are not relevant to the selected model"""
    # Define parameter groups for each model
    model_parameters = {
        "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state"],
        "LightGBM": ["n_estimators", "max_depth", "learning_rate", "num_leaves", "subsample", "colsample_bytree", "random_state"],
        "Logistic Regression": ["C", "max_iter", "random_state"],
        "Linear Regression": ["fit_intercept", "normalize"],
        "Ridge Regression": ["alpha", "fit_intercept", "normalize"],
        "Lasso Regression": ["alpha", "fit_intercept", "normalize"],
        "Elastic Net": ["alpha", "l1_ratio", "fit_intercept", "normalize"],
        "SVM": ["C", "kernel", "gamma", "random_state"],
        "SVR": ["C", "kernel", "gamma", "epsilon"],
        "Decision Tree": ["max_depth", "min_samples_split", "min_samples_leaf", "random_state"],
        "K-Nearest Neighbors": ["n_neighbors", "weights", "algorithm"],
        "Naive Bayes": ["var_smoothing"]
    }
    
    # Clear parameters not relevant to current model
    if model_type in model_parameters:
        relevant_params = model_parameters[model_type]
        for param in list(st.session_state.keys()):
            if param.startswith('param_') and param[6:] not in relevant_params:
                if param in st.session_state:
                    del st.session_state[param]


def _show_model_specific_configuration(model_type: str, task_type: str):
    """Show model-specific configuration options using dynamic configuration"""
    st.subheader(f"{model_type} Configuration")
    
    # Import the dynamic configuration function
    from healthcare_dss.ui.dashboards.model_training_model_creation import get_model_configurations
    
    configurations = get_model_configurations()
    
    if model_type not in configurations or task_type not in configurations[model_type]:
        st.error(f"Model {model_type} not supported for {task_type} task")
        return {}
    
    config = configurations[model_type][task_type]
    default_params = config.get("default_params", {})
    param_ranges = config.get("param_ranges", {})
    
    parameters = {}
    
    # Configure parameters dynamically
    for param_name, default_value in default_params.items():
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            
            if isinstance(default_value, int):
                parameters[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_val, max_val, default_value,
                    help=f"Configure {param_name}",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
            elif isinstance(default_value, float):
                parameters[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    float(min_val), float(max_val), default_value,
                    help=f"Configure {param_name}",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
        else:
            # For parameters without ranges, use appropriate input type
            if param_name == "random_state":
                parameters[param_name] = st.number_input(
                    "Random State",
                    value=default_value,
                    help="Random seed for reproducibility",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
            elif param_name == "kernel":
                parameters[param_name] = st.selectbox(
                    "Kernel",
                    options=["rbf", "linear", "poly", "sigmoid"],
                    index=0,
                    help="Kernel type for SVM",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
            elif param_name == "weights":
                parameters[param_name] = st.selectbox(
                    "Weights",
                    options=["uniform", "distance"],
                    index=0,
                    help="Weight function for KNN",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
            else:
                # Default to number input for other parameters
                parameters[param_name] = st.number_input(
                    param_name.replace('_', ' ').title(),
                    value=default_value,
                    help=f"Configure {param_name}",
                    key=f"{model_type.lower().replace(' ', '_')}_{param_name}"
                )
    
    return parameters


def _show_binning_options(df: pd.DataFrame, target_column: str):
    """Show intelligent binning options when classification is used on continuous data"""
    st.subheader("Binning Configuration")
    
    st.info("""
    **Classification on Continuous Data**
    
    You're using classification on continuous data. The system will convert your continuous 
    target into discrete classes using intelligent binning.
    """)
    
    try:
        from healthcare_dss.utils.intelligent_binning import intelligent_binning
        
        target_data = df[target_column].values
        
        # Detect binning need and get suggestions
        needs_binning, analysis = intelligent_binning.detect_binning_need(target_data, 'classification')
        suggestions = intelligent_binning.suggest_optimal_bins(target_data, analysis)
        
        # Show data analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Analysis**")
            st.write(f"Unique values: {analysis['unique_values']}")
            st.write(f"Unique ratio: {analysis['unique_ratio']:.1%}")
            st.write(f"Range: {analysis['min_value']:.2f} to {analysis['max_value']:.2f}")
            st.write(f"Standard deviation: {analysis['std']:.2f}")
        
        with col2:
            st.write("**Recommended Settings**")
            st.write(f"Strategy: {suggestions['recommended_strategy']}")
            st.write(f"Number of bins: {suggestions['optimal_bins']}")
            st.write(f"Range: {suggestions['min_bins']}-{suggestions['max_bins']} bins")
        
        # Binning mode selection
        st.subheader("Binning Mode")
        
        binning_mode = st.radio(
            "Choose binning approach:",
            options=["Automatic", "Manual Configuration", "Advanced Override"],
            help="Select how you want to configure the binning"
        )
        
        if binning_mode == "Automatic":
            _show_automatic_binning(target_data, suggestions)
            
        elif binning_mode == "Manual Configuration":
            _show_manual_binning_configuration(target_data, suggestions)
            
        elif binning_mode == "Advanced Override":
            _show_advanced_binning_override(target_data, analysis)
        
    except ImportError as e:
        st.error(f"Binning system not available: {e}")
        st.info("Please use regression instead of classification for continuous data.")
    except Exception as e:
        st.error(f"Error in binning analysis: {e}")
        debug_manager.log_debug(f"Binning analysis error: {str(e)}", "ERROR")


def _show_automatic_binning(target_data: np.ndarray, suggestions: Dict[str, Any]):
    """Show automatic binning configuration"""
    st.write("**Automatic Binning**")
    st.write("The system will use optimal settings based on your data characteristics.")
    
    # Show recommended settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Strategy:** {suggestions['recommended_strategy']}")
        st.write(f"**Bins:** {suggestions['optimal_bins']}")
    
    with col2:
        if suggestions['reasoning']:
            st.write("**Reasoning:**")
            for reason in suggestions['reasoning'][:3]:  # Show first 3 reasons
                st.write(f"• {reason}")
    
    # Preview button
    if st.button("Preview Automatic Binning", help="See how automatic binning will affect your data"):
        from healthcare_dss.utils.intelligent_binning import intelligent_binning
        
        strategy = suggestions['recommended_strategy']
        n_bins = suggestions['optimal_bins']
        
        preview = intelligent_binning.get_binning_preview(target_data, strategy, n_bins)
        
        if preview['success']:
            st.success("Binning Preview:")
            
            # Show bin distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bin Distribution:**")
                for i, (label, count) in enumerate(zip(preview['bin_labels'], preview['bin_counts'])):
                    st.write(f"{label}: {count} samples ({count/len(target_data):.1%})")
            
            with col2:
                st.write("**Bin Ranges:**")
                edges = preview['bin_edges']
                for i in range(len(edges) - 1):
                    st.write(f"{preview['bin_labels'][i]}: {edges[i]:.2f} to {edges[i+1]:.2f}")
            
            # Show class balance
            balance = preview['class_balance']
            if balance > 0.5:
                st.success(f"Good class balance: {balance:.2f}")
            elif balance > 0.2:
                st.warning(f"Moderate class imbalance: {balance:.2f}")
            else:
                st.error(f"Poor class balance: {balance:.2f}")
            
            # Store configuration
            st.session_state.binning_config = {
                'mode': 'automatic',
                'strategy': strategy,
                'n_bins': n_bins,
                'preview': preview,
                'enabled': True
            }
            
        else:
            st.error(f"Binning preview failed: {preview['error']}")


def _show_manual_binning_configuration(target_data: np.ndarray, suggestions: Dict[str, Any]):
    """Show manual binning configuration options"""
    st.write("**Manual Configuration**")
    st.write("Configure binning settings manually.")
    
    # Strategy selection
    strategy_options = {
        'quantile': 'Quantile-based (equal frequency)',
        'uniform': 'Uniform (equal width)', 
        'kmeans': 'K-means clustering',
        'jenks': 'Jenks natural breaks'
    }
    
    default_strategy = suggestions['recommended_strategy']
    strategy = st.selectbox(
        "Binning Strategy:",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=list(strategy_options.keys()).index(default_strategy),
        help="Choose how to divide the continuous values into classes"
    )
    
    # Number of bins
    n_bins = st.slider(
        "Number of Bins:",
        min_value=suggestions['min_bins'],
        max_value=suggestions['max_bins'],
        value=suggestions['optimal_bins'],
        help="Number of discrete classes to create"
    )
    
    # Preview button
    if st.button("Preview Manual Configuration", help="See how your manual configuration will affect the data"):
        from healthcare_dss.utils.intelligent_binning import intelligent_binning
        
        preview = intelligent_binning.get_binning_preview(target_data, strategy, n_bins)
        
        if preview['success']:
            st.success("Manual Configuration Preview:")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bin Distribution:**")
                for i, (label, count) in enumerate(zip(preview['bin_labels'], preview['bin_counts'])):
                    st.write(f"{label}: {count} samples ({count/len(target_data):.1%})")
            
            with col2:
                st.write("**Bin Ranges:**")
                edges = preview['bin_edges']
                for i in range(len(edges) - 1):
                    st.write(f"{preview['bin_labels'][i]}: {edges[i]:.2f} to {edges[i+1]:.2f}")
            
            # Show class balance
            balance = preview['class_balance']
            if balance > 0.5:
                st.success(f"Good class balance: {balance:.2f}")
            elif balance > 0.2:
                st.warning(f"Moderate class imbalance: {balance:.2f}")
            else:
                st.error(f"Poor class balance: {balance:.2f}")
            
            # Store configuration
            st.session_state.binning_config = {
                'mode': 'manual',
                'strategy': strategy,
                'n_bins': n_bins,
                'preview': preview,
                'enabled': True
            }
            
        else:
            st.error(f"Configuration preview failed: {preview['error']}")


def _show_advanced_binning_override(target_data: np.ndarray, analysis: Dict[str, Any]):
    """Show advanced binning override options"""
    st.write("**Advanced Override**")
    st.write("Advanced users can override binning behavior completely.")
    
    override_options = st.multiselect(
        "Override Options:",
        options=["Custom Thresholds", "Custom Labels", "Force Optimal Binning", "Disable Binning"],
        help="Select override options"
    )
    
    override_config = {}
    
    if "Custom Thresholds" in override_options:
        st.write("**Custom Thresholds**")
        
        # Show suggested thresholds
        min_val = analysis['min_value']
        max_val = analysis['max_value']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Suggested Thresholds:**")
            st.write(f"Quartiles: {np.percentile(target_data, [25, 50, 75]).tolist()}")
            st.write(f"Equal width: {np.linspace(min_val, max_val, 4)[1:-1].tolist()}")
        
        with col2:
            st.write("**Manual Thresholds:**")
            threshold_input = st.text_input(
                "Enter thresholds (comma-separated):",
                placeholder="e.g., 50, 100, 150",
                help="Enter custom threshold values"
            )
            
            if threshold_input:
                try:
                    thresholds = [float(x.strip()) for x in threshold_input.split(',')]
                    override_config['custom_thresholds'] = thresholds
                    st.success(f"Custom thresholds: {thresholds}")
                except ValueError:
                    st.error("Invalid threshold format. Use comma-separated numbers.")
    
    if "Custom Labels" in override_options:
        st.write("**Custom Labels**")
        
        label_options = {
            'severity': ['Low', 'Medium', 'High'],
            'risk': ['Low Risk', 'Moderate Risk', 'High Risk'],
            'performance': ['Poor', 'Average', 'Good', 'Excellent'],
            'manual': []
        }
        
        label_type = st.selectbox(
            "Label Type:",
            options=list(label_options.keys()),
            help="Choose predefined labels or enter custom ones"
        )
        
        if label_type == 'manual':
            custom_labels = st.text_input(
                "Enter custom labels (comma-separated):",
                placeholder="e.g., Class A, Class B, Class C",
                help="Enter custom class labels"
            )
            if custom_labels:
                labels = [x.strip() for x in custom_labels.split(',')]
                override_config['custom_bin_labels'] = labels
                st.success(f"Custom labels: {labels}")
        else:
            override_config['custom_bin_labels'] = label_options[label_type]
            st.success(f"Using {label_type} labels: {label_options[label_type]}")
    
    if "Force Optimal Binning" in override_options:
        st.write("**Force Optimal Binning**")
        st.write("Force the system to use optimal binning settings regardless of data characteristics.")
        override_config['force_binning'] = True
    
    if "Disable Binning" in override_options:
        st.write("**Disable Binning**")
        st.warning("This will disable binning completely. Classification may fail with continuous data.")
        override_config['disable_binning'] = True
    
    # Apply override
    if override_options and st.button("Apply Advanced Override"):
        from healthcare_dss.utils.intelligent_binning import intelligent_binning
        
        try:
            y_binned, binning_info = intelligent_binning.apply_user_override_binning(target_data, override_config)
            
            if binning_info.get('warning'):
                st.warning(binning_info['warning'])
            
            st.success("Advanced Override Applied:")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Override Results:**")
                st.write(f"Strategy: {binning_info['strategy']}")
                st.write(f"Bins: {binning_info['binned_unique']}")
                st.write(f"Bin counts: {binning_info['bin_counts']}")
            
            with col2:
                if binning_info.get('bin_labels'):
                    st.write("**Bin Labels:**")
                    for i, label in enumerate(binning_info['bin_labels']):
                        st.write(f"{label}: {binning_info['bin_counts'][i]} samples")
            
            # Store configuration
            st.session_state.binning_config = {
                'mode': 'advanced_override',
                'override_config': override_config,
                'binning_info': binning_info,
                'enabled': True
            }
            
        except Exception as e:
            st.error(f"Advanced override failed: {str(e)}")


def _configure_random_forest():
    """Configure Random Forest parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 500, 100, help="Number of trees in the forest", key="rf_n_estimators")
        max_depth = st.slider("Max Depth", 1, 20, 10, help="Maximum depth of the tree", key="rf_max_depth")
    
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, help="Minimum samples required to split", key="rf_min_samples_split")
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, help="Minimum samples in leaf nodes", key="rf_min_samples_leaf")
    
    random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="rf_random_state")
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state
    }


def _configure_xgboost():
    """Configure XGBoost parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 1000, 100, help="Number of boosting rounds", key="xgb_n_estimators")
        max_depth = st.slider("Max Depth", 1, 10, 6, help="Maximum depth of trees", key="xgb_max_depth")
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, help="Step size shrinkage", key="xgb_learning_rate")
    
    with col2:
        subsample = st.slider("Subsample", 0.5, 1.0, 1.0, help="Fraction of samples for each tree", key="xgb_subsample")
        colsample_bytree = st.slider("Column Sample", 0.5, 1.0, 1.0, help="Fraction of features for each tree", key="xgb_colsample_bytree")
    
    random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="xgb_random_state")
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': random_state
    }


def _configure_lightgbm():
    """Configure LightGBM parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 1000, 100, help="Number of boosting rounds", key="lgb_n_estimators")
        max_depth = st.slider("Max Depth", 1, 10, 6, help="Maximum depth of trees", key="lgb_max_depth")
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, help="Step size shrinkage", key="lgb_learning_rate")
    
    with col2:
        num_leaves = st.slider("Number of Leaves", 10, 100, 31, help="Maximum number of leaves", key="lgb_num_leaves")
        subsample = st.slider("Subsample", 0.5, 1.0, 1.0, help="Fraction of samples for each tree", key="lgb_subsample")
        colsample_bytree = st.slider("Column Sample", 0.5, 1.0, 1.0, help="Fraction of features for each tree", key="lgb_colsample_bytree")
    
    random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="lgb_random_state")
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': random_state
    }


def _configure_logistic_regression():
    """Configure Logistic Regression parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, help="Inverse of regularization strength", key="lr_C")
        max_iter = st.slider("Max Iterations", 100, 1000, 100, help="Maximum number of iterations", key="lr_max_iter")
    
    with col2:
        random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="lr_random_state")
    
    return {
        'C': C,
        'max_iter': max_iter,
        'random_state': random_state
    }


def _configure_linear_regression():
    """Configure Linear Regression parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate intercept", key="linear_fit_intercept")
    
    with col2:
        normalize = st.checkbox("Normalize", value=False, help="Whether to normalize features", key="linear_normalize")
    
    return {
        'fit_intercept': fit_intercept,
        'normalize': normalize
    }


def _configure_ridge_regression():
    """Configure Ridge Regression parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, help="Regularization strength", key="ridge_alpha")
        fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate intercept", key="ridge_fit_intercept")
    
    with col2:
        normalize = st.checkbox("Normalize", value=False, help="Whether to normalize features", key="ridge_normalize")
    
    return {
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'normalize': normalize
    }


def _configure_lasso_regression():
    """Configure Lasso Regression parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, help="Regularization strength", key="lasso_alpha")
        fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate intercept", key="lasso_fit_intercept")
    
    with col2:
        normalize = st.checkbox("Normalize", value=False, help="Whether to normalize features", key="lasso_normalize")
    
    return {
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'normalize': normalize
    }


def _configure_elastic_net():
    """Configure Elastic Net parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, help="Regularization strength", key="elastic_alpha")
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, help="Balance between L1 and L2 regularization", key="elastic_l1_ratio")
    
    with col2:
        fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate intercept", key="elastic_fit_intercept")
        normalize = st.checkbox("Normalize", value=False, help="Whether to normalize features", key="elastic_normalize")
    
    return {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'fit_intercept': fit_intercept,
        'normalize': normalize
    }


def _configure_svm():
    """Configure SVM parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, help="Regularization parameter", key="svm_C")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], help="Kernel type", key="svm_kernel")
    
    with col2:
        gamma = st.selectbox("Gamma", ["scale", "auto", "0.001", "0.01", "0.1", "1"], help="Kernel coefficient", key="svm_gamma")
        random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="svm_random_state")
    
    return {
        'C': C,
        'kernel': kernel,
        'gamma': gamma,
        'random_state': random_state
    }


def _configure_svr():
    """Configure SVR parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, help="Regularization parameter", key="svr_C")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], help="Kernel type", key="svr_kernel")
    
    with col2:
        gamma = st.selectbox("Gamma", ["scale", "auto", "0.001", "0.01", "0.1", "1"], help="Kernel coefficient", key="svr_gamma")
        epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, help="Epsilon-tube", key="svr_epsilon")
    
    return {
        'C': C,
        'kernel': kernel,
        'gamma': gamma,
        'epsilon': epsilon
    }


def _configure_decision_tree():
    """Configure Decision Tree parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        max_depth = st.slider("Max Depth", 1, 20, 10, help="Maximum depth of the tree", key="dt_max_depth")
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, help="Minimum samples required to split", key="dt_min_samples_split")
    
    with col2:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, help="Minimum samples in leaf nodes", key="dt_min_samples_leaf")
        random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="dt_random_state")
    
    return {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state
    }


def _configure_knn():
    """Configure K-Nearest Neighbors parameters"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, help="Number of neighbors to use", key="knn_n_neighbors")
        weights = st.selectbox("Weights", ["uniform", "distance"], help="Weight function", key="knn_weights")
    
    with col2:
        algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], help="Algorithm to use", key="knn_algorithm")
    
    return {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'algorithm': algorithm
    }


def _configure_naive_bayes():
    """Configure Naive Bayes parameters"""
    var_smoothing = st.slider("Variance Smoothing", 1e-9, 1e-3, 1e-9, format="%.0e", help="Smoothing parameter", key="nb_var_smoothing")
    
    return {
        'var_smoothing': var_smoothing
    }


def show_advanced_settings_tab():
    """Show advanced training settings"""
    st.subheader("Advanced Settings")
    
    # Cross-validation settings
    st.subheader("Cross-Validation")
    cv_folds = st.slider("CV Folds", 3, 10, 5, help="Number of cross-validation folds")
    
    # Test size
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, help="Proportion of data for testing")
    
    # Random state
    random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility", key="advanced_random_state")
    
    # Data leakage prevention
    st.subheader("Data Leakage Prevention")
    
    st.markdown("""
    **Data leakage** occurs when information from the test set inadvertently influences the training process, 
    leading to overly optimistic performance estimates that don't generalize to new data. This is a critical 
    issue in machine learning that can result in models that perform poorly in production.
    """)
    
    prevent_data_leakage = st.checkbox(
        "Enable Data Leakage Prevention", 
        value=True, 
        help="Apply proper preprocessing pipeline to prevent data leakage"
    )
    
    if prevent_data_leakage:
        st.info("""
        **Data leakage prevention is enabled.** The preprocessing pipeline will be applied correctly:
        - Train-test split occurs BEFORE any preprocessing
        - All statistics (mean, median, mode) are calculated only on training data
        - Categorical encoding is fitted only on training data
        - Outlier detection uses training data statistics
        """)
        
        # Missing value strategy
        st.markdown("#### Missing Value Handling")
        st.markdown("""
        Missing values are imputed using statistics calculated **only from the training set** to prevent 
        information leakage from the test set.
        """)
        missing_value_strategy = st.selectbox(
            "Missing Value Strategy",
            options=["mean", "median", "mode", "drop"],
            index=0,
            help="How to handle missing values in features"
        )
        
        # Categorical encoding strategy
        st.markdown("#### Categorical Variable Encoding")
        st.markdown("""
        Categorical variables are encoded using information **only from the training set**. 
        Unseen categories in the test set are handled appropriately.
        """)
        categorical_strategy = st.selectbox(
            "Categorical Encoding Strategy", 
            options=["one_hot", "label", "target"],
            index=0,
            help="How to encode categorical variables"
        )
        
        # Outlier handling
        st.markdown("#### Outlier Detection and Handling")
        st.markdown("""
        Outlier detection thresholds are calculated **only from the training set** to ensure 
        consistent treatment of outliers in both training and test sets.
        """)
        handle_outliers = st.checkbox("Handle Outliers", value=False, help="Detect and handle outliers")
        
        if handle_outliers:
            outlier_method = st.selectbox(
                "Outlier Detection Method",
                options=["iqr", "zscore", "isolation_forest"],
                index=0,
                help="Method to detect outliers"
            )
        else:
            outlier_method = "iqr"  # Default value when not handling outliers
    else:
        st.warning("""
        **WARNING: Data leakage prevention is disabled.** 
        
        This will apply preprocessing to the entire dataset before splitting, which can lead to:
        - Overly optimistic performance estimates
        - Poor generalization to new data
        - Models that perform worse in production
        
        Only disable this for educational purposes or when comparing with legacy implementations.
        """)
        missing_value_strategy = "mean"
        categorical_strategy = "one_hot"
        handle_outliers = False
        outlier_method = "iqr"
    
    # Feature scaling
    st.subheader("Feature Scaling")
    scale_features = st.checkbox("Scale Features", value=True, help="Apply feature scaling")
    
    # Feature selection
    st.subheader("Feature Selection")
    feature_selection = st.checkbox("Enable Feature Selection", value=False, help="Apply feature selection")
    
    if feature_selection:
        max_features = st.slider("Max Features", 5, 50, 20, help="Maximum number of features to select")
    else:
        max_features = None
    
    # Early stopping (for tree-based models)
    st.subheader("Early Stopping")
    early_stopping = st.checkbox("Enable Early Stopping", value=False, help="Stop training early if no improvement")
    
    if early_stopping:
        patience = st.slider("Patience", 5, 50, 10, help="Number of rounds to wait before stopping")
    else:
        patience = None
    
    # Store advanced settings
    advanced_settings = {
        'cv_folds': cv_folds,
        'test_size': test_size,
        'random_state': random_state,
        'prevent_data_leakage': prevent_data_leakage,
        'missing_value_strategy': missing_value_strategy,
        'categorical_strategy': categorical_strategy,
        'handle_outliers': handle_outliers,
        'outlier_method': outlier_method,
        'scale_features': scale_features,
        'feature_selection': feature_selection,
        'max_features': max_features,
        'early_stopping': early_stopping,
        'patience': patience
    }
    
    st.session_state.advanced_settings = advanced_settings
    
    return advanced_settings
