"""
Model Training Dashboard Module
===============================

This module provides comprehensive model training capabilities with:
- Interactive model training interface
- Advanced preprocessing configuration
- Hyperparameter optimization
- Real-time training progress
- Model comparison tools

This is the main coordinator module that integrates all training components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message,
    display_success_message,
    display_warning_message
)
from healthcare_dss.utils.debug_manager import debug_manager

# Import split components
from healthcare_dss.ui.dashboards.model_training_data_selection import show_data_selection_tab
from healthcare_dss.ui.dashboards.model_training_configuration import show_model_configuration_tab, show_advanced_settings_tab
from healthcare_dss.ui.dashboards.model_training_execution import train_model_with_progress
from healthcare_dss.ui.dashboards.model_training_results import show_training_results_tab


def show_model_training():
    """Show model training interface with comprehensive options"""
    st.header("Model Training")
    st.markdown("**Comprehensive machine learning model training with advanced configuration**")
    
    # Check system initialization
    if not check_system_initialization():
        st.error("System not initialized. Please refresh the page.")
        return
    
    # Debug mode is controlled from the navigation drawer
    debug_mode = st.session_state.get('debug_mode', False)
    
    if debug_mode:
        debug_manager.log_debug("Model training interface opened")
    
    # Create tabs for different training phases
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Selection", 
        "Model Configuration", 
        "Advanced Settings",
        "Training", 
        "Results"
    ])
    
    with tab1:
        # Data selection and analysis
        selected_data = show_data_selection_tab()
        if selected_data:
            st.session_state.selected_data = selected_data
            display_success_message("Data selected successfully!")
            
            if debug_mode:
                debug_manager.log_debug(f"Data selected: {selected_data['dataset_name']}, Target: {selected_data['target_column']}")
    
    with tab2:
        # Model configuration
        model_config = show_model_configuration_tab()
        if model_config:
            display_success_message("Model configured successfully!")
            
            if debug_mode:
                debug_manager.log_debug(f"Model configured: {model_config['model_type']}")
    
    with tab3:
        # Advanced settings
        advanced_settings = show_advanced_settings_tab()
        if advanced_settings:
            display_success_message("Advanced settings configured!")
            
            if debug_mode:
                debug_manager.log_debug("Advanced settings configured")
    
    with tab4:
        # Training execution
        _show_training_tab()
    
    with tab5:
        # Results display
        show_training_results_tab()

def _show_training_tab():
    """Show training execution tab"""
    st.subheader("Model Training")
    
    # Check prerequisites
    if 'selected_data' not in st.session_state or st.session_state.selected_data is None:
        st.warning("Please select data first in the Data Selection tab.")
        return
    
    if 'model_config' not in st.session_state or st.session_state.model_config is None:
        st.warning("Please configure model first in the Model Configuration tab.")
        return
    
    # Training summary
    _display_training_summary()
    
    # Training controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Start Training", type="primary", width="stretch"):
            _execute_training()
    
    with col2:
        if st.button("Reset", width="stretch"):
            _reset_training_state()
    
    with col3:
        if st.button("Save Config", width="stretch"):
            _save_training_config()
    
    # Training status
    if 'training_results' in st.session_state and st.session_state.training_results:
        # Quick metrics display
        results = st.session_state.training_results
        _display_quick_metrics(results)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("View Detailed Results", width="stretch"):
                st.session_state.active_tab = "Results"
                st.rerun()
        
        with col2:
            if st.button("Train Another Model", width="stretch"):
                _reset_training_state()
                st.rerun()
        
        with col3:
            if st.button("Export Results", width="stretch"):
                _export_training_results(results)


def _display_training_summary():
    """Display training configuration summary"""
    st.subheader("Training Summary")
    
    # Data summary
    if 'selected_data' in st.session_state and st.session_state.selected_data:
        selected_data = st.session_state.selected_data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset", selected_data['dataset_name'])
        
        with col2:
            st.metric("Target Column", selected_data['target_column'])
        
        with col3:
            df = selected_data['dataframe']
            st.metric("Data Shape", f"{df.shape[0]} × {df.shape[1]}")
    
    # Model summary
    if 'model_config' in st.session_state and st.session_state.model_config:
        model_config = st.session_state.model_config
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_config['model_type'])
        
        with col2:
            st.metric("Task Type", model_config['task_type'])
        
        with col3:
            param_count = len(model_config['parameters'])
            st.metric("Parameters", param_count)
    
    # Advanced settings summary
    if 'advanced_settings' in st.session_state and st.session_state.advanced_settings:
        advanced_settings = st.session_state.advanced_settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CV Folds", advanced_settings.get('cv_folds', 5))
        
        with col2:
            test_size = advanced_settings.get('test_size', 0.2)
            st.metric("Test Size", f"{test_size:.1%}")
        
        with col3:
            scale_features = advanced_settings.get('scale_features', True)
            st.metric("Feature Scaling", "Yes" if scale_features else "No")


def _execute_training():
    """Execute model training"""
    try:
        # Clear previous results
        if 'training_results' in st.session_state:
            del st.session_state.training_results
        
        # Start training
        with st.spinner("Training model..."):
            results = train_model_with_progress()
        
        if results:
            st.session_state.training_results = results
            display_success_message("Model training completed successfully!")
            
            # Log training completion
            if st.session_state.get('debug_mode', False):
                debug_manager.log_debug(f"Model training completed: {results['model_type']}")
        else:
            display_error_message("Model training failed!")
            
    except Exception as e:
        display_error_message(f"Training error: {str(e)}")
        debug_manager.log_debug(f"Training error: {str(e)}")


def _display_quick_metrics(results: Dict):
    """Display quick metrics after training"""
    st.subheader("Quick Metrics")
    
    metrics = results['metrics']
    task_type = results['task_type']
    
    # Use actual_task_type from metrics if available (more accurate)
    actual_task_type = metrics.get('actual_task_type', task_type)
    
    # Debug logging
    if st.session_state.get('debug_mode', False):
        st.write(f"DEBUG: Quick Metrics - Stored Task Type: {task_type}")
        st.write(f"DEBUG: Quick Metrics - Actual Task Type: {actual_task_type}")
        st.write(f"DEBUG: Quick Metrics - Available metrics: {list(metrics.keys())}")
    
    if actual_task_type == 'classification':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = metrics.get('accuracy', 0.0)
            st.metric("Accuracy", f"{accuracy:.3f}" if accuracy is not None else "N/A")
        
        with col2:
            precision = metrics.get('precision', 0.0)
            st.metric("Precision", f"{precision:.3f}" if precision is not None else "N/A")
        
        with col3:
            recall = metrics.get('recall', 0.0)
            st.metric("Recall", f"{recall:.3f}" if recall is not None else "N/A")
        
        with col4:
            f1_score = metrics.get('f1_score', 0.0)
            st.metric("F1 Score", f"{f1_score:.3f}" if f1_score is not None else "N/A")
    
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            r2_score = metrics.get('r2_score', 0.0)
            st.metric("R² Score", f"{r2_score:.3f}" if r2_score is not None else "N/A")
        
        with col2:
            mse = metrics.get('mse', 0.0)
            st.metric("MSE", f"{mse:.3f}" if mse is not None else "N/A")
        
        with col3:
            rmse = metrics.get('rmse', 0.0)
            st.metric("RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")
        
        with col4:
            mae = metrics.get('mae', 0.0)
            st.metric("MAE", f"{mae:.3f}" if mae is not None else "N/A")


def _reset_training_state():
    """Reset training state"""
    keys_to_reset = [
        'selected_data', 'model_config', 'advanced_settings', 
        'training_results', 'recommended_task_type', 'task_type_confidence'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    display_success_message("Training state reset successfully!")
    
    if st.session_state.get('debug_mode', False):
        debug_manager.log_debug("Training state reset")


def _save_training_config():
    """Save training configuration"""
    config = {}
    
    if 'selected_data' in st.session_state:
        config['selected_data'] = st.session_state.selected_data
    
    if 'model_config' in st.session_state:
        config['model_config'] = st.session_state.model_config
    
    if 'advanced_settings' in st.session_state:
        config['advanced_settings'] = st.session_state.advanced_settings
    
    if config:
        config_json = json.dumps(config, indent=2, default=str)
        
        st.download_button(
            label="💾 Download Configuration",
            data=config_json,
            file_name=f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        display_success_message("Configuration saved!")
    else:
        display_warning_message("No configuration to save")


def _export_training_results(results: Dict):
    """Export training results"""
    export_data = {
        'model_type': results['model_type'],
        'task_type': results['task_type'],
        'training_time': results['training_time'],
        'cv_scores': results['cv_scores'].tolist(),
        'metrics': results['metrics'],
        'parameters': results['parameters'],
        'timestamp': results['timestamp']
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="💾 Download Results",
        data=json_str,
        file_name=f"training_results_{results['model_type'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
