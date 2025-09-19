"""
Model Management Dashboard Module
"""

import streamlit as st
import pandas as pd
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message,
    display_success_message,
    display_warning_message,
    safe_dataframe_display
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_model_management():
    """Show model management interface"""
    st.header("Model Management")
    st.markdown("**Manage and monitor machine learning models**")
    
    # Check if user has permission to delete models
    user_role = st.session_state.get('user_role', '')
    can_delete_models = user_role in ['Clinical Leadership', 'Data Analyst']
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("Debug Information", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Model Management", {
                "Has Model Manager": hasattr(st.session_state, 'model_manager'),
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(st.session_state.data_manager.datasets) if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager else 0,
                "System Initialized": check_system_initialization(),
                "User Role": user_role,
                "Can Delete Models": can_delete_models
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager:
                st.markdown("---")
                st.subheader("Model Manager Status")
                st.success("Model Manager Available")
            else:
                st.warning("Model Manager Not Available")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                st.markdown("---")
                st.subheader("Available Datasets for Model Training")
                from healthcare_dss.ui.utils.common import get_dataset_names, get_dataset_from_managers
                datasets = get_dataset_names()
                st.write(f"**Total datasets:** {len(datasets)}")
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if check_system_initialization():
        try:
            # Get available models from the model registry
            registry = st.session_state.model_manager.registry
            
            if hasattr(registry, 'list_models'):
                models_df = registry.list_models()
                
                if not models_df.empty:
                    # Header with model count and bulk actions
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.subheader(f"Available Models ({len(models_df)} total)")
                    
                    with col2:
                        if st.button("View Details", key="view_all_details"):
                            st.session_state['show_all_details'] = True
                    
                    with col3:
                        if can_delete_models:
                            if st.button("Delete All Models", key="delete_all_models", type="secondary"):
                                st.session_state['confirm_delete_all'] = True
                                st.rerun()
                    
                    # Show confirmation dialog for delete all
                    if st.session_state.get('confirm_delete_all', False):
                        st.error("**Confirm Delete All Models**")
                        st.write(f"This will permanently delete all {len(models_df)} models.")
                        st.write("**This action cannot be undone!**")
                        
                        col_confirm, col_cancel = st.columns(2)
                        
                        with col_confirm:
                            if st.button("Confirm Delete All", key="confirm_delete_all_yes", type="primary"):
                                deleted_count = 0
                                failed_count = 0
                                
                                for index, row in models_df.iterrows():
                                    model_key = row.get('model_key', f'model_{index}')
                                    success = registry.delete_model(model_key)
                                    if success:
                                        deleted_count += 1
                                    else:
                                        failed_count += 1
                                
                                if failed_count == 0:
                                    display_success_message(f"Successfully deleted all {deleted_count} models.")
                                else:
                                    display_warning_message(f"Deleted {deleted_count} models. {failed_count} deletions failed.")
                                
                                # Clean up session state
                                del st.session_state['confirm_delete_all']
                                st.rerun()
                        
                        with col_cancel:
                            if st.button("Cancel", key="cancel_delete_all"):
                                del st.session_state['confirm_delete_all']
                                st.rerun()
                    
                    # Show all model details if requested
                    if st.session_state.get('show_all_details', False):
                        st.subheader("All Model Details")
                        st.dataframe(models_df)
                        if st.button("Hide Details", key="hide_all_details"):
                            del st.session_state['show_all_details']
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Display individual models
                    for index, row in models_df.iterrows():
                        model_key = row.get('model_key', f'model_{index}')
                        model_name = row.get('model_name', row.get('model_key', 'Unknown Model'))
                        
                        with st.expander(f"{model_name}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Model Key:** {model_key}")
                                st.write(f"**Task Type:** {row.get('task_type', 'Unknown')}")
                                st.write(f"**Dataset:** {row.get('dataset_name', 'Unknown')}")
                                st.write(f"**Target Column:** {row.get('target_column', 'N/A')}")
                                st.write(f"**Created:** {row.get('created_at', 'Unknown')}")
                                st.write(f"**Status:** {row.get('status', 'Unknown')}")
                                st.write(f"**Version:** {row.get('model_version', 'N/A')}")
                                
                                if row.get('description'):
                                    st.write(f"**Description:** {row.get('description')}")
                            
                            with col2:
                                if st.button("View Details", key=f"details_{model_key}"):
                                    st.json(row.to_dict())
                                
                                # Delete button with permission check
                                if can_delete_models:
                                    if st.button("Delete", key=f"delete_{model_key}", type="secondary"):
                                        # Store model info for confirmation
                                        st.session_state[f'delete_model_{model_key}'] = {
                                            'model_key': model_key,
                                            'model_name': model_name,
                                            'dataset': row.get('dataset_name', 'Unknown'),
                                            'created': row.get('created_at', 'Unknown')
                                        }
                                        st.rerun()
                                
                                # Show confirmation dialog if delete was clicked
                                if f'delete_model_{model_key}' in st.session_state:
                                    model_info = st.session_state[f'delete_model_{model_key}']
                                    
                                    st.warning("**Confirm Model Deletion**")
                                    st.write(f"**Model:** {model_info['model_name']}")
                                    st.write(f"**Dataset:** {model_info['dataset']}")
                                    st.write(f"**Created:** {model_info['created']}")
                                    st.write("**This action cannot be undone!**")
                                    
                                    col_confirm, col_cancel = st.columns(2)
                                    
                                    with col_confirm:
                                        if st.button("Confirm Delete", key=f"confirm_delete_{model_key}", type="primary"):
                                            # Delete the model
                                            success = registry.delete_model(model_key)
                                            
                                            if success:
                                                display_success_message(f"Model '{model_info['model_name']}' deleted successfully!")
                                                # Clean up session state
                                                del st.session_state[f'delete_model_{model_key}']
                                                st.rerun()
                                            else:
                                                display_error_message(f"Failed to delete model '{model_info['model_name']}'")
                                    
                                    with col_cancel:
                                        if st.button("Cancel", key=f"cancel_delete_{model_key}"):
                                            # Clean up session state
                                            del st.session_state[f'delete_model_{model_key}']
                                            st.rerun()
                else:
                    st.info("No models available. Train some models first.")
                    
                    # Add a section to show how to train models
                    st.subheader("How to Train Models")
                    st.markdown("""
                    To train models, you can:
                    1. Use the **Analytics Dashboard** to run data analysis
                    2. Use the **CRISP-DM Workflow** for structured model development
                    3. Use the **Model Training** functionality in the analytics section
                    """)
                    
                    # Show permission info
                    if can_delete_models:
                        st.info("**Note:** As a user with model management permissions, you will be able to delete models once they are trained.")
                    else:
                        st.info("**Note:** Model deletion is restricted to Clinical Leadership and Data Analyst roles.")
            else:
                st.info("Model registry not available. Please check the model management system.")
                
        except Exception as e:
            display_error_message(e, "accessing models")
            st.markdown("**Debug Information:**")
            st.code(f"Exception type: {type(e).__name__}")
            st.code(f"Exception details: {str(e)}")
    else:
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
