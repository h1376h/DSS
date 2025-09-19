#!/usr/bin/env python3
"""
Smart Target Suggestions Helper
Provides easy access to smart target suggestions throughout the DSS application
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_smart_target_suggestions(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get smart target suggestions for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing smart target suggestions or None if not available
    """
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        
        return {
            'targets': smart_manager.get_dataset_targets(dataset_name),
            'functionalities': smart_manager.get_smart_functionalities(dataset_name),
            'models': smart_manager.get_recommended_models(dataset_name),
            'insights': smart_manager.get_smart_insights(dataset_name)
        }
    except ImportError:
        logger.warning("Smart target manager not available")
        return None
    except Exception as e:
        logger.error(f"Error getting smart suggestions: {e}")
        return None

def render_smart_target_suggestions(dataset_name: str, key_prefix: str = "smart") -> Optional[str]:
    """
    Render smart target suggestions in Streamlit UI
    
    Args:
        dataset_name: Name of the dataset
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Selected target column name or None
    """
    suggestions = get_smart_target_suggestions(dataset_name)
    
    if not suggestions or not suggestions['targets']:
        return None
    
    st.info("Smart Target Recommendations Available!")
    
    # Show smart targets
    st.write("**Recommended Target Variables:**")
    selected_target = None
    
    for i, target in enumerate(suggestions['targets']):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{target['column']}** ({target['target_type']})")
            st.caption(target.get('business_meaning', 'Target variable'))
        
        with col2:
            # Convert unique_values to int for display
            unique_values = int(target['unique_values']) if isinstance(target['unique_values'], str) else target['unique_values']
            st.write(f"Unique values: {unique_values}")
            
            # Convert missing_values to int for comparison
            missing_values = int(target['missing_values']) if isinstance(target['missing_values'], str) else target['missing_values']
            if missing_values > 0:
                st.warning(f"Missing: {missing_values}")
        
        with col3:
            if st.button(f"Select", key=f"{key_prefix}_select_{i}"):
                selected_target = target['column']
                st.session_state[f"{key_prefix}_selected_target"] = selected_target
                st.rerun()
    
    # Show smart functionalities
    if suggestions['functionalities']:
        st.markdown("---")
        st.write("**Available Smart Features:**")
        for feature in suggestions['functionalities'][:5]:  # Show first 5
            st.write(f"• {feature}")
        if len(suggestions['functionalities']) > 5:
            st.write(f"... and {len(suggestions['functionalities']) - 5} more")
    
    st.markdown("---")
    
    return selected_target

def render_smart_model_suggestions(dataset_name: str, target_column: str, key_prefix: str = "smart") -> Optional[Dict[str, Any]]:
    """
    Render smart model suggestions in Streamlit UI
    
    Args:
        dataset_name: Name of the dataset
        target_column: Name of the target column
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Dictionary containing model recommendations or None
    """
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        
        model_recommendations = smart_manager.get_model_recommendations(dataset_name, target_column)
        
        if not model_recommendations.get('recommended_models'):
            return None
        
        st.info("Smart Model Recommendations Available!")
        
        # Show recommended models
        st.write("**Recommended Models:**")
        for model in model_recommendations['recommended_models']:
            st.write(f"• {model}")
        
        # Show recommended metrics
        if model_recommendations.get('metrics'):
            st.write("**Recommended Metrics:**")
            for metric in model_recommendations['metrics']:
                st.write(f"• {metric}")
        
        # Show recommended visualizations
        if model_recommendations.get('visualizations'):
            st.write("**Recommended Visualizations:**")
            for viz in model_recommendations['visualizations']:
                st.write(f"• {viz}")
        
        st.markdown("---")
        
        return model_recommendations
        
    except Exception as e:
        logger.error(f"Error getting smart model suggestions: {e}")
        return None

def get_smart_preprocessing_suggestions(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get smart preprocessing suggestions for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing preprocessing suggestions or None
    """
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        
        return smart_manager.get_preprocessing_recommendations(dataset_name)
        
    except Exception as e:
        logger.error(f"Error getting smart preprocessing suggestions: {e}")
        return None

def render_smart_preprocessing_suggestions(dataset_name: str, key_prefix: str = "smart") -> Optional[Dict[str, Any]]:
    """
    Render smart preprocessing suggestions in Streamlit UI
    
    Args:
        dataset_name: Name of the dataset
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Dictionary containing preprocessing suggestions or None
    """
    suggestions = get_smart_preprocessing_suggestions(dataset_name)
    
    if not suggestions:
        return None
    
    st.info("Smart Preprocessing Recommendations Available!")
    
    # Show preprocessing steps
    st.write("**Recommended Preprocessing Steps:**")
    for step in suggestions.get('steps', []):
        st.write(f"• {step}")
    
    # Show parameters
    if suggestions.get('parameters'):
        st.write("**Recommended Parameters:**")
        for param, value in suggestions['parameters'].items():
            st.write(f"• {param}: {value}")
    
    # Show dataset-specific needs
    if suggestions.get('dataset_specific_needs'):
        st.write("**Dataset-Specific Needs:**")
        for need in suggestions['dataset_specific_needs']:
            st.write(f"• {need}")
    
    st.markdown("---")
    
    return suggestions

def get_smart_insights_for_dataset(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get smart insights for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing smart insights or None
    """
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        
        return smart_manager.get_smart_insights(dataset_name)
        
    except Exception as e:
        logger.error(f"Error getting smart insights: {e}")
        return None

def render_smart_insights(dataset_name: str, key_prefix: str = "smart") -> Optional[Dict[str, Any]]:
    """
    Render smart insights in Streamlit UI
    
    Args:
        dataset_name: Name of the dataset
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Dictionary containing smart insights or None
    """
    insights = get_smart_insights_for_dataset(dataset_name)
    
    if not insights:
        return None
    
    st.info("Smart Insights Available!")
    
    # Show business value
    if insights.get('business_value'):
        st.write("**Business Value:**")
        for value in insights['business_value']:
            st.write(f"• {value}")
    
    # Show recommended use cases
    if insights.get('recommended_use_cases'):
        st.write("**Recommended Use Cases:**")
        for use_case in insights['recommended_use_cases']:
            st.write(f"• {use_case}")
    
    # Show integration suggestions
    if insights.get('integration_suggestions'):
        st.write("**Integration Suggestions:**")
        for suggestion in insights['integration_suggestions']:
            st.write(f"• {suggestion}")
    
    st.markdown("---")
    
    return insights

def create_smart_target_selector(dataset_name: str, df: pd.DataFrame, key_prefix: str = "smart") -> str:
    """
    Create a smart target selector that combines smart suggestions with standard selection
    
    Args:
        dataset_name: Name of the dataset
        df: DataFrame containing the data
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Selected target column name
    """
    # Try to get smart suggestions
    selected_target = render_smart_target_suggestions(dataset_name, key_prefix)
    
    if selected_target:
        return selected_target
    
    # Fall back to standard selection
    return st.selectbox(
        "Select Target Column",
        df.columns.tolist(),
        key=f"{key_prefix}_target_select"
    )

def create_smart_model_selector(dataset_name: str, target_column: str, task_type: str, key_prefix: str = "smart") -> str:
    """
    Create a smart model selector that prioritizes smart recommendations
    
    Args:
        dataset_name: Name of the dataset
        target_column: Name of the target column
        task_type: Type of ML task (classification/regression)
        key_prefix: Prefix for Streamlit keys
        
    Returns:
        Selected model name
    """
    # Get smart model recommendations
    model_recommendations = render_smart_model_suggestions(dataset_name, target_column, key_prefix)
    
    # Base model options
    if task_type == "classification":
        base_models = ["random_forest", "xgboost", "lightgbm", "svm", "neural_network", "knn", "logistic_regression"]
        model_descriptions = {
            "random_forest": "Random Forest - Robust, handles mixed data types well",
            "xgboost": "XGBoost - High performance, good for structured data",
            "lightgbm": "LightGBM - Fast training, memory efficient",
            "svm": "Support Vector Machine - Good for high-dimensional data",
            "neural_network": "Neural Network - Can capture complex patterns",
            "knn": "K-Nearest Neighbors - Simple, interpretable",
            "logistic_regression": "Logistic Regression - Linear, interpretable"
        }
    else:  # regression
        base_models = ["random_forest", "xgboost", "lightgbm", "linear_regression", "neural_network", "knn", "svm"]
        model_descriptions = {
            "random_forest": "Random Forest - Robust, handles mixed data types well",
            "xgboost": "XGBoost - High performance, good for structured data",
            "lightgbm": "LightGBM - Fast training, memory efficient",
            "linear_regression": "Linear Regression - Simple, interpretable",
            "neural_network": "Neural Network - Can capture complex patterns",
            "knn": "K-Nearest Neighbors - Simple, interpretable",
            "svm": "Support Vector Machine - Good for high-dimensional data"
        }
    
    # Prioritize smart recommendations
    if model_recommendations and model_recommendations.get('recommended_models'):
        # Map smart recommendations to our model names
        smart_model_mapping = {
            'RandomForestClassifier': 'random_forest',
            'RandomForestRegressor': 'random_forest',
            'XGBClassifier': 'xgboost',
            'XGBRegressor': 'xgboost',
            'SVM': 'svm',
            'SVR': 'svm',
            'LogisticRegression': 'logistic_regression',
            'LinearRegression': 'linear_regression',
            'NaiveBayes': 'naive_bayes',
            'KNeighborsClassifier': 'knn',
            'KNeighborsRegressor': 'knn'
        }
        
        # Get smart recommended models
        smart_recommended = []
        for model in model_recommendations['recommended_models']:
            mapped_model = smart_model_mapping.get(model, model.lower().replace('classifier', '').replace('regressor', ''))
            if mapped_model in base_models:
                smart_recommended.append(mapped_model)
        
        # Reorder model options to prioritize smart recommendations
        model_options = smart_recommended + [m for m in base_models if m not in smart_recommended]
        
        # Add smart indicators to descriptions
        for model in smart_recommended:
            if model in model_descriptions:
                model_descriptions[model] += " (Recommended)"
    else:
        model_options = base_models
    
    return st.selectbox(
        "Select Model Type",
        model_options,
        format_func=lambda x: model_descriptions.get(x, x),
        key=f"{key_prefix}_model_select"
    )

# Example usage functions for different UI components
def show_smart_target_suggestions_sidebar(dataset_name: str):
    """Show smart target suggestions in sidebar"""
    with st.sidebar:
        st.subheader("Smart Suggestions")
        suggestions = get_smart_target_suggestions(dataset_name)
        
        if suggestions:
            if suggestions['targets']:
                st.write("**Recommended Targets:**")
                for target in suggestions['targets'][:3]:  # Show top 3
                    st.write(f"• {target['column']} ({target['target_type']})")
            
            if suggestions['functionalities']:
                st.write("**Smart Features:**")
                for feature in suggestions['functionalities'][:3]:  # Show top 3
                    st.write(f"• {feature}")
        else:
            st.info("No smart suggestions available for this dataset")

def show_smart_insights_expander(dataset_name: str):
    """Show smart insights in an expander"""
    with st.expander("Smart Insights", expanded=False):
        insights = get_smart_insights_for_dataset(dataset_name)
        
        if insights:
            if insights.get('business_value'):
                st.write("**Business Value:**")
                for value in insights['business_value']:
                    st.write(f"• {value}")
            
            if insights.get('recommended_use_cases'):
                st.write("**Use Cases:**")
                for use_case in insights['recommended_use_cases']:
                    st.write(f"• {use_case}")
        else:
            st.info("No smart insights available for this dataset")
