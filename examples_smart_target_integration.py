#!/usr/bin/env python3
"""
Smart Target Integration Examples
Shows how to integrate smart target suggestions throughout the DSS application
"""

import streamlit as st
import pandas as pd
from healthcare_dss.utils.smart_target_suggestions import (
    get_smart_target_suggestions,
    render_smart_target_suggestions,
    render_smart_model_suggestions,
    render_smart_preprocessing_suggestions,
    render_smart_insights,
    create_smart_target_selector,
    create_smart_model_selector,
    show_smart_target_suggestions_sidebar,
    show_smart_insights_expander
)

def example_model_training_with_smart_suggestions():
    """Example of enhanced model training with smart suggestions"""
    
    st.title("Enhanced Model Training with Smart Suggestions")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["breast_cancer_scikit", "diabetes_scikit", "clinical_outcomes_synthetic"])
    
    if dataset_name:
        # Load dataset
        df = pd.read_csv(f"datasets/raw/{dataset_name}.csv")
        
        # Show smart suggestions in sidebar
        show_smart_target_suggestions_sidebar(dataset_name)
        
        # Smart target selection
        st.subheader("Target Selection")
        selected_target = create_smart_target_selector(dataset_name, df, "training")
        
        if selected_target:
            # Analyze target
            st.write(f"Selected target: **{selected_target}**")
            
            # Determine task type
            unique_values = df[selected_target].nunique()
            if unique_values <= 10:
                task_type = "classification"
            else:
                task_type = "regression"
            
            st.write(f"Detected task type: **{task_type}**")
            
            # Smart model selection
            st.subheader("Model Selection")
            selected_model = create_smart_model_selector(dataset_name, selected_target, task_type, "training")
            
            # Show smart preprocessing suggestions
            st.subheader("Preprocessing Suggestions")
            preprocessing_suggestions = render_smart_preprocessing_suggestions(dataset_name, "training")
            
            # Show smart insights
            show_smart_insights_expander(dataset_name)
            
            # Training button
            if st.button("Train Model with Smart Configuration"):
                st.success("Model training would start with smart configuration!")
                st.write(f"Dataset: {dataset_name}")
                st.write(f"Target: {selected_target}")
                st.write(f"Task Type: {task_type}")
                st.write(f"Model: {selected_model}")

def example_analytics_with_smart_suggestions():
    """Example of analytics with smart suggestions"""
    
    st.title("Analytics with Smart Suggestions")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["breast_cancer_scikit", "diabetes_scikit", "clinical_outcomes_synthetic"])
    
    if dataset_name:
        # Load dataset
        df = pd.read_csv(f"datasets/raw/{dataset_name}.csv")
        
        # Get smart suggestions
        suggestions = get_smart_target_suggestions(dataset_name)
        
        if suggestions:
            st.info("Smart Suggestions Available!")
            
            # Show recommended targets
            if suggestions['targets']:
                st.subheader("Recommended Analysis Targets")
                for target in suggestions['targets']:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{target['column']}** ({target['target_type']})")
                        st.caption(target.get('business_meaning', 'Target variable'))
                    with col2:
                        if st.button(f"Analyze", key=f"analyze_{target['column']}"):
                            st.write(f"Analyzing {target['column']}...")
                            # Here you would run the actual analysis
            
            # Show smart functionalities
            if suggestions['functionalities']:
                st.subheader("Available Smart Features")
                for feature in suggestions['functionalities']:
                    st.write(f"• {feature}")
        
        else:
            st.warning("No smart suggestions available for this dataset")

def example_dashboard_with_smart_suggestions():
    """Example of dashboard with smart suggestions"""
    
    st.title("Dashboard with Smart Suggestions")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["breast_cancer_scikit", "diabetes_scikit", "clinical_outcomes_synthetic"])
    
    if dataset_name:
        # Load dataset
        df = pd.read_csv(f"datasets/raw/{dataset_name}.csv")
        
        # Show smart insights
        show_smart_insights_expander(dataset_name)
        
        # Get smart suggestions
        suggestions = get_smart_target_suggestions(dataset_name)
        
        if suggestions and suggestions['targets']:
            # Create tabs for different targets
            target_names = [target['column'] for target in suggestions['targets']]
            tabs = st.tabs(target_names)
            
            for i, (tab, target) in enumerate(zip(tabs, suggestions['targets'])):
                with tab:
                    st.subheader(f"Analysis: {target['column']}")
                    st.write(f"**Type:** {target['target_type']}")
                    st.write(f"**Business Meaning:** {target.get('business_meaning', 'Target variable')}")
                    
                    # Show target distribution
                    if target['target_type'] == 'binary_classification':
                        st.bar_chart(df[target['column']].value_counts())
                    elif target['target_type'] == 'regression':
                        st.line_chart(df[target['column']].head(100))
                    
                    # Show smart features for this target
                    if suggestions['functionalities']:
                        st.write("**Available Features:**")
                        for feature in suggestions['functionalities']:
                            st.write(f"• {feature}")

def example_workflow_with_smart_suggestions():
    """Example of workflow with smart suggestions"""
    
    st.title("Workflow with Smart Suggestions")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["breast_cancer_scikit", "diabetes_scikit", "clinical_outcomes_synthetic"])
    
    if dataset_name:
        # Load dataset
        df = pd.read_csv(f"datasets/raw/{dataset_name}.csv")
        
        # Get smart suggestions
        suggestions = get_smart_target_suggestions(dataset_name)
        
        if suggestions:
            st.info("Smart Workflow Recommendations Available!")
            
            # Show workflow steps
            st.subheader("Recommended Workflow Steps")
            
            steps = [
                "1. Data Exploration and Understanding",
                "2. Target Variable Analysis",
                "3. Feature Engineering",
                "4. Model Selection and Training",
                "5. Model Evaluation",
                "6. Deployment and Monitoring"
            ]
            
            for step in steps:
                st.write(f"• {step}")
            
            # Show specific recommendations
            if suggestions['targets']:
                st.subheader("Target-Specific Recommendations")
                for target in suggestions['targets']:
                    st.write(f"**For {target['column']}:**")
                    st.write(f"  - Task Type: {target['target_type']}")
                    st.write(f"  - Business Value: {target.get('business_meaning', 'Target variable')}")
                    
                    # Get model recommendations for this target
                    try:
                        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
                        smart_manager = SmartDatasetTargetManager()
                        model_recs = smart_manager.get_model_recommendations(dataset_name, target['column'])
                        
                        if model_recs.get('recommended_models'):
                            st.write(f"  - Recommended Models: {', '.join(model_recs['recommended_models'])}")
                        
                        if model_recs.get('metrics'):
                            st.write(f"  - Recommended Metrics: {', '.join(model_recs['metrics'])}")
                    
                    except Exception as e:
                        st.write(f"  - Error getting recommendations: {e}")
            
            # Show smart functionalities
            if suggestions['functionalities']:
                st.subheader("Smart Features to Implement")
                for feature in suggestions['functionalities']:
                    st.write(f"• {feature}")

def main():
    """Main function to demonstrate smart target integration"""
    
    st.set_page_config(page_title="Smart Target Integration Examples", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Smart Target Integration Examples")
    
    example_type = st.sidebar.selectbox(
        "Select Example",
        [
            "Model Training",
            "Analytics",
            "Dashboard",
            "Workflow"
        ]
    )
    
    # Show selected example
    if example_type == "Model Training":
        example_model_training_with_smart_suggestions()
    elif example_type == "Analytics":
        example_analytics_with_smart_suggestions()
    elif example_type == "Dashboard":
        example_dashboard_with_smart_suggestions()
    elif example_type == "Workflow":
        example_workflow_with_smart_suggestions()
    
    # Show smart target manager status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Smart Target Manager Status")
    
    try:
        from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
        smart_manager = SmartDatasetTargetManager()
        summary = smart_manager.get_dataset_summary()
        
        st.sidebar.write(f"**Total Datasets:** {summary['total_datasets']}")
        st.sidebar.write(f"**Healthcare Datasets:** {summary['healthcare_datasets']}")
        st.sidebar.write(f"**Classification Datasets:** {summary['classification_datasets']}")
        st.sidebar.write(f"**Regression Datasets:** {summary['regression_datasets']}")
        
        st.sidebar.success("Smart Target Manager Active")
        
    except Exception as e:
        st.sidebar.error(f"Smart Target Manager Error: {e}")

if __name__ == "__main__":
    main()
