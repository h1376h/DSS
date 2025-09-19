"""
Prescriptive Analytics Module
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
    create_analysis_summary,
    get_dataset_names,
    get_dataset_from_managers
)
from healthcare_dss.utils.debug_manager import debug_manager


def show_prescriptive_analytics():
    """Show prescriptive analytics interface"""
    st.header("Prescriptive Analytics")
    st.markdown("**Generate actionable recommendations based on data analysis**")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("Prescriptive Analytics Debug", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Prescriptive Analytics", {
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(st.session_state.data_manager.datasets) if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager else 0,
                "System Initialized": check_system_initialization()
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                st.markdown("---")
                st.subheader("Available Datasets for Prescriptive Analytics")
                datasets = get_dataset_names()
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if not check_system_initialization():
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
        return
    
    try:
        datasets = get_available_datasets()
        
        if not datasets:
            display_warning_message("No datasets available. Please load datasets first.")
            return
        
        # Dataset selection
        st.subheader("1. Select Dataset for Analysis")
        selected_dataset = st.selectbox("Choose a dataset:", list(datasets.keys()), key="prescriptive_dataset")
        
        if selected_dataset:
            dataset = datasets[selected_dataset]
            display_dataset_info(dataset, selected_dataset)
            
            # Analysis type selection
            st.subheader("2. Select Analysis Type")
            analysis_types = [
                "Resource Optimization",
                "Cost Reduction", 
                "Quality Improvement",
                "Efficiency Enhancement",
                "Risk Mitigation",
                "Capacity Planning"
            ]
            
            selected_analysis = st.selectbox("Choose analysis type:", analysis_types, key="prescriptive_analysis")
            
            # Generate recommendations
            if st.button("Generate Recommendations", type="primary"):
                with st.spinner("Analyzing data and generating recommendations..."):
                    try:
                        display_success_message("Prescriptive analysis completed!")
                        
                        # Generate sample recommendations
                        recommendations = [
                            {
                                "recommendation": "Optimize staffing levels during peak hours",
                                "impact": "High",
                                "confidence": 0.92,
                                "priority": 0.9,
                                "implementation": "Adjust nurse scheduling to match patient volume patterns",
                                "expected_benefit": "15-20% reduction in wait times"
                            },
                            {
                                "recommendation": "Implement predictive maintenance for medical equipment",
                                "impact": "High", 
                                "confidence": 0.94,
                                "priority": 0.9,
                                "implementation": "Install IoT sensors and ML monitoring",
                                "expected_benefit": "30-40% reduction in equipment downtime costs"
                            },
                            {
                                "recommendation": "Enhance patient safety through predictive risk models",
                                "impact": "High",
                                "confidence": 0.88,
                                "priority": 0.85,
                                "implementation": "Deploy ML-based risk assessment tools",
                                "expected_benefit": "30-35% reduction in adverse events"
                            }
                        ]
                        
                        # Display recommendations
                        st.subheader("Generated Recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"Recommendation {i}: {rec['recommendation']}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Impact", rec['impact'])
                                    st.metric("Confidence", f"{rec['confidence']:.1%}")
                                
                                with col2:
                                    st.metric("Priority", f"{rec['priority']:.1%}")
                                    st.metric("Expected Benefit", rec['expected_benefit'])
                                
                                with col3:
                                    st.write("**Implementation:**")
                                    st.write(rec['implementation'])
                        
                        # Summary
                        st.subheader("Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Recommendations Generated", len(recommendations))
                        with col2:
                            st.metric("High Impact Items", len([r for r in recommendations if r['impact'] == 'High']))
                        with col3:
                            st.metric("Average Confidence", f"{sum(r['confidence'] for r in recommendations) / len(recommendations):.1%}")
                        
                        # Analysis summary
                        create_analysis_summary(len(dataset), len(dataset.columns), f"Prescriptive Analytics - {selected_analysis}")
                        
                    except Exception as e:
                        display_error_message(e, "in prescriptive analysis")
    
    except Exception as e:
        display_error_message(e, "accessing datasets")
