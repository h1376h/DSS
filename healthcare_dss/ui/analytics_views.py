"""
Analytics Views Module
=====================

Contains analytics and knowledge management views for the Healthcare DSS:
- Knowledge Management
- KPI Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import logging
from healthcare_dss.ui.utils.common import safe_dataframe_display

def show_knowledge_management():
    """Show enhanced knowledge management interface with interactive features"""
    st.header("Knowledge Management")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("Knowledge Management Debug", expanded=True):
            st.write("**Knowledge Management Debug:**")
            st.write(f"- Function called: show_knowledge_management()")
            st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
            st.write(f"- Has knowledge_manager: {hasattr(st.session_state, 'knowledge_manager')}")
            st.write(f"- Debug mode: {st.session_state.get('debug_mode', False)}")
            
            if hasattr(st.session_state, 'knowledge_manager') and st.session_state.knowledge_manager is not None:
                try:
                    knowledge_summary = st.session_state.knowledge_manager.get_knowledge_summary()
                    st.write(f"- Knowledge summary: {knowledge_summary}")
                except Exception as e:
                    st.write(f"- Error getting knowledge summary: {e}")
            else:
                st.write("- Knowledge Manager not available")
    
    # Knowledge base overview with enhanced metrics
    st.subheader("Knowledge Base Overview")
    
    try:
        if hasattr(st.session_state, 'knowledge_manager') and st.session_state.knowledge_manager:
            knowledge_summary = st.session_state.knowledge_manager.get_knowledge_summary()
        else:
            # Mock knowledge summary
            knowledge_summary = {
                'clinical_rules': 15,
                'clinical_guidelines': 8,
                'decision_trees': 5,
                'active_rules': 12
            }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Clinical Rules", 
                knowledge_summary['clinical_rules'],
                delta=f"+{knowledge_summary['clinical_rules']}"
            )
        
        with col2:
            st.metric(
                "Guidelines", 
                knowledge_summary['clinical_guidelines'],
                delta=f"+{knowledge_summary['clinical_guidelines']}"
            )
        
        with col3:
            st.metric(
                "Decision Trees", 
                knowledge_summary['decision_trees'],
                delta=f"+{knowledge_summary['decision_trees']}"
            )
        
        with col4:
            st.metric(
                "Active Rules", 
                knowledge_summary['active_rules'],
                delta=f"+{knowledge_summary['active_rules']}"
            )
        
        with col5:
            total_knowledge = (knowledge_summary['clinical_rules'] + 
                             knowledge_summary['clinical_guidelines'] + 
                             knowledge_summary['decision_trees'])
            st.metric(
                "Total Knowledge", 
                total_knowledge,
                delta=f"+{total_knowledge}"
            )
        
        st.markdown("---")
        
        # Enhanced knowledge management with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Explore Knowledge", "Patient Assessment", "Knowledge Analytics", "Management"])
        
        with tab1:
            st.subheader("Knowledge Base Explorer")
            
            # Knowledge categories
            knowledge_type = st.selectbox(
                "Select Knowledge Type",
                ["Clinical Rules", "Clinical Guidelines", "Decision Trees", "All"]
            )
            
            if knowledge_type == "Clinical Rules":
                try:
                    if hasattr(st.session_state, 'knowledge_manager') and st.session_state.knowledge_manager:
                        rules = st.session_state.knowledge_manager.get_clinical_rules()
                    else:
                        # Mock clinical rules
                        rules = [
                            {
                                'name': 'Diabetes Risk Assessment',
                                'severity': 'High',
                                'condition': 'HbA1c > 7.0%',
                                'action': 'Consider insulin therapy',
                                'evidence_level': 'A',
                                'source': 'ADA Guidelines 2023'
                            },
                            {
                                'name': 'Hypertension Management',
                                'severity': 'Medium',
                                'condition': 'Systolic BP > 140 mmHg',
                                'action': 'Start antihypertensive medication',
                                'evidence_level': 'B',
                                'source': 'AHA Guidelines 2023'
                            },
                            {
                                'name': 'Cardiovascular Risk',
                                'severity': 'High',
                                'condition': 'Age > 65 AND smoking',
                                'action': 'Cardiovascular assessment',
                                'evidence_level': 'A',
                                'source': 'ACC Guidelines 2023'
                            }
                        ]
                    
                    if rules:
                        st.write(f"**Found {len(rules)} Clinical Rules:**")
                        
                        for i, rule in enumerate(rules, 1):
                            with st.expander(f"Rule {i}: {rule.get('name', 'Unnamed Rule')}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Name:** {rule.get('name', 'N/A')}")
                                    st.write(f"**Severity:** {rule.get('severity', 'N/A')}")
                                    st.write(f"**Condition:** {rule.get('condition', 'N/A')}")
                                
                                with col2:
                                    st.write(f"**Action:** {rule.get('action', 'N/A')}")
                                    st.write(f"**Evidence Level:** {rule.get('evidence_level', 'N/A')}")
                                    st.write(f"**Source:** {rule.get('source', 'N/A')}")
                    else:
                        st.info("No clinical rules found.")
                        
                except Exception as e:
                    st.error(f"Error retrieving clinical rules: {e}")
            
            elif knowledge_type == "All":
                st.subheader("Complete Knowledge Base")
                
                # Create a comprehensive knowledge overview
                knowledge_overview = {
                    'Type': ['Clinical Rules', 'Clinical Guidelines', 'Decision Trees'],
                    'Count': [
                        knowledge_summary['clinical_rules'],
                        knowledge_summary['clinical_guidelines'],
                        knowledge_summary['decision_trees']
                    ],
                    'Status': ['Active', 'Active', 'Active']
                }
                
                overview_df = pd.DataFrame(knowledge_overview)
                safe_dataframe_display(overview_df, width="stretch")
                
                # Knowledge distribution pie chart
                fig_knowledge = px.pie(
                    overview_df,
                    values='Count',
                    names='Type',
                    title='Knowledge Base Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_knowledge, width="stretch")
        
        with tab2:
            st.subheader("Interactive Patient Assessment")
            
            # Enhanced patient input form
            st.write("**Enter Patient Information:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Demographics:**")
                age = st.number_input("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Patient's gender")
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, help="Body Mass Index")
            
            with col2:
                st.write("**Vital Signs:**")
                systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=250, value=120, help="Systolic blood pressure")
                diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80, help="Diastolic blood pressure")
                heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72, help="Heart rate per minute")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Medical History:**")
                family_history = st.checkbox("Family History of Diabetes", help="Family history of diabetes")
                smoking = st.checkbox("Current Smoker", help="Current smoking status")
                exercise = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"], help="Physical activity level")
            
            with col4:
                st.write("**Lab Values:**")
                hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, help="Hemoglobin A1c percentage")
                glucose = st.number_input("Glucose (mg/dL)", min_value=70, max_value=500, value=100, help="Blood glucose level")
                cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, help="Total cholesterol")
            
            # Assessment button
            if st.button("Assess Patient", type="primary", width="stretch"):
                patient_data = {
                    'age': age,
                    'gender': gender,
                    'bmi': bmi,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'heart_rate': heart_rate,
                    'family_history': family_history,
                    'smoking': smoking,
                    'exercise': exercise,
                    'hba1c': hba1c,
                    'glucose': glucose,
                    'cholesterol': cholesterol
                }
                
                with st.spinner("Evaluating patient data..."):
                    try:
                        # Evaluate clinical rules
                        if hasattr(st.session_state, 'knowledge_manager') and st.session_state.knowledge_manager:
                            triggered_rules = st.session_state.knowledge_manager.evaluate_clinical_rules(patient_data)
                        else:
                            # Mock triggered rules based on patient data
                            triggered_rules = []
                            if hba1c > 7.0:
                                triggered_rules.append({
                                    'name': 'Diabetes Risk Assessment',
                                    'severity': 'High',
                                    'condition': 'HbA1c > 7.0%',
                                    'action': 'Consider insulin therapy',
                                    'evidence_level': 'A'
                                })
                            if systolic_bp > 140 or diastolic_bp > 90:
                                triggered_rules.append({
                                    'name': 'Hypertension Management',
                                    'severity': 'Medium',
                                    'condition': 'Systolic BP > 140 mmHg',
                                    'action': 'Start antihypertensive medication',
                                    'evidence_level': 'B'
                                })
                            if age > 65 and smoking:
                                triggered_rules.append({
                                    'name': 'Cardiovascular Risk',
                                    'severity': 'High',
                                    'condition': 'Age > 65 AND smoking',
                                    'action': 'Cardiovascular assessment',
                                    'evidence_level': 'A'
                                })
                        
                        # Display results in tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs(["Alerts", "Recommendations", "Risk Assessment"])
                        
                        with result_tab1:
                            st.subheader("Clinical Alerts")
                            
                            if triggered_rules:
                                for i, rule in enumerate(triggered_rules, 1):
                                    severity_color = {
                                        'high': '🔴',
                                        'medium': '🟡', 
                                        'low': '🟢'
                                    }.get(rule.get('severity', 'low').lower(), '⚪')
                                    
                                    with st.expander(f"{severity_color} Alert {i}: {rule.get('name', 'Unnamed Rule')}"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write(f"**Severity:** {rule.get('severity', 'N/A')}")
                                            st.write(f"**Condition:** {rule.get('condition', 'N/A')}")
                                        
                                        with col2:
                                            st.write(f"**Action:** {rule.get('action', 'N/A')}")
                                            st.write(f"**Evidence Level:** {rule.get('evidence_level', 'N/A')}")
                            else:
                                st.success("No clinical alerts triggered")
                        
                        with result_tab2:
                            st.subheader("Clinical Recommendations")
                            
                            try:
                                if hasattr(st.session_state, 'knowledge_manager') and st.session_state.knowledge_manager:
                                    recommendations = st.session_state.knowledge_manager.get_clinical_recommendations(patient_data)
                                else:
                                    # Mock recommendations based on patient data
                                    recommendations = []
                                    if hba1c > 6.5:
                                        recommendations.append({
                                            'recommendation': 'Consider diabetes management program',
                                            'source': 'ADA Guidelines',
                                            'evidence_level': 'A',
                                            'priority': 'High'
                                        })
                                    if bmi > 30:
                                        recommendations.append({
                                            'recommendation': 'Weight management counseling',
                                            'source': 'CDC Guidelines',
                                            'evidence_level': 'B',
                                            'priority': 'Medium'
                                        })
                                    if smoking:
                                        recommendations.append({
                                            'recommendation': 'Smoking cessation program',
                                            'source': 'AHA Guidelines',
                                            'evidence_level': 'A',
                                            'priority': 'High'
                                        })
                                
                                if recommendations:
                                    for i, rec in enumerate(recommendations[:10], 1):
                                        with st.expander(f"Recommendation {i}"):
                                            st.write(f"**Recommendation:** {rec.get('recommendation', 'N/A')}")
                                            st.write(f"**Source:** {rec.get('source', 'N/A')}")
                                            st.write(f"**Evidence Level:** {rec.get('evidence_level', 'N/A')}")
                                            st.write(f"**Priority:** {rec.get('priority', 'N/A')}")
                                else:
                                    st.info("No specific recommendations available")
                                    
                            except Exception as e:
                                st.error(f"Error getting recommendations: {e}")
                        
                        with result_tab3:
                            st.subheader("Risk Assessment")
                            
                            # Calculate risk scores
                            risk_factors = []
                            risk_score = 0
                            
                            # Age risk
                            if age > 65:
                                risk_factors.append("Advanced age (>65)")
                                risk_score += 2
                            elif age > 45:
                                risk_factors.append("Middle age (45-65)")
                                risk_score += 1
                            
                            # BMI risk
                            if bmi > 30:
                                risk_factors.append("Obesity (BMI >30)")
                                risk_score += 2
                            elif bmi > 25:
                                risk_factors.append("Overweight (BMI 25-30)")
                                risk_score += 1
                            
                            # Blood pressure risk
                            if systolic_bp > 140 or diastolic_bp > 90:
                                risk_factors.append("Hypertension")
                                risk_score += 2
                            
                            # HbA1c risk
                            if hba1c > 6.5:
                                risk_factors.append("Diabetes (HbA1c >6.5%)")
                                risk_score += 3
                            elif hba1c > 5.7:
                                risk_factors.append("Pre-diabetes (HbA1c 5.7-6.5%)")
                                risk_score += 1
                            
                            # Family history risk
                            if family_history:
                                risk_factors.append("Family history of diabetes")
                                risk_score += 1
                            
                            # Smoking risk
                            if smoking:
                                risk_factors.append("Current smoker")
                                risk_score += 1
                            
                            # Display risk assessment
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Risk Score", risk_score)
                                
                                if risk_score >= 5:
                                    risk_level = "High"
                                    risk_color = "🔴"
                                elif risk_score >= 3:
                                    risk_level = "Moderate"
                                    risk_color = "🟡"
                                else:
                                    risk_level = "Low"
                                    risk_color = "🟢"
                                
                                st.metric("Risk Level", f"{risk_color} {risk_level}")
                            
                            with col2:
                                st.write("**Risk Factors Identified:**")
                                if risk_factors:
                                    for factor in risk_factors:
                                        st.write(f"• {factor}")
                                else:
                                    st.write("• No significant risk factors identified")
                            
                            # Risk visualization
                            risk_data = {
                                'Risk Level': ['Low', 'Moderate', 'High'],
                                'Score Range': ['0-2', '3-4', '5+'],
                                'Count': [1 if risk_score <= 2 else 0, 1 if 3 <= risk_score <= 4 else 0, 1 if risk_score >= 5 else 0]
                            }
                            
                            fig_risk = px.bar(
                                risk_data,
                                x='Risk Level',
                                y='Count',
                                title='Risk Level Distribution',
                                color='Risk Level',
                                color_discrete_map={'Low': '#32CD32', 'Moderate': '#FFD700', 'High': '#DC143C'}
                            )
                            st.plotly_chart(fig_risk, width="stretch")
                    
                    except Exception as e:
                        st.error(f"Error evaluating patient: {e}")
        
        with tab3:
            st.subheader("Knowledge Analytics")
            
            # Knowledge usage analytics
            st.write("**Knowledge Usage Statistics**")
            
            usage_data = {
                'Knowledge Type': ['Clinical Rules', 'Guidelines', 'Decision Trees'],
                'Usage Count': [245, 189, 156],
                'Success Rate': [94.2, 91.8, 88.5]
            }
            
            usage_df = pd.DataFrame(usage_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_usage = px.bar(usage_df, x='Knowledge Type', y='Usage Count', 
                                 title='Knowledge Usage Count')
                st.plotly_chart(fig_usage, width="stretch")
            
            with col2:
                fig_success = px.bar(usage_df, x='Knowledge Type', y='Success Rate',
                                   title='Knowledge Success Rate')
                st.plotly_chart(fig_success, width="stretch")
        
        with tab4:
            st.subheader("Knowledge Management")
            
            st.write("**Knowledge Base Administration**")
            
            # Knowledge management actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Add New Rule", width="stretch"):
                    st.info("Rule addition interface would be implemented here")
            
            with col2:
                if st.button("Update Guidelines", width="stretch"):
                    st.info("Guideline update interface would be implemented here")
            
            with col3:
                if st.button("Export Knowledge", width="stretch"):
                    st.info("Knowledge export functionality would be implemented here")
    
    except Exception as e:
        st.error(f"Error accessing knowledge management: {e}")
        st.info("Please ensure the knowledge manager is properly initialized.")


def show_kpi_dashboard():
    """Show enhanced KPI dashboard with interactive visualizations"""
    st.header("KPI Dashboard")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("KPI Dashboard Debug", expanded=True):
            st.write("**KPI Dashboard Debug:**")
            st.write(f"- Function called: show_kpi_dashboard()")
            st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
            st.write(f"- Has kpi_dashboard: {hasattr(st.session_state, 'kpi_dashboard')}")
            st.write(f"- Has data_manager: {hasattr(st.session_state, 'data_manager')}")
            st.write(f"- Debug mode: {st.session_state.get('debug_mode', False)}")
            
            if hasattr(st.session_state, 'kpi_dashboard') and st.session_state.kpi_dashboard is not None:
                try:
                    kpis = st.session_state.kpi_dashboard.calculate_healthcare_kpis()
                    st.write(f"- KPI metrics calculated: {len(kpis)}")
                    st.write(f"- KPI keys: {list(kpis.keys())}")
                except Exception as e:
                    st.write(f"- Error calculating KPIs: {e}")
            else:
                st.write("- KPI Dashboard not available")
    
    # KPI calculation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Key Performance Indicators")
    
    with col2:
        if st.button("Calculate KPIs", type="primary", width="stretch"):
            st.session_state.kpis_calculated = True
    
    with col3:
        if st.button("Generate Report", width="stretch"):
            st.session_state.generate_report = True
    
    # Auto-calculate KPIs on first load
    if 'kpis_calculated' not in st.session_state:
        st.session_state.kpis_calculated = True
    
    if st.session_state.kpis_calculated:
        with st.spinner("Calculating KPIs..."):
            try:
                if hasattr(st.session_state, 'kpi_dashboard') and st.session_state.kpi_dashboard:
                    kpis = st.session_state.kpi_dashboard.calculate_healthcare_kpis()
                elif hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                    # Use real data from consolidated DataManager
                    patient_metrics = st.session_state.data_manager.get_patient_metrics()
                    clinical_metrics = st.session_state.data_manager.get_clinical_metrics()
                    financial_metrics = st.session_state.data_manager.get_financial_metrics()
                    staff_metrics = st.session_state.data_manager.get_staff_metrics()
                    department_metrics = st.session_state.data_manager.get_department_metrics()
                    
                    kpis = {
                        'system_datasets_loaded': len(st.session_state.data_manager.datasets),
                        'system_total_records': patient_metrics['total_patients'],
                        'system_avg_data_quality': clinical_metrics['average_quality_score'],
                        'diabetes_patients_total': patient_metrics['total_patients'],
                        'diabetes_high_risk_count': int(patient_metrics['total_patients'] * patient_metrics['diabetes_rate'] / 100),
                        'diabetes_high_risk_percentage': patient_metrics['diabetes_rate'],
                        'diabetes_target_mean': 125.5,
                        'cancer_patients_total': patient_metrics['total_patients'],
                        'cancer_malignant_count': int(patient_metrics['total_patients'] * 0.2),
                        'cancer_malignancy_rate': 20.0,
                        'expenditure_countries_total': department_metrics['total_departments'],
                        'expenditure_global_average': financial_metrics['revenue_per_patient'],
                        'expenditure_avg_growth_rate': 3.2,
                        'patient_satisfaction': clinical_metrics['average_satisfaction'],
                        'staff_satisfaction': staff_metrics['average_satisfaction'],
                        'treatment_success_rate': clinical_metrics['treatment_success_rate'],
                        'complication_rate': clinical_metrics['complication_rate']
                    }
                else:
                    # Mock KPIs
                    kpis = {
                        'system_datasets_loaded': 5,
                        'system_total_records': 15000,
                        'system_avg_data_quality': 92.5,
                        'diabetes_patients_total': 1000,
                        'diabetes_high_risk_count': 150,
                        'diabetes_high_risk_percentage': 15.0,
                        'diabetes_target_mean': 125.5,
                        'cancer_patients_total': 500,
                        'cancer_malignant_count': 200,
                        'cancer_malignancy_rate': 40.0,
                        'expenditure_countries_total': 25,
                        'expenditure_global_average': 4500,
                        'expenditure_avg_growth_rate': 3.2
                    }
                
                # System overview metrics
                st.subheader("System Overview")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Datasets", 
                        kpis.get('system_datasets_loaded', 0),
                        delta=f"+{kpis.get('system_datasets_loaded', 0)}"
                    )
                
                with col2:
                    total_records = kpis.get('system_total_records', 0)
                    st.metric(
                        "Total Records", 
                        f"{total_records:,}",
                        delta=f"+{total_records:,}"
                    )
                
                with col3:
                    avg_quality = kpis.get('system_avg_data_quality', 0)
                    st.metric(
                        "Data Quality", 
                        f"{avg_quality:.1f}%",
                        delta=f"+{avg_quality:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Last Updated", 
                        "Live",
                        delta="Active"
                    )
                
                with col5:
                    st.metric(
                        "Performance", 
                        "Optimal",
                        delta="Fast"
                    )
                
                st.markdown("---")
                
                # Enhanced KPI visualization with tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Healthcare", "Trends", "Detailed"])
                
                with tab1:
                    # System metrics visualization
                    st.subheader("System Performance Metrics")
                    
                    # Create system metrics chart
                    system_metrics = {
                        'Datasets': kpis.get('system_datasets_loaded', 0),
                        'Records': min(kpis.get('system_total_records', 0) / 1000, 100),  # Scale for visualization
                        'Quality': kpis.get('system_avg_data_quality', 0),
                        'Performance': 95  # Mock performance score
                    }
                    
                    fig_system = px.bar(
                        x=list(system_metrics.keys()),
                        y=list(system_metrics.values()),
                        title='System Performance Overview',
                        color=list(system_metrics.values()),
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_system, width="stretch")
                
                with tab2:
                    # Healthcare-specific KPIs
                    st.subheader("Healthcare Analytics")
                    
                    # Diabetes analysis
                    if 'diabetes_patients_total' in kpis:
                        st.write("**Diabetes Analysis**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Patients", f"{kpis['diabetes_patients_total']:,}")
                        
                        with col2:
                            high_risk_pct = kpis.get('diabetes_high_risk_percentage', 0)
                            st.metric("High Risk", f"{high_risk_pct:.1f}%")
                        
                        with col3:
                            avg_target = kpis.get('diabetes_target_mean', 0)
                            st.metric("Avg Target", f"{avg_target:.1f}")
                        
                        with col4:
                            st.metric("Status", "Active")
                        
                        # Diabetes risk distribution
                        diabetes_risk_data = {
                            'Risk Level': ['Low', 'Moderate', 'High'],
                            'Percentage': [
                                100 - high_risk_pct - (high_risk_pct * 0.3),  # Mock distribution
                                high_risk_pct * 0.3,
                                high_risk_pct
                            ]
                        }
                        
                        fig_diabetes = px.pie(
                            diabetes_risk_data,
                            values='Percentage',
                            names='Risk Level',
                            title='Diabetes Risk Distribution',
                            color_discrete_map={'Low': '#32CD32', 'Moderate': '#FFD700', 'High': '#DC143C'}
                        )
                        st.plotly_chart(fig_diabetes, width="stretch")
                    
                    # Cancer analysis
                    if 'cancer_patients_total' in kpis:
                        st.write("**Cancer Analysis**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Patients", f"{kpis['cancer_patients_total']:,}")
                        
                        with col2:
                            malignancy_rate = kpis.get('cancer_malignancy_rate', 0)
                            st.metric("Malignancy Rate", f"{malignancy_rate:.1f}%")
                        
                        with col3:
                            malignant_count = kpis.get('cancer_malignant_count', 0)
                            st.metric("Malignant Cases", f"{malignant_count:,}")
                        
                        with col4:
                            st.metric("Status", "Active")
                        
                        # Cancer diagnosis distribution
                        cancer_data = {
                            'Diagnosis': ['Benign', 'Malignant'],
                            'Count': [
                                kpis['cancer_patients_total'] - malignant_count,
                                malignant_count
                            ]
                        }
                        
                        fig_cancer = px.pie(
                            cancer_data,
                            values='Count',
                            names='Diagnosis',
                            title='Cancer Diagnosis Distribution',
                            color_discrete_map={'Benign': '#32CD32', 'Malignant': '#DC143C'}
                        )
                        st.plotly_chart(fig_cancer, width="stretch")
                
                with tab3:
                    st.subheader("Performance Trends")
                    
                    # Mock trend data
                    trend_data = {
                        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
                        'Data Quality': np.random.normal(85, 5, 30),
                        'System Performance': np.random.normal(95, 2, 30),
                        'User Satisfaction': np.random.normal(88, 3, 30)
                    }
                    
                    trend_df = pd.DataFrame(trend_data)
                    
                    fig_trends = px.line(
                        trend_df,
                        x='Date',
                        y=['Data Quality', 'System Performance', 'User Satisfaction'],
                        title='Performance Trends Over Time',
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig_trends, width="stretch")
                
                with tab4:
                    st.subheader("Detailed KPI Breakdown")
                    
                    # Create detailed KPI table
                    kpi_details = []
                    
                    for key, value in kpis.items():
                        if isinstance(value, (int, float)):
                            kpi_details.append({
                                'KPI': key.replace('_', ' ').title(),
                                'Value': value,
                                'Status': 'Good' if value > 0 else 'Needs Attention',
                                'Category': 'System' if 'system' in key else 'Healthcare'
                            })
                    
                    kpi_df = pd.DataFrame(kpi_details)
                    
                    if len(kpi_df) > 0:
                        safe_dataframe_display(kpi_df, width="stretch")
                        
                        # KPI status distribution
                        status_counts = kpi_df['Status'].value_counts()
                        fig_status = px.pie(
                            values=status_counts.values,
                            names=status_counts.index,
                            title='KPI Status Distribution'
                        )
                        st.plotly_chart(fig_status, width="stretch")
                
                # Report generation
                if st.session_state.get('generate_report', False):
                    st.subheader("KPI Report Generation")
                    
                    try:
                        if hasattr(st.session_state, 'kpi_dashboard') and st.session_state.kpi_dashboard:
                            report = st.session_state.kpi_dashboard.generate_kpi_report()
                        else:
                            # Mock report
                            report = {
                                'executive_summary': 'System performing optimally with high data quality and user satisfaction.',
                                'detailed_analysis': 'All KPIs are within acceptable ranges with room for improvement in cancer detection accuracy.',
                                'recommendations': [
                                    'Implement additional data validation checks',
                                    'Enhance cancer detection algorithms',
                                    'Improve user interface responsiveness'
                                ]
                            }
                        
                        # Display report in expandable sections
                        with st.expander("Executive Summary"):
                            st.write(report.get('executive_summary', 'No summary available'))
                        
                        with st.expander("Detailed Analysis"):
                            st.write(report.get('detailed_analysis', 'No detailed analysis available'))
                        
                        with st.expander("Recommendations"):
                            recommendations = report.get('recommendations', [])
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    st.write(f"{i}. {rec}")
                            else:
                                st.write("No recommendations available")
                        
                        # Download report
                        report_text = f"""
Healthcare DSS KPI Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Executive Summary:
{report.get('executive_summary', 'No summary available')}

Detailed Analysis:
{report.get('detailed_analysis', 'No detailed analysis available')}

Recommendations:
{chr(10).join([f"{i}. {rec}" for i, rec in enumerate(report.get('recommendations', []), 1)])}
                        """
                        
                        st.download_button(
                            label="Download Report",
                            data=report_text,
                            file_name=f"healthcare_dss_kpi_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                        st.session_state.generate_report = False
                        
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                
            except Exception as e:
                st.error(f"Error calculating KPIs: {e}")
                st.info("Please ensure the KPI dashboard is properly initialized.")
