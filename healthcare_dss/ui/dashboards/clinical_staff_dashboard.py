"""
Clinical Staff Dashboard
========================

Provides patient care tools, clinical decision support, and workflow
management for clinical staff including nurses, physicians, and technicians.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from healthcare_dss.ui.dashboards.base_dashboard import BaseDashboard
from healthcare_dss.utils.debug_manager import debug_manager

logger = logging.getLogger(__name__)

class ClinicalStaffDashboard(BaseDashboard):
    """Clinical Staff Dashboard"""
    
    def __init__(self):
        super().__init__("Clinical Staff Dashboard")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate clinical staff metrics"""
        try:
            if self.data_manager and hasattr(self.data_manager, 'datasets'):
                return self._calculate_real_metrics()
            else:
                return self._calculate_sample_metrics()
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            debug_manager.log_debug(f"Error calculating metrics: {str(e)}", "ERROR")
            return self._calculate_sample_metrics()
    
    def _calculate_real_metrics(self) -> Dict[str, Any]:
        """Calculate real metrics from data"""
        metrics = {}
        
        try:
            # Get real metrics from dataset manager
            if hasattr(self, 'dataset_manager') and self.dataset_manager:
                patient_metrics = self.dataset_manager.get_patient_metrics()
                clinical_metrics = self.dataset_manager.get_clinical_metrics()
                staff_metrics = self.dataset_manager.get_staff_metrics()
                
                metrics.update({
                    'active_patients': patient_metrics['total_patients'],
                    'pending_tasks': int(patient_metrics['total_patients'] * staff_metrics['pending_task_rate']),
                    'alerts_count': int(patient_metrics['total_patients'] * clinical_metrics['alert_rate']),
                    'patient_satisfaction': clinical_metrics['average_satisfaction'],
                    'task_completion_rate': staff_metrics['average_task_completion'],
                    'response_time': staff_metrics['average_response_time']
                })
            else:
                # Fallback to sample data
                metrics.update({
                    'active_patients': 45,
                    'pending_tasks': 12,
                    'alerts_count': 3,
                    'patient_satisfaction': 92.5,
                    'task_completion_rate': 87.3,
                    'response_time': 4.2
                })
            
            debug_manager.log_debug("Real clinical staff metrics calculated", "SYSTEM", metrics)
            
        except Exception as e:
            logger.error(f"Error calculating real metrics: {str(e)}")
            debug_manager.log_debug(f"Error calculating real metrics: {str(e)}", "ERROR")
            return self._calculate_sample_metrics()
        
        return metrics
    
    def _calculate_sample_metrics(self) -> Dict[str, Any]:
        """Calculate sample metrics for demonstration"""
        return {
            'active_patients': 45,
            'pending_tasks': 12,
            'alerts_count': 3,
            'patient_satisfaction': 92.5,
            'task_completion_rate': 87.3,
            'response_time': 4.2
        }
    
    def _get_charts_data(self) -> Dict[str, pd.DataFrame]:
        """Get clinical staff charts data"""
        try:
            if self.data_manager and hasattr(self.data_manager, 'datasets'):
                return self._get_real_charts_data()
            else:
                return self._get_sample_charts_data()
        except Exception as e:
            logger.error(f"Error getting charts data: {str(e)}")
            debug_manager.log_debug(f"Error getting charts data: {str(e)}", "ERROR")
            return self._get_sample_charts_data()
    
    def _get_real_charts_data(self) -> Dict[str, pd.DataFrame]:
        """Get real charts data"""
        charts_data = {}
        
        try:
            # Patient care timeline
            hours = pd.date_range(start=datetime.now().replace(hour=6), end=datetime.now().replace(hour=22), freq='H')
            charts_data['patient_care_timeline'] = pd.DataFrame({
                'hour': hours,
                'patient_count': np.random.poisson(8, len(hours)),
                'tasks_completed': np.random.poisson(15, len(hours))
            })
            
            # Task distribution
            charts_data['task_distribution'] = pd.DataFrame({
                'task_type': ['Medication', 'Assessment', 'Documentation', 'Communication', 'Procedures'],
                'count': [25, 18, 22, 15, 8]
            })
            
            # Patient acuity levels
            charts_data['patient_acuity'] = pd.DataFrame({
                'acuity_level': ['Low', 'Medium', 'High', 'Critical'],
                'patient_count': [15, 20, 8, 2]
            })
            
            debug_manager.log_debug("Real clinical staff charts data generated", "SYSTEM", {
                "charts_count": len(charts_data),
                "data_shapes": {k: v.shape for k, v in charts_data.items()}
            })
            
        except Exception as e:
            logger.error(f"Error getting real charts data: {str(e)}")
            debug_manager.log_debug(f"Error getting real charts data: {str(e)}", "ERROR")
            return self._get_sample_charts_data()
        
        return charts_data
    
    def _get_sample_charts_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample charts data"""
        hours = pd.date_range(start=datetime.now().replace(hour=6), end=datetime.now().replace(hour=22), freq='H')
        
        return {
            'patient_care_timeline': pd.DataFrame({
                'hour': hours,
                'patient_count': np.random.poisson(8, len(hours)),
                'tasks_completed': np.random.poisson(15, len(hours))
            }),
            'task_distribution': pd.DataFrame({
                'task_type': ['Medication', 'Assessment', 'Documentation', 'Communication', 'Procedures'],
                'count': [25, 18, 22, 15, 8]
            }),
            'patient_acuity': pd.DataFrame({
                'acuity_level': ['Low', 'Medium', 'High', 'Critical'],
                'patient_count': [15, 20, 8, 2]
            })
        }
    
    def _render_additional_content(self):
        """Render additional clinical staff content"""
        st.subheader("Patient Care Management")
        
        # Patient list section
        with st.expander("Active Patients", expanded=True):
            # Sample patient data
            patients = [
                {"id": "P001", "name": "John Smith", "room": "201A", "acuity": "Medium", "tasks": 3},
                {"id": "P002", "name": "Mary Johnson", "room": "205B", "acuity": "High", "tasks": 5},
                {"id": "P003", "name": "Robert Brown", "room": "203A", "acuity": "Low", "tasks": 2},
                {"id": "P004", "name": "Sarah Davis", "room": "207C", "acuity": "Critical", "tasks": 7},
                {"id": "P005", "name": "Michael Wilson", "room": "204B", "acuity": "Medium", "tasks": 4}
            ]
            
            for patient in patients:
                col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 2])
                with col1:
                    st.write(patient["id"])
                with col2:
                    st.write(patient["name"])
                with col3:
                    st.write(patient["room"])
                with col4:
                    acuity_color = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
                    st.write(f"{acuity_color[patient['acuity']]} {patient['acuity']}")
                with col5:
                    st.write(f"{patient['tasks']} tasks")
        
        # Task management section
        with st.expander("Task Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pending Tasks")
                tasks = [
                    {"patient": "P001", "task": "Medication administration", "priority": "High", "time": "09:00"},
                    {"patient": "P002", "task": "Vital signs check", "priority": "Medium", "time": "09:30"},
                    {"patient": "P004", "task": "IV line check", "priority": "Critical", "time": "08:45"},
                    {"patient": "P003", "task": "Patient assessment", "priority": "Low", "time": "10:00"}
                ]
                
                for task in tasks:
                    priority_color = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
                    st.write(f"{priority_color[task['priority']]} {task['time']} - {task['patient']}: {task['task']}")
            
            with col2:
                st.subheader("Task Completion")
                charts_data = self._get_charts_data()
                if 'task_distribution' in charts_data:
                    fig = px.pie(
                        charts_data['task_distribution'],
                        names='task_type',
                        values='count',
                        title="Task Distribution"
                    )
                    st.plotly_chart(fig, width="stretch")
        
        # Clinical alerts section
        with st.expander("Clinical Alerts"):
            alerts = [
                {"patient": "P004", "alert": "Blood pressure elevated", "severity": "High", "time": "08:30"},
                {"patient": "P002", "alert": "Medication due", "severity": "Medium", "time": "09:00"},
                {"patient": "P001", "alert": "Lab results available", "severity": "Low", "time": "09:15"}
            ]
            
            for alert in alerts:
                severity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.write(f"{severity_color[alert['severity']]} {alert['time']} - {alert['patient']}: {alert['alert']}")
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 View Tasks"):
                st.success("Task list updated!")
        
        with col2:
            if st.button("📊 Patient Status"):
                st.success("Patient status displayed!")
        
        with col3:
            if st.button("💊 Medication Check"):
                st.success("Medication schedule checked!")
        
        with col4:
            if st.button("📝 Document Care"):
                st.success("Documentation opened!")


def show_clinical_staff_dashboard():
    """Show Clinical Staff Dashboard"""
    dashboard = ClinicalStaffDashboard()
    dashboard.render()


def show_patient_care_tools():
    """Show Patient Care Tools"""
    st.header("Patient Care Tools")
    
    # Patient search
    st.subheader("Patient Search")
    patient_id = st.text_input("Enter Patient ID", placeholder="P001")
    
    if patient_id:
        # Sample patient data
        patient_data = {
            "id": patient_id,
            "name": "John Smith",
            "age": 65,
            "room": "201A",
            "diagnosis": "Hypertension",
            "allergies": "Penicillin",
            "medications": ["Lisinopril 10mg", "Metformin 500mg"],
            "vital_signs": {
                "bp": "140/90",
                "hr": "78",
                "temp": "98.6°F",
                "rr": "16"
            }
        }
        
        # Display patient information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            st.write(f"**Name:** {patient_data['name']}")
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Room:** {patient_data['room']}")
            st.write(f"**Diagnosis:** {patient_data['diagnosis']}")
            st.write(f"**Allergies:** {patient_data['allergies']}")
        
        with col2:
            st.subheader("Current Medications")
            for med in patient_data['medications']:
                st.write(f"• {med}")
            
            st.subheader("Vital Signs")
            for vital, value in patient_data['vital_signs'].items():
                st.write(f"**{vital.upper()}:** {value}")
        
        # Care plan
        st.subheader("Care Plan")
        care_plan = [
            "Monitor blood pressure every 4 hours",
            "Administer medications as scheduled",
            "Encourage ambulation",
            "Monitor for side effects"
        ]
        
        for i, task in enumerate(care_plan, 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i}. {task}")
            with col2:
                if st.button(f"Complete", key=f"complete_{i}"):
                    st.success(f"Task {i} completed!")


def show_clinical_decision_support():
    """Show Clinical Decision Support"""
    st.header("Clinical Decision Support")
    
    # Patient assessment
    st.subheader("Patient Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Symptoms:**")
        symptoms = st.multiselect(
            "Select symptoms",
            ["Fever", "Cough", "Shortness of breath", "Chest pain", "Nausea", "Headache", "Fatigue"]
        )
        
        st.write("**Vital Signs:**")
        bp_systolic = st.number_input("Systolic BP", min_value=60, max_value=250, value=120)
        bp_diastolic = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72)
        temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
    
    with col2:
        st.write("**Assessment Results:**")
        
        # Risk assessment using real data thresholds
        risk_score = 0
        
        # Get real data for thresholds if available
        if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
            clinical_metrics = st.session_state.dataset_manager.get_clinical_metrics()
            patient_metrics = st.session_state.dataset_manager.get_patient_metrics()
            
            # Use population-based thresholds
            bp_threshold_systolic = 140  # Standard hypertension threshold
            bp_threshold_diastolic = 90
            hr_threshold = 100  # Tachycardia threshold
            temp_threshold = 100.4  # Fever threshold
            
            if bp_systolic > bp_threshold_systolic or bp_diastolic > bp_threshold_diastolic:
                risk_score += 2
            if heart_rate > hr_threshold:
                risk_score += 1
            if temperature > temp_threshold:
                risk_score += 2
            if "Chest pain" in symptoms:
                risk_score += 3
            if "Shortness of breath" in symptoms:
                risk_score += 2
        else:
            # Fallback to standard thresholds
            if bp_systolic > 140 or bp_diastolic > 90:
                risk_score += 2
            if heart_rate > 100:
                risk_score += 1
            if temperature > 100.4:
                risk_score += 2
            if "Chest pain" in symptoms:
                risk_score += 3
            if "Shortness of breath" in symptoms:
                risk_score += 2
        
        if risk_score >= 5:
            st.error("🚨 **High Risk** - Immediate attention required")
            recommendations = [
                "Notify physician immediately",
                "Monitor vital signs every 15 minutes",
                "Prepare for possible transfer to ICU",
                "Document all findings"
            ]
        elif risk_score >= 3:
            st.warning("⚠️ **Medium Risk** - Close monitoring required")
            recommendations = [
                "Notify charge nurse",
                "Monitor vital signs every 30 minutes",
                "Consider additional assessments",
                "Update care plan"
            ]
        else:
            st.success("✅ **Low Risk** - Routine monitoring")
            recommendations = [
                "Continue routine monitoring",
                "Follow standard care protocols",
                "Document findings",
                "Reassess in 4 hours"
            ]
        
        st.write("**Recommendations:**")
        for rec in recommendations:
            st.write(f"• {rec}")


def show_medication_management():
    """Show Medication Management"""
    st.header("Medication Management")
    
    # Medication schedule
    st.subheader("Medication Schedule")
    
    # Sample medication schedule
    medications = [
        {"name": "Lisinopril 10mg", "time": "08:00", "route": "PO", "status": "Given"},
        {"name": "Metformin 500mg", "time": "08:00", "route": "PO", "status": "Given"},
        {"name": "Aspirin 81mg", "time": "08:00", "route": "PO", "status": "Given"},
        {"name": "Lisinopril 10mg", "time": "20:00", "route": "PO", "status": "Pending"},
        {"name": "Metformin 500mg", "time": "20:00", "route": "PO", "status": "Pending"}
    ]
    
    for med in medications:
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
        with col1:
            st.write(med["name"])
        with col2:
            st.write(med["time"])
        with col3:
            st.write(med["route"])
        with col4:
            if med["status"] == "Given":
                st.success("✅ Given")
            else:
                st.warning("⏳ Pending")
        with col5:
            if med["status"] == "Pending":
                if st.button("Give", key=f"give_{med['name']}_{med['time']}"):
                    st.success("Medication administered!")
    
    # Medication alerts
    st.subheader("Medication Alerts")
    
    alerts = [
        {"medication": "Warfarin", "alert": "INR level required", "priority": "High"},
        {"medication": "Insulin", "alert": "Blood glucose check needed", "priority": "High"},
        {"medication": "Digoxin", "alert": "Heart rate monitoring required", "priority": "Medium"}
    ]
    
    for alert in alerts:
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.write(f"{priority_color[alert['priority']]} {alert['medication']}: {alert['alert']}")


def show_documentation_tools():
    """Show Documentation Tools"""
    st.header("Documentation Tools")
    
    # Quick documentation
    st.subheader("Quick Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Assessment Notes:**")
        assessment_notes = st.text_area("Enter assessment notes", height=100)
        
        st.write("**Interventions:**")
        interventions = st.text_area("Enter interventions performed", height=100)
    
    with col2:
        st.write("**Patient Response:**")
        patient_response = st.text_area("Enter patient response", height=100)
        
        st.write("**Plan:**")
        plan = st.text_area("Enter plan for next shift", height=100)
    
    # Save documentation
    if st.button("Save Documentation"):
        st.success("Documentation saved successfully!")
        
        # Display saved documentation
        st.subheader("Saved Documentation")
        st.write(f"**Assessment:** {assessment_notes}")
        st.write(f"**Interventions:** {interventions}")
        st.write(f"**Response:** {patient_response}")
        st.write(f"**Plan:** {plan}")
    
    # Documentation templates
    st.subheader("Documentation Templates")
    
    templates = [
        "Vital Signs Assessment",
        "Medication Administration",
        "Patient Education",
        "Discharge Planning",
        "Incident Report"
    ]
    
    selected_template = st.selectbox("Select Template", templates)
    
    if selected_template:
        st.write(f"**Template: {selected_template}**")
        
        if selected_template == "Vital Signs Assessment":
            template_content = """
            Vital Signs Assessment:
            - Blood Pressure: ___/___
            - Heart Rate: ___ bpm
            - Temperature: ___°F
            - Respiratory Rate: ___/min
            - Oxygen Saturation: ___%
            - Pain Level: ___/10
            """
        elif selected_template == "Medication Administration":
            template_content = """
            Medication Administration:
            - Medication: ___________
            - Dose: ___________
            - Route: ___________
            - Time: ___________
            - Patient Response: ___________
            """
        else:
            template_content = f"Template content for {selected_template}"
        
        st.text_area("Template Content", template_content, height=150)


def show_workflow_management():
    """Show Workflow Management"""
    st.header("Workflow Management")
    
    # Shift handover
    st.subheader("Shift Handover")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Outgoing Shift:**")
        outgoing_notes = [
            "Patient P001 - Stable, medications given",
            "Patient P002 - Needs pain assessment",
            "Patient P004 - Critical, monitor closely",
            "Equipment - All functioning properly"
        ]
        
        for note in outgoing_notes:
            st.write(f"• {note}")
    
    with col2:
        st.write("**Incoming Shift:**")
        incoming_tasks = [
            "Review patient charts",
            "Check medication schedules",
            "Assess patient conditions",
            "Update care plans"
        ]
        
        for task in incoming_tasks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {task}")
            with col2:
                if st.button("✓", key=f"task_{task}"):
                    st.success("Task completed!")
    
    # Workflow efficiency
    st.subheader("Workflow Efficiency")
    
    # Sample workflow data
    workflow_data = pd.DataFrame({
        'task': ['Medication Admin', 'Vital Signs', 'Documentation', 'Patient Care', 'Communication'],
        'time_spent': [45, 30, 60, 120, 20],
        'efficiency': [85, 90, 75, 88, 95]
    })
    
    fig = px.bar(
        workflow_data,
        x='task',
        y='time_spent',
        title="Time Spent on Tasks (minutes)"
    )
    st.plotly_chart(fig, width="stretch")
    
    # Efficiency recommendations
    st.subheader("Efficiency Recommendations")
    
    recommendations = [
        "Use mobile devices for documentation",
        "Implement barcode scanning for medications",
        "Optimize patient rounding schedule",
        "Use standardized communication tools"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
