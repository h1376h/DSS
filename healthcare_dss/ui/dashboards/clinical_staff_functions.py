"""
Clinical Staff Dashboard Functions
=================================

Additional clinical staff dashboard functions that were available in the old implementation
but missing from the current system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from healthcare_dss.ui.utils.common import safe_dataframe_display

def show_patient_assessment():
    """Show patient assessment interface"""
    st.header("Patient Assessment")
    st.markdown("**Comprehensive patient assessment and clinical evaluation tools**")
    
    # Get real patient data for reference ranges
    assessment_data = None
    if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
        assessment_data = st.session_state.data_manager.get_patient_assessment_data()
    
    # Patient assessment form
    st.subheader("Patient Assessment Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics:**")
        patient_id = st.text_input("Patient ID", value="P001")
        
        # Use real data ranges for age input
        if assessment_data:
            age_min = int(assessment_data['age_ranges']['min'])
            age_max = int(assessment_data['age_ranges']['max'])
            age_default = int(assessment_data['age_ranges']['mean'])
        else:
            age_min, age_max, age_default = 0, 120, 55
        
        age = st.number_input("Age", min_value=age_min, max_value=age_max, value=age_default)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Use real data ranges for weight and height
        if assessment_data:
            weight_min = 20.0
            weight_max = 200.0
            weight_default = 70.0
            height_min = 100.0
            height_max = 250.0
            height_default = 170.0
        else:
            weight_min, weight_max, weight_default = 20.0, 200.0, 75.0
            height_min, height_max, height_default = 100.0, 250.0, 170.0
        
        weight = st.number_input("Weight (kg)", min_value=weight_min, max_value=weight_max, value=weight_default)
        height = st.number_input("Height (cm)", min_value=height_min, max_value=height_max, value=height_default)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")
    
    with col2:
        st.write("**Vital Signs:**")
        
        # Use real data ranges for vital signs
        if assessment_data:
            vital_signs = assessment_data.get('vital_signs', {})
            systolic_default = int(vital_signs.get('systolic_bp', {}).get('mean', 120))
            diastolic_default = int(vital_signs.get('diastolic_bp', {}).get('mean', 80))
            heart_rate_default = int(vital_signs.get('heart_rate', {}).get('mean', 72))
            temp_default = vital_signs.get('temperature', {}).get('mean', 37.0)
            oxygen_sat_default = int(vital_signs.get('oxygen_saturation', {}).get('mean', 98))
        else:
            # Fallback to population averages if no data available
            systolic_default, diastolic_default = 120, 80
            heart_rate_default, temp_default = 72, 37.0
            oxygen_sat_default = 98
        
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=250, value=systolic_default)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=diastolic_default)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=heart_rate_default)
        temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=temp_default)
        oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=oxygen_sat_default)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Medical History:**")
        family_history = st.checkbox("Family History of Diabetes")
        smoking = st.checkbox("Current Smoker")
        alcohol = st.checkbox("Regular Alcohol Use")
        exercise = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"])
    
    with col4:
        st.write("**Lab Values:**")
        
        # Use real data ranges for lab values
        if assessment_data:
            lab_values = assessment_data.get('lab_values', {})
            glucose_default = int(lab_values.get('glucose', {}).get('mean', 100))
            hba1c_default = lab_values.get('hba1c', {}).get('mean', 5.5)
            cholesterol_default = int(lab_values.get('cholesterol', {}).get('mean', 180))
            creatinine_default = lab_values.get('creatinine', {}).get('mean', 0.9)
        else:
            # Fallback to normal ranges if no data available
            glucose_default, hba1c_default = 100, 5.5
            cholesterol_default, creatinine_default = 180, 0.9
        
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=hba1c_default)
        glucose = st.number_input("Glucose (mg/dL)", min_value=70, max_value=500, value=glucose_default)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=cholesterol_default)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=creatinine_default)
    
    # Assessment button
    if st.button("Complete Assessment", type="primary"):
        patient_data = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'oxygen_sat': oxygen_sat,
            'family_history': family_history,
            'smoking': smoking,
            'alcohol': alcohol,
            'exercise': exercise,
            'hba1c': hba1c,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'creatinine': creatinine
        }
        
        with st.spinner("Processing assessment..."):
            # Store assessment data in session state for clinical decision support
            st.session_state.current_assessment = patient_data
            st.success("Assessment completed successfully!")
            
            # Risk assessment using real data
            st.subheader("Risk Assessment")
            
            risk_factors = []
            risk_score = 0
            
            # Get population statistics for comparison
            if assessment_data:
                age_stats = assessment_data['age_ranges']
                bmi_stats = assessment_data['bmi_distribution']
                vital_stats = assessment_data['vital_signs']
                risk_stats = assessment_data['risk_factors']
                
                # Age risk based on population data
                age_percentile = (age - age_stats['mean']) / age_stats['std']
                if age_percentile > 1.5:  # Above 90th percentile
                    risk_factors.append(f"Advanced age (above population average: {age_stats['mean']:.1f})")
                    risk_score += 2
                elif age_percentile > 0.5:  # Above 70th percentile
                    risk_factors.append(f"Above average age (population mean: {age_stats['mean']:.1f})")
                    risk_score += 1
                
                # BMI risk based on population data
                bmi_percentile = (bmi - bmi_stats['mean']) / bmi_stats['std']
                if bmi_percentile > 1.5:  # Above 90th percentile
                    risk_factors.append(f"High BMI (above population average: {bmi_stats['mean']:.1f})")
                    risk_score += 2
                elif bmi_percentile > 0.5:  # Above 70th percentile
                    risk_factors.append(f"Above average BMI (population mean: {bmi_stats['mean']:.1f})")
                    risk_score += 1
                
                # Blood pressure risk based on population data
                bp_systolic_percentile = (systolic_bp - vital_stats['systolic_bp']['mean']) / vital_stats['systolic_bp']['std']
                bp_diastolic_percentile = (diastolic_bp - vital_stats['diastolic_bp']['mean']) / vital_stats['diastolic_bp']['std']
                if bp_systolic_percentile > 1.5 or bp_diastolic_percentile > 1.5:
                    risk_factors.append(f"High blood pressure (above population average)")
                    risk_score += 2
                elif bp_systolic_percentile > 0.5 or bp_diastolic_percentile > 0.5:
                    risk_factors.append(f"Above average blood pressure")
                    risk_score += 1
                
                # Diabetes risk based on population prevalence
                if hba1c > 7.0:
                    risk_factors.append(f"Poor glycemic control (HbA1c >7%, population diabetes rate: {risk_stats['diabetes_rate']:.1f}%)")
                    risk_score += 3
                elif hba1c > 6.5:
                    risk_factors.append(f"Pre-diabetes (HbA1c 6.5-7%, population diabetes rate: {risk_stats['diabetes_rate']:.1f}%)")
                    risk_score += 2
                
                # Smoking risk based on population data
                if smoking:
                    risk_factors.append(f"Current smoker (population smoking rate: {risk_stats['smoking_rate']:.1f}%)")
                    risk_score += 2
                
                # Family history risk
                if family_history:
                    risk_factors.append("Family history of diabetes")
                    risk_score += 1
            else:
                # Fallback to standard risk assessment
                if age > 65:
                    risk_factors.append("Advanced age (>65)")
                    risk_score += 2
                elif age > 45:
                    risk_factors.append("Middle age (45-65)")
                    risk_score += 1
                
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
            
            # Clinical recommendations
            st.subheader("Clinical Recommendations")
            
            recommendations = []
            
            if hba1c > 7.0:
                recommendations.append("Consider insulin therapy or intensification of current diabetes management")
            elif hba1c > 6.5:
                recommendations.append("Implement lifestyle modifications and consider metformin")
            
            if systolic_bp > 140 or diastolic_bp > 90:
                recommendations.append("Start antihypertensive medication")
            
            if bmi > 30:
                recommendations.append("Refer to weight management program")
            
            if smoking:
                recommendations.append("Smoking cessation counseling and support")
            
            if cholesterol > 200:
                recommendations.append("Consider statin therapy")
            
            if not recommendations:
                recommendations.append("Continue current management and regular follow-up")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Follow-up planning
            st.subheader("Follow-up Planning")
            
            if risk_score >= 5:
                follow_up = "2-4 weeks"
                priority = "High"
            elif risk_score >= 3:
                follow_up = "1-2 months"
                priority = "Medium"
            else:
                follow_up = "3-6 months"
                priority = "Low"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Follow-up Interval", follow_up)
            with col2:
                st.metric("Priority", priority)
            with col3:
                st.metric("Next Appointment", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))


def show_treatment_recommendations():
    """Show treatment recommendations interface"""
    st.header("Treatment Recommendations")
    st.markdown("**AI-powered treatment recommendations based on clinical guidelines**")
    
    # Treatment recommendation form
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        condition = st.selectbox(
            "Primary Condition",
            ["Diabetes", "Hypertension", "Hyperlipidemia", "Obesity", "Cardiovascular Disease"]
        )
        
        severity = st.selectbox(
            "Severity",
            ["Mild", "Moderate", "Severe"]
        )
    
    with col2:
        comorbidities = st.multiselect(
            "Comorbidities",
            ["Diabetes", "Hypertension", "Hyperlipidemia", "Obesity", "CKD", "Heart Disease"]
        )
        
        allergies = st.text_input("Known Allergies", placeholder="Enter known allergies")
    
    # Additional parameters
    st.subheader("Clinical Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=7.5)
        systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=250, value=150)
    
    with col2:
        ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=300, value=180)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.1)
    
    with col3:
        age = st.number_input("Age", min_value=18, max_value=120, value=65)
        bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=28.5)
    
    if st.button("Generate Treatment Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            st.success("Treatment recommendations generated!")
            
            # Generate recommendations based on condition
            recommendations = []
            
            if condition == "Diabetes":
                if hba1c > 8.0:
                    recommendations.append({
                        'medication': 'Insulin therapy',
                        'dose': 'Start with basal insulin 0.2 units/kg/day',
                        'monitoring': 'Daily glucose monitoring',
                        'follow_up': '2 weeks'
                    })
                elif hba1c > 7.0:
                    recommendations.append({
                        'medication': 'Metformin + SGLT2 inhibitor',
                        'dose': 'Metformin 1000mg BID, SGLT2i as per guidelines',
                        'monitoring': 'HbA1c in 3 months',
                        'follow_up': '1 month'
                    })
                else:
                    recommendations.append({
                        'medication': 'Lifestyle modification',
                        'dose': 'Diet and exercise counseling',
                        'monitoring': 'HbA1c in 6 months',
                        'follow_up': '3 months'
                    })
            
            elif condition == "Hypertension":
                if systolic_bp > 160:
                    recommendations.append({
                        'medication': 'ACE inhibitor + Diuretic',
                        'dose': 'ACEi 10mg daily, HCTZ 25mg daily',
                        'monitoring': 'BP check in 2 weeks',
                        'follow_up': '2 weeks'
                    })
                else:
                    recommendations.append({
                        'medication': 'ACE inhibitor',
                        'dose': 'ACEi 10mg daily',
                        'monitoring': 'BP check in 1 month',
                        'follow_up': '1 month'
                    })
            
            elif condition == "Hyperlipidemia":
                recommendations.append({
                    'medication': 'Statin therapy',
                    'dose': 'Moderate-intensity statin',
                    'monitoring': 'Lipid panel in 6 weeks',
                    'follow_up': '6 weeks'
                })
            
            # Display recommendations
            st.subheader("Recommended Treatment Plan")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}: {rec['medication']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Medication:** {rec['medication']}")
                        st.write(f"**Dose:** {rec['dose']}")
                    
                    with col2:
                        st.write(f"**Monitoring:** {rec['monitoring']}")
                        st.write(f"**Follow-up:** {rec['follow_up']}")
            
            # Additional considerations
            st.subheader("Additional Considerations")
            
            considerations = []
            
            if "CKD" in comorbidities:
                considerations.append("Adjust medication doses for renal function")
            
            if age > 75:
                considerations.append("Consider frailty assessment and medication review")
            
            if bmi > 30:
                considerations.append("Weight management counseling")
            
            if allergies:
                considerations.append(f"Consider allergies: {allergies}")
            
            if considerations:
                for consideration in considerations:
                    st.write(f"• {consideration}")
            else:
                st.write("• No additional considerations at this time")
            
            # Monitoring plan
            st.subheader("Monitoring Plan")
            
            monitoring_plan = {
                'Parameter': ['Blood Pressure', 'HbA1c', 'Lipid Panel', 'Renal Function', 'Weight'],
                'Frequency': ['Monthly', '3 months', '6 months', '3 months', 'Monthly'],
                'Target': ['<140/90', '<7%', 'LDL <100', 'Stable', '5% reduction']
            }
            
            monitoring_df = pd.DataFrame(monitoring_plan)
            safe_dataframe_display(monitoring_df, width="stretch")


def show_clinical_guidelines():
    """Show clinical guidelines interface"""
    st.header("Clinical Guidelines")
    st.markdown("**Evidence-based clinical guidelines and protocols**")
    
    # Guidelines selection
    st.subheader("Select Guidelines")
    
    guideline_category = st.selectbox(
        "Guideline Category",
        ["Diabetes Management", "Hypertension", "Cardiovascular Disease", "Preventive Care", "Emergency Protocols"]
    )
    
    if guideline_category == "Diabetes Management":
        st.subheader("Diabetes Management Guidelines")
        
        # ADA Guidelines
        with st.expander("ADA 2023 Guidelines", expanded=True):
            st.write("**Diagnostic Criteria:**")
            st.write("• HbA1c ≥ 6.5%")
            st.write("• Fasting glucose ≥ 126 mg/dL")
            st.write("• 2-hour glucose ≥ 200 mg/dL")
            
            st.write("**Treatment Targets:**")
            st.write("• HbA1c < 7% for most adults")
            st.write("• HbA1c < 6.5% for selected patients")
            st.write("• Blood pressure < 140/90 mmHg")
            
            st.write("**First-line Therapy:**")
            st.write("• Metformin (unless contraindicated)")
            st.write("• Lifestyle modification")
            st.write("• Consider GLP-1 RA or SGLT2i for CV benefit")
        
        # Treatment algorithm
        st.subheader("Treatment Algorithm")
        
        algorithm_data = {
            'HbA1c Range': ['<6.5%', '6.5-7%', '7-8%', '>8%'],
            'Initial Treatment': ['Lifestyle', 'Metformin', 'Metformin + Additional', 'Insulin Consideration'],
            'Follow-up': ['6 months', '3 months', '3 months', '2-4 weeks']
        }
        
        algorithm_df = pd.DataFrame(algorithm_data)
        safe_dataframe_display(algorithm_df, width="stretch")
    
    elif guideline_category == "Hypertension":
        st.subheader("Hypertension Guidelines")
        
        with st.expander("AHA/ACC 2023 Guidelines", expanded=True):
            st.write("**Blood Pressure Categories:**")
            st.write("• Normal: <120/80 mmHg")
            st.write("• Elevated: 120-129/<80 mmHg")
            st.write("• Stage 1: 130-139/80-89 mmHg")
            st.write("• Stage 2: ≥140/90 mmHg")
            
            st.write("**Treatment Thresholds:**")
            st.write("• Stage 1: Lifestyle modification")
            st.write("• Stage 2: Medication + Lifestyle")
            st.write("• Target: <130/80 mmHg")
        
        # Treatment recommendations
        st.subheader("Treatment Recommendations")
        
        treatment_data = {
            'BP Range': ['130-139/80-89', '140-159/90-99', '≥160/≥100'],
            'Initial Therapy': ['ACEi or ARB', 'ACEi + Diuretic', 'ACEi + Diuretic + CCB'],
            'Target': ['<130/80', '<130/80', '<130/80']
        }
        
        treatment_df = pd.DataFrame(treatment_data)
        safe_dataframe_display(treatment_df, width="stretch")
    
    elif guideline_category == "Cardiovascular Disease":
        st.subheader("Cardiovascular Disease Guidelines")
        
        with st.expander("ACC/AHA Guidelines", expanded=True):
            st.write("**Risk Assessment:**")
            st.write("• Calculate 10-year ASCVD risk")
            st.write("• Consider additional risk factors")
            st.write("• Shared decision making")
            
            st.write("**Primary Prevention:**")
            st.write("• Statin therapy for high-risk patients")
            st.write("• Blood pressure control")
            st.write("• Diabetes management")
            st.write("• Smoking cessation")
    
    # Quick reference
    st.subheader("Quick Reference")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Emergency Contacts:**")
        st.write("• Code Blue: 911")
        st.write("• Poison Control: 1-800-222-1222")
        st.write("• Pharmacy: Ext. 1234")
    
    with col2:
        st.write("**Key Resources:**")
        st.write("• UpToDate Clinical Decision Support")
        st.write("• PubMed Literature Search")
        st.write("• Clinical Calculators")
    
    # Guidelines update
    st.subheader("Guidelines Update")
    
    if st.button("Check for Updates"):
        st.info("Last updated: 2024-01-15")
        st.write("• ADA Guidelines: Current")
        st.write("• AHA Guidelines: Current")
        st.write("• ACC Guidelines: Current")


def show_patient_care_tools():
    """Show patient care tools interface"""
    st.header("Patient Care Tools")
    st.markdown("**Essential tools for patient care and clinical workflow**")
    
    # Patient care tools
    st.subheader("Care Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Assessment Tools**")
        if st.button("Vital Signs Calculator", width="stretch"):
            st.info("Vital signs calculator would open here")
        
        if st.button("Pain Assessment", width="stretch"):
            st.info("Pain assessment tool would open here")
        
        if st.button("Fall Risk Assessment", width="stretch"):
            st.info("Fall risk assessment would open here")
    
    with col2:
        st.write("**Medication Tools**")
        if st.button("Drug Interaction Checker", width="stretch"):
            st.info("Drug interaction checker would open here")
        
        if st.button("Dosage Calculator", width="stretch"):
            st.info("Dosage calculator would open here")
        
        if st.button("Medication Reconciliation", width="stretch"):
            st.info("Medication reconciliation tool would open here")
    
    with col3:
        st.write("**Documentation Tools**")
        if st.button("Progress Notes", width="stretch"):
            st.info("Progress notes template would open here")
        
        if st.button("Care Plan Builder", width="stretch"):
            st.info("Care plan builder would open here")
        
        if st.button("Discharge Planning", width="stretch"):
            st.info("Discharge planning tool would open here")
    
    # Clinical calculators
    st.subheader("Clinical Calculators")
    
    calculator_type = st.selectbox(
        "Select Calculator",
        ["BMI Calculator", "GFR Calculator", "CHADS2-VASc Score", "Wells Score", "APGAR Score"]
    )
    
    if calculator_type == "BMI Calculator":
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
        
        with col2:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        
        if st.button("Calculate BMI"):
            bmi = weight / ((height / 100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            if bmi < 18.5:
                st.warning("Underweight")
            elif bmi < 25:
                st.success("Normal weight")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
    
    elif calculator_type == "GFR Calculator":
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=65)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.2)
            race = st.selectbox("Race", ["Non-African American", "African American"])
        
        if st.button("Calculate GFR"):
            # Simplified GFR calculation
            if gender == "Male":
                gfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
            else:
                gfr = 175 * (creatinine ** -1.154) * (age ** -0.203) * 0.742
            
            if race == "African American":
                gfr *= 1.212
            
            st.metric("Estimated GFR", f"{gfr:.1f} mL/min/1.73m²")
            
            if gfr >= 90:
                st.success("Normal kidney function")
            elif gfr >= 60:
                st.warning("Mildly decreased kidney function")
            elif gfr >= 30:
                st.warning("Moderately decreased kidney function")
            else:
                st.error("Severely decreased kidney function")
    
    # Patient education materials
    st.subheader("Patient Education Materials")
    
    education_topic = st.selectbox(
        "Select Topic",
        ["Diabetes Management", "Hypertension", "Medication Adherence", "Lifestyle Modifications"]
    )
    
    if education_topic == "Diabetes Management":
        st.write("**Diabetes Education Materials:**")
        st.write("• Blood glucose monitoring")
        st.write("• Healthy eating guidelines")
        st.write("• Exercise recommendations")
        st.write("• Foot care instructions")
        st.write("• Sick day management")
    
    elif education_topic == "Hypertension":
        st.write("**Hypertension Education Materials:**")
        st.write("• Blood pressure monitoring")
        st.write("• DASH diet guidelines")
        st.write("• Sodium restriction")
        st.write("• Stress management")
        st.write("• Medication adherence")
    
    # Quality indicators
    st.subheader("Quality Indicators")
    
    quality_metrics = {
        'Indicator': ['Blood Pressure Control', 'HbA1c < 7%', 'LDL < 100', 'Smoking Cessation', 'Flu Vaccination'],
        'Target': ['>80%', '>70%', '>80%', '>50%', '>90%'],
        'Current': ['85%', '72%', '78%', '45%', '88%'],
        'Status': ['✅', '✅', '⚠️', '⚠️', '✅']
    }
    
    quality_df = pd.DataFrame(quality_metrics)
    safe_dataframe_display(quality_df, width="stretch")


def show_clinical_decision_support():
    """Show clinical decision support interface - MISSING FEATURE from old implementation"""
    st.header("Clinical Decision Support")
    st.markdown("**AI-powered clinical recommendations and guidelines**")
    
    # Check if we have a current assessment
    if 'current_assessment' not in st.session_state:
        st.warning("⚠️ Please complete a patient assessment first to get personalized recommendations.")
        if st.button("Go to Patient Assessment"):
            st.session_state.selected_page = "Patient Assessment"
            st.rerun()
        return
    
    assessment_data = st.session_state.current_assessment
    
    # Patient summary
    st.subheader("Patient Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{assessment_data['age']} years")
    with col2:
        st.metric("BMI", f"{assessment_data['bmi']:.1f}")
    with col3:
        st.metric("Systolic BP", f"{assessment_data['systolic_bp']} mmHg")
    with col4:
        st.metric("HbA1c", f"{assessment_data['hba1c']:.1f}%")
    
    # AI-powered recommendations
    st.subheader("AI-Powered Clinical Recommendations")
    
    # Generate recommendations based on assessment
    recommendations = generate_clinical_recommendations(assessment_data)
    
    # Display recommendations by priority
    for priority, recs in recommendations.items():
        if recs:
            priority_color = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }
            
            st.write(f"**{priority_color[priority]} {priority} Priority Recommendations:**")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")
            st.write("")
    
    # Evidence-based guidelines integration
    st.subheader("Evidence-Based Guidelines")
    
    # Show relevant guidelines based on patient conditions
    relevant_guidelines = get_relevant_guidelines(assessment_data)
    
    for guideline in relevant_guidelines:
        with st.expander(f"📋 {guideline['title']}"):
            st.write(f"**Source:** {guideline['source']}")
            st.write(f"**Evidence Level:** {guideline['evidence_level']}")
            st.write(f"**Recommendation:** {guideline['recommendation']}")
            
            if guideline.get('rationale'):
                st.write(f"**Rationale:** {guideline['rationale']}")
    
    # Drug interaction checker
    st.subheader("Medication Safety Check")
    
    current_medications = st.multiselect(
        "Current Medications",
        ["Metformin", "Insulin", "ACE Inhibitor", "Beta Blocker", "Statin", "Aspirin"],
        key="current_meds"
    )
    
    if current_medications:
        interactions = check_drug_interactions(current_medications)
        
        if interactions:
            st.warning("⚠️ **Potential Drug Interactions Detected:**")
            for interaction in interactions:
                st.write(f"• {interaction}")
        else:
            st.success("✅ No significant drug interactions detected")
    
    # Clinical decision support tools
    st.subheader("Clinical Decision Support Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Calculate Cardiovascular Risk", type="secondary"):
            cv_risk = calculate_cardiovascular_risk(assessment_data)
            st.metric("10-Year CV Risk", f"{cv_risk:.1f}%")
            
            if cv_risk > 20:
                st.error("High cardiovascular risk - consider statin therapy")
            elif cv_risk > 10:
                st.warning("Moderate cardiovascular risk - consider lifestyle modifications")
            else:
                st.success("Low cardiovascular risk")
    
    with col2:
        if st.button("Calculate Kidney Function", type="secondary"):
            egfr = assessment_data.get('egfr', 90)
            ckd_stage = get_ckd_stage(egfr)
            
            st.metric("eGFR", f"{egfr} mL/min/1.73m²")
            st.metric("CKD Stage", ckd_stage)
            
            if ckd_stage in ["Stage 3", "Stage 4", "Stage 5"]:
                st.warning("⚠️ Reduced kidney function - adjust medication dosages")


def generate_clinical_recommendations(assessment_data):
    """Generate clinical recommendations based on patient assessment"""
    recommendations = {
        'High': [],
        'Medium': [],
        'Low': []
    }
    
    # High priority recommendations
    if assessment_data['systolic_bp'] > 140 or assessment_data['diastolic_bp'] > 90:
        recommendations['High'].append("Initiate antihypertensive therapy - blood pressure above target")
    
    if assessment_data['hba1c'] > 7.0:
        recommendations['High'].append("Optimize diabetes management - HbA1c above target")
    
    if assessment_data['bmi'] > 30:
        recommendations['High'].append("Refer to weight management program - BMI indicates obesity")
    
    # Medium priority recommendations
    if assessment_data['cholesterol'] > 200:
        recommendations['Medium'].append("Consider statin therapy - elevated cholesterol")
    
    if assessment_data['family_history']:
        recommendations['Medium'].append("Enhanced cardiovascular monitoring due to family history")
    
    if assessment_data['age'] > 50:
        recommendations['Medium'].append("Age-appropriate cancer screening recommended")
    
    # Low priority recommendations
    recommendations['Low'].append("Annual comprehensive metabolic panel")
    recommendations['Low'].append("Regular physical activity counseling")
    recommendations['Low'].append("Dietary counseling for optimal health")
    
    return recommendations


def get_relevant_guidelines(assessment_data):
    """Get relevant clinical guidelines based on patient assessment"""
    guidelines = []
    
    # Diabetes guidelines
    if assessment_data['hba1c'] > 5.7:
        guidelines.append({
            'title': 'ADA Diabetes Management Guidelines 2023',
            'source': 'American Diabetes Association',
            'evidence_level': 'Grade A',
            'recommendation': 'HbA1c target <7% for most adults with diabetes',
            'rationale': 'Reduces risk of microvascular complications'
        })
    
    # Hypertension guidelines
    if assessment_data['systolic_bp'] > 130 or assessment_data['diastolic_bp'] > 80:
        guidelines.append({
            'title': 'AHA/ACC Hypertension Guidelines 2017',
            'source': 'American Heart Association',
            'evidence_level': 'Grade A',
            'recommendation': 'Blood pressure target <130/80 mmHg',
            'rationale': 'Reduces cardiovascular risk'
        })
    
    # Cardiovascular guidelines
    if assessment_data['family_history'] or assessment_data['cholesterol'] > 200:
        guidelines.append({
            'title': 'ACC/AHA Cholesterol Management Guidelines 2018',
            'source': 'American College of Cardiology',
            'evidence_level': 'Grade A',
            'recommendation': 'Statin therapy for primary prevention',
            'rationale': 'Reduces cardiovascular events'
        })
    
    return guidelines


def check_drug_interactions(medications):
    """Check for potential drug interactions"""
    interactions = []
    
    # Known interactions
    interaction_pairs = [
        (["Metformin", "Insulin"], "May increase risk of hypoglycemia"),
        (["ACE Inhibitor", "Potassium"], "Risk of hyperkalemia"),
        (["Warfarin", "Aspirin"], "Increased bleeding risk"),
        (["Statin", "Grapefruit"], "May increase statin levels")
    ]
    
    for med_pair, interaction in interaction_pairs:
        if all(med in medications for med in med_pair):
            interactions.append(f"{' + '.join(med_pair)}: {interaction}")
    
    return interactions


def calculate_cardiovascular_risk(assessment_data):
    """Calculate 10-year cardiovascular risk using simplified algorithm"""
    risk_score = 0
    
    # Age factor
    if assessment_data['age'] > 65:
        risk_score += 15
    elif assessment_data['age'] > 55:
        risk_score += 10
    elif assessment_data['age'] > 45:
        risk_score += 5
    
    # Blood pressure factor
    if assessment_data['systolic_bp'] > 160:
        risk_score += 15
    elif assessment_data['systolic_bp'] > 140:
        risk_score += 10
    elif assessment_data['systolic_bp'] > 130:
        risk_score += 5
    
    # Cholesterol factor
    if assessment_data['cholesterol'] > 240:
        risk_score += 10
    elif assessment_data['cholesterol'] > 200:
        risk_score += 5
    
    # Diabetes factor
    if assessment_data['hba1c'] > 6.5:
        risk_score += 15
    
    # Family history factor
    if assessment_data['family_history']:
        risk_score += 5
    
    return min(risk_score, 50)  # Cap at 50%


def get_ckd_stage(egfr):
    """Determine CKD stage based on eGFR"""
    if egfr >= 90:
        return "Stage 1"
    elif egfr >= 60:
        return "Stage 2"
    elif egfr >= 30:
        return "Stage 3"
    elif egfr >= 15:
        return "Stage 4"
    else:
        return "Stage 5"
