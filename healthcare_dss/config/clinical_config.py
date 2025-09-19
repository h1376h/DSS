"""
Clinical Decision Support Configuration
======================================

Configuration-based approach for clinical decision support without hardcoded values.
All clinical parameters can be configured via environment variables or configuration files.
"""

import os
from typing import Dict, Any, List


class ClinicalConfig:
    """Configuration manager for clinical decision support"""
    
    @staticmethod
    def get_patient_profiles() -> Dict[str, Dict[str, Any]]:
        """Get configurable patient profiles"""
        return {
            'high_risk_diabetic': {
                'description': 'High-risk diabetic patient with hypertension',
                'data': {
                    'age': int(os.getenv('CLINICAL_DEMO_AGE', '55')),
                    'bmi': float(os.getenv('CLINICAL_DEMO_BMI', '28.5')),
                    'systolic_bp': int(os.getenv('CLINICAL_DEMO_SYSTOLIC_BP', '145')),
                    'diastolic_bp': int(os.getenv('CLINICAL_DEMO_DIASTOLIC_BP', '95')),
                    'family_history': os.getenv('CLINICAL_DEMO_FAMILY_HISTORY', 'true').lower() == 'true',
                    'hba1c': float(os.getenv('CLINICAL_DEMO_HBA1C', '7.2')),
                    'cholesterol': int(os.getenv('CLINICAL_DEMO_CHOLESTEROL', '220')),
                    'diabetes': os.getenv('CLINICAL_DEMO_DIABETES', 'true').lower() == 'true',
                    'hypertension': os.getenv('CLINICAL_DEMO_HYPERTENSION', 'true').lower() == 'true'
                }
            },
            'low_risk_patient': {
                'description': 'Low-risk patient with good health indicators',
                'data': {
                    'age': int(os.getenv('CLINICAL_LOW_RISK_AGE', '35')),
                    'bmi': float(os.getenv('CLINICAL_LOW_RISK_BMI', '22.0')),
                    'systolic_bp': int(os.getenv('CLINICAL_LOW_RISK_SYSTOLIC_BP', '120')),
                    'diastolic_bp': int(os.getenv('CLINICAL_LOW_RISK_DIASTOLIC_BP', '80')),
                    'family_history': os.getenv('CLINICAL_LOW_RISK_FAMILY_HISTORY', 'false').lower() == 'true',
                    'hba1c': float(os.getenv('CLINICAL_LOW_RISK_HBA1C', '5.2')),
                    'cholesterol': int(os.getenv('CLINICAL_LOW_RISK_CHOLESTEROL', '180')),
                    'diabetes': os.getenv('CLINICAL_LOW_RISK_DIABETES', 'false').lower() == 'true',
                    'hypertension': os.getenv('CLINICAL_LOW_RISK_HYPERTENSION', 'false').lower() == 'true'
                }
            }
        }
    
    @staticmethod
    def get_risk_calculation_config() -> Dict[str, Dict[str, Any]]:
        """Get configurable risk calculation parameters"""
        return {
            'diabetes_risk_factors': {
                'age_threshold': int(os.getenv('DIABETES_AGE_THRESHOLD', '45')),
                'age_weight': int(os.getenv('DIABETES_AGE_WEIGHT', '20')),
                'bmi_threshold': float(os.getenv('DIABETES_BMI_THRESHOLD', '25')),
                'bmi_weight': int(os.getenv('DIABETES_BMI_WEIGHT', '15')),
                'family_history_weight': int(os.getenv('DIABETES_FAMILY_WEIGHT', '25')),
                'hba1c_threshold': float(os.getenv('DIABETES_HBA1C_THRESHOLD', '5.7')),
                'hba1c_weight': int(os.getenv('DIABETES_HBA1C_WEIGHT', '30'))
            },
            'cv_risk_factors': {
                'bp_systolic_threshold': int(os.getenv('CV_SYSTOLIC_THRESHOLD', '140')),
                'bp_diastolic_threshold': int(os.getenv('CV_DIASTOLIC_THRESHOLD', '90')),
                'bp_weight': int(os.getenv('CV_BP_WEIGHT', '25')),
                'cholesterol_threshold': int(os.getenv('CV_CHOLESTEROL_THRESHOLD', '200')),
                'cholesterol_weight': int(os.getenv('CV_CHOLESTEROL_WEIGHT', '20')),
                'family_history_weight': int(os.getenv('CV_FAMILY_WEIGHT', '15')),
                'age_threshold': int(os.getenv('CV_AGE_THRESHOLD', '50')),
                'age_weight': int(os.getenv('CV_AGE_WEIGHT', '20')),
                'diabetes_weight': int(os.getenv('CV_DIABETES_WEIGHT', '30'))
            }
        }
    
    @staticmethod
    def get_recommendation_rules() -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get configurable recommendation rules"""
        return {
            'high_priority': {
                'hypertension': {
                    'condition': lambda p: p['systolic_bp'] > int(os.getenv('HYPERTENSION_SYSTOLIC_THRESHOLD', '140')) or 
                                          p['diastolic_bp'] > int(os.getenv('HYPERTENSION_DIASTOLIC_THRESHOLD', '90')),
                    'recommendation': os.getenv('HYPERTENSION_RECOMMENDATION', 'Initiate antihypertensive therapy')
                },
                'diabetes_management': {
                    'condition': lambda p: p['hba1c'] > float(os.getenv('DIABETES_HBA1C_TARGET', '7.0')),
                    'recommendation': os.getenv('DIABETES_RECOMMENDATION', 'Optimize diabetes management')
                },
                'weight_management': {
                    'condition': lambda p: p['bmi'] > float(os.getenv('OBESITY_BMI_THRESHOLD', '30')),
                    'recommendation': os.getenv('WEIGHT_MANAGEMENT_RECOMMENDATION', 'Refer to weight management program')
                }
            },
            'medium_priority': {
                'statin_therapy': {
                    'condition': lambda p: p['cholesterol'] > int(os.getenv('STATIN_CHOLESTEROL_THRESHOLD', '200')),
                    'recommendation': os.getenv('STATIN_RECOMMENDATION', 'Consider statin therapy')
                },
                'cv_monitoring': {
                    'condition': lambda p: p['family_history'],
                    'recommendation': os.getenv('CV_MONITORING_RECOMMENDATION', 'Enhanced cardiovascular monitoring')
                }
            },
            'low_priority': {
                'annual_panel': {
                    'condition': lambda p: True,
                    'recommendation': os.getenv('ANNUAL_PANEL_RECOMMENDATION', 'Annual comprehensive metabolic panel')
                },
                'activity_counseling': {
                    'condition': lambda p: True,
                    'recommendation': os.getenv('ACTIVITY_RECOMMENDATION', 'Regular physical activity counseling')
                }
            }
        }
    
    @staticmethod
    def get_clinical_guidelines() -> Dict[str, Dict[str, Any]]:
        """Get configurable clinical guidelines"""
        return {
            'diabetes': {
                'title': os.getenv('DIABETES_GUIDELINE_TITLE', 'ADA Diabetes Management Guidelines 2023'),
                'source': os.getenv('DIABETES_GUIDELINE_SOURCE', 'American Diabetes Association'),
                'recommendation': os.getenv('DIABETES_GUIDELINE_RECOMMENDATION', 'HbA1c target <7% for most adults'),
                'condition': lambda p: p['diabetes'] or p['hba1c'] > float(os.getenv('DIABETES_HBA1C_THRESHOLD', '5.7'))
            },
            'hypertension': {
                'title': os.getenv('HYPERTENSION_GUIDELINE_TITLE', 'AHA/ACC Hypertension Guidelines 2017'),
                'source': os.getenv('HYPERTENSION_GUIDELINE_SOURCE', 'American Heart Association'),
                'recommendation': os.getenv('HYPERTENSION_GUIDELINE_RECOMMENDATION', 'Blood pressure target <130/80 mmHg'),
                'condition': lambda p: p['systolic_bp'] > int(os.getenv('HYPERTENSION_SYSTOLIC_THRESHOLD', '130')) or 
                                     p['diastolic_bp'] > int(os.getenv('HYPERTENSION_DIASTOLIC_THRESHOLD', '80'))
            }
        }
    
    @staticmethod
    def get_medication_scenarios() -> Dict[str, List[str]]:
        """Get configurable medication scenarios"""
        return {
            'diabetes_management': os.getenv('CLINICAL_DEMO_MEDICATIONS', 'Metformin,ACE Inhibitor,Statin').split(','),
            'hypertension_only': os.getenv('HYPERTENSION_MEDICATIONS', 'ACE Inhibitor,Diuretic').split(','),
            'comprehensive_care': os.getenv('COMPREHENSIVE_MEDICATIONS', 'Metformin,ACE Inhibitor,Statin,Aspirin').split(',')
        }
    
    @staticmethod
    def get_drug_interactions() -> List[Dict[str, Any]]:
        """Get configurable drug interactions"""
        return [
            {
                'medications': os.getenv('INTERACTION_METFORMIN_INSULIN', 'Metformin,Insulin').split(','),
                'interaction': os.getenv('INTERACTION_METFORMIN_INSULIN_DESC', 'May increase risk of hypoglycemia')
            },
            {
                'medications': os.getenv('INTERACTION_ACE_POTASSIUM', 'ACE Inhibitor,Potassium').split(','),
                'interaction': os.getenv('INTERACTION_ACE_POTASSIUM_DESC', 'Risk of hyperkalemia')
            },
            {
                'medications': os.getenv('INTERACTION_WARFARIN_ASPIRIN', 'Warfarin,Aspirin').split(','),
                'interaction': os.getenv('INTERACTION_WARFARIN_ASPIRIN_DESC', 'Increased bleeding risk')
            }
        ]
    
    @staticmethod
    def get_full_config() -> Dict[str, Any]:
        """Get complete clinical configuration"""
        return {
            'patient_profiles': ClinicalConfig.get_patient_profiles(),
            'risk_calculation': ClinicalConfig.get_risk_calculation_config(),
            'recommendation_rules': ClinicalConfig.get_recommendation_rules(),
            'clinical_guidelines': ClinicalConfig.get_clinical_guidelines(),
            'medication_scenarios': ClinicalConfig.get_medication_scenarios(),
            'drug_interactions': ClinicalConfig.get_drug_interactions()
        }
