"""
Clinical Guidelines Manager for Healthcare DSS
==============================================

This module manages clinical guidelines and evidence-based protocols:
- Clinical guideline management
- Evidence-based protocol handling
- Clinical recommendations generation
- Guidelines integration and validation
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from healthcare_dss.core.knowledge_models import ClinicalGuideline
from healthcare_dss.core.rule_engine import ClinicalRuleEngine

# Configure logging
logger = logging.getLogger(__name__)


class ClinicalGuidelinesManager:
    """
    Manages clinical guidelines and evidence-based protocols
    """
    
    def __init__(self, rule_engine: ClinicalRuleEngine):
        """
        Initialize Clinical Guidelines Manager
        
        Args:
            rule_engine: Instance of ClinicalRuleEngine for condition evaluation
        """
        self.rule_engine = rule_engine
        self.guidelines = {}
        self.default_guidelines_loaded = False
        
        # Load default guidelines
        self._load_default_guidelines()
    
    def _load_default_guidelines(self):
        """Load default clinical guidelines"""
        if self.default_guidelines_loaded:
            return
            
        self._load_diabetes_guidelines()
        self._load_breast_cancer_guidelines()
        self.default_guidelines_loaded = True
        logger.info("Default clinical guidelines loaded")
    
    def _load_diabetes_guidelines(self):
        """Load diabetes-related clinical guidelines"""
        diabetes_guidelines = [
            ClinicalGuideline(
                guideline_id="diabetes_001",
                title="Diabetes Risk Assessment",
                description="Guidelines for assessing diabetes risk based on patient characteristics",
                category="diabetes",
                conditions=["age > 45", "bmi > 25", "family_history", "sedentary_lifestyle"],
                recommendations=[
                    "Perform HbA1c test",
                    "Monitor blood glucose levels",
                    "Recommend lifestyle modifications",
                    "Consider preventive medications"
                ],
                evidence_level="A",
                source="American Diabetes Association",
                version="2023.1",
                created_at=datetime.now()
            ),
            ClinicalGuideline(
                guideline_id="diabetes_002",
                title="Diabetes Management Protocol",
                description="Comprehensive diabetes management guidelines",
                category="diabetes",
                conditions=["confirmed_diabetes", "hba1c > 7%"],
                recommendations=[
                    "Initiate metformin therapy",
                    "Dietary counseling",
                    "Regular exercise program",
                    "Quarterly HbA1c monitoring",
                    "Annual eye examination",
                    "Annual foot examination"
                ],
                evidence_level="A",
                source="American Diabetes Association",
                version="2023.1",
                created_at=datetime.now()
            )
        ]
        
        for guideline in diabetes_guidelines:
            self.add_clinical_guideline(guideline)
    
    def _load_breast_cancer_guidelines(self):
        """Load breast cancer-related clinical guidelines"""
        breast_cancer_guidelines = [
            ClinicalGuideline(
                guideline_id="breast_cancer_001",
                title="Breast Cancer Screening Protocol",
                description="Guidelines for breast cancer screening and early detection",
                category="breast_cancer",
                conditions=["age >= 40", "family_history", "genetic_predisposition"],
                recommendations=[
                    "Annual mammography",
                    "Clinical breast examination",
                    "Breast self-examination education",
                    "Genetic counseling if indicated"
                ],
                evidence_level="A",
                source="American Cancer Society",
                version="2023.1",
                created_at=datetime.now()
            ),
            ClinicalGuideline(
                guideline_id="breast_cancer_002",
                title="Breast Cancer Risk Stratification",
                description="Risk stratification based on imaging and clinical features",
                category="breast_cancer",
                conditions=["suspicious_imaging", "high_risk_features"],
                recommendations=[
                    "Biopsy if BI-RADS 4 or 5",
                    "MRI for high-risk patients",
                    "Multidisciplinary team consultation",
                    "Patient counseling and support"
                ],
                evidence_level="B",
                source="American College of Radiology",
                version="2023.1",
                created_at=datetime.now()
            )
        ]
        
        for guideline in breast_cancer_guidelines:
            self.add_clinical_guideline(guideline)
    
    def add_clinical_guideline(self, guideline: ClinicalGuideline):
        """Add a clinical guideline to the manager"""
        self.guidelines[guideline.guideline_id] = guideline
        logger.info(f"Added clinical guideline: {guideline.title}")
    
    def get_clinical_recommendations(self, patient_data: Dict[str, Any], 
                                   category: str = None) -> List[Dict[str, Any]]:
        """
        Get clinical recommendations based on patient data and guidelines
        
        Args:
            patient_data: Dictionary containing patient information
            category: Optional category filter for guidelines
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Get applicable guidelines
        applicable_guidelines = []
        for guideline_id, guideline in self.guidelines.items():
            if category and guideline.category != category:
                continue
            
            # Check if patient meets guideline conditions
            if self.rule_engine.evaluate_guideline_conditions(guideline.conditions, patient_data):
                applicable_guidelines.append(guideline)
        
        # Generate recommendations from guidelines
        for guideline in applicable_guidelines:
            for recommendation in guideline.recommendations:
                recommendations.append({
                    'guideline_id': guideline.guideline_id,
                    'title': guideline.title,
                    'recommendation': recommendation,
                    'evidence_level': guideline.evidence_level,
                    'source': guideline.source
                })
        
        return recommendations
    
    def get_guidelines_by_category(self, category: str) -> List[ClinicalGuideline]:
        """Get all guidelines for a specific category"""
        return [guideline for guideline in self.guidelines.values() 
                if guideline.category == category]
    
    def get_guideline_by_id(self, guideline_id: str) -> Optional[ClinicalGuideline]:
        """Get a specific guideline by ID"""
        return self.guidelines.get(guideline_id)
    
    def get_clinical_guidelines_integration(self) -> Dict[str, Any]:
        """
        Get clinical guidelines integration following Healthcare_DSS_Architecture.md
        
        Returns:
            Dictionary containing clinical guidelines and evidence-based protocols
        """
        guidelines_integration = {
            'evidence_based_protocols': {
                'diabetes_management': {
                    'guideline_source': 'ADA 2023 Guidelines',
                    'evidence_level': 'Grade A',
                    'protocols': [
                        'HbA1c monitoring every 3 months',
                        'Blood pressure target <130/80 mmHg',
                        'LDL cholesterol target <100 mg/dL',
                        'Annual comprehensive foot examination'
                    ],
                    'clinical_rules': [
                        'IF HbA1c > 7.0% THEN recommend medication adjustment',
                        'IF blood pressure > 140/90 THEN initiate antihypertensive therapy',
                        'IF LDL > 100 mg/dL THEN consider statin therapy'
                    ]
                },
                'hypertension_management': {
                    'guideline_source': 'AHA/ACC 2017 Guidelines',
                    'evidence_level': 'Grade A',
                    'protocols': [
                        'Blood pressure measurement at each visit',
                        'Lifestyle modifications as first-line therapy',
                        'ACE inhibitor or ARB as first-line medication',
                        'Target BP <130/80 mmHg for most patients'
                    ],
                    'clinical_rules': [
                        'IF systolic BP > 140 OR diastolic BP > 90 THEN diagnose hypertension',
                        'IF BP > 160/100 THEN consider immediate medication',
                        'IF BP <120/80 THEN consider lifestyle counseling'
                    ]
                },
                'cardiac_care': {
                    'guideline_source': 'ACC/AHA 2021 Guidelines',
                    'evidence_level': 'Grade A',
                    'protocols': [
                        'ECG within 10 minutes of chest pain',
                        'Troponin levels at presentation and 3-6 hours',
                        'Aspirin 325mg for suspected MI',
                        'Beta-blocker within 24 hours if no contraindications'
                    ],
                    'clinical_rules': [
                        'IF chest pain + ST elevation THEN activate STEMI protocol',
                        'IF troponin elevated THEN consider cardiac catheterization',
                        'IF heart rate > 100 THEN consider beta-blocker'
                    ]
                }
            },
            'clinical_decision_support_rules': {
                'medication_interactions': {
                    'warfarin_aspirin': {
                        'interaction': 'Increased bleeding risk',
                        'severity': 'High',
                        'recommendation': 'Monitor INR closely, consider alternative',
                        'evidence_level': 'Grade B'
                    },
                    'ace_inhibitor_potassium': {
                        'interaction': 'Hyperkalemia risk',
                        'severity': 'Moderate',
                        'recommendation': 'Monitor potassium levels',
                        'evidence_level': 'Grade A'
                    }
                },
                'contraindications': {
                    'metformin_renal': {
                        'condition': 'eGFR < 30 mL/min/1.73m²',
                        'contraindication': 'Metformin contraindicated',
                        'alternative': 'Consider insulin therapy',
                        'evidence_level': 'Grade A'
                    },
                    'ace_inhibitor_pregnancy': {
                        'condition': 'Pregnancy',
                        'contraindication': 'ACE inhibitors contraindicated',
                        'alternative': 'Consider methyldopa or labetalol',
                        'evidence_level': 'Grade A'
                    }
                }
            },
            'quality_measures': {
                'diabetes_care': {
                    'hba1c_testing': {
                        'measure': 'Percentage of patients with HbA1c tested in past 12 months',
                        'target': '>90%',
                        'numerator': 'Patients with HbA1c test',
                        'denominator': 'All diabetic patients'
                    },
                    'eye_examination': {
                        'measure': 'Percentage of patients with annual eye examination',
                        'target': '>80%',
                        'numerator': 'Patients with eye exam',
                        'denominator': 'All diabetic patients'
                    }
                },
                'hypertension_care': {
                    'bp_control': {
                        'measure': 'Percentage of patients with BP <140/90',
                        'target': '>70%',
                        'numerator': 'Patients with controlled BP',
                        'denominator': 'All hypertensive patients'
                    }
                }
            }
        }
        
        return guidelines_integration
    
    def get_evidence_based_protocols(self, condition: str = "diabetes") -> Dict[str, Any]:
        """
        Get evidence-based protocols for specific conditions
        
        Args:
            condition: Medical condition (diabetes, hypertension, cardiac, etc.)
            
        Returns:
            Dictionary containing evidence-based protocols
        """
        protocols = {
            'diabetes': {
                'diagnostic_criteria': {
                    'hba1c': '≥6.5%',
                    'fasting_glucose': '≥126 mg/dL',
                    'random_glucose': '≥200 mg/dL with symptoms',
                    'ogtt_2hr': '≥200 mg/dL'
                },
                'treatment_algorithm': {
                    'step_1': 'Lifestyle modifications (diet, exercise)',
                    'step_2': 'Metformin as first-line medication',
                    'step_3': 'Add second agent (SGLT2 inhibitor, GLP-1 RA)',
                    'step_4': 'Consider insulin therapy'
                },
                'monitoring_schedule': {
                    'hba1c': 'Every 3 months until <7%, then every 6 months',
                    'blood_pressure': 'Every visit',
                    'lipid_panel': 'Annually',
                    'eye_exam': 'Annually',
                    'foot_exam': 'Annually'
                },
                'complications_screening': {
                    'retinopathy': 'Annual dilated eye exam',
                    'nephropathy': 'Annual microalbuminuria test',
                    'neuropathy': 'Annual monofilament test',
                    'cardiovascular': 'Annual ECG, consider stress test'
                }
            },
            'hypertension': {
                'diagnostic_criteria': {
                    'stage_1': '130-139/80-89 mmHg',
                    'stage_2': '≥140/90 mmHg',
                    'hypertensive_crisis': '≥180/120 mmHg'
                },
                'treatment_algorithm': {
                    'stage_1': 'Lifestyle modifications for 3-6 months',
                    'stage_2': 'Lifestyle + medication',
                    'crisis': 'Immediate medication adjustment'
                },
                'first_line_medications': [
                    'ACE inhibitor',
                    'ARB',
                    'Thiazide diuretic',
                    'Calcium channel blocker'
                ],
                'monitoring_schedule': {
                    'blood_pressure': 'Every visit',
                    'kidney_function': 'Every 6-12 months',
                    'electrolytes': 'Every 3-6 months'
                }
            },
            'cardiac_care': {
                'chest_pain_protocol': {
                    'immediate_assessment': 'ECG within 10 minutes',
                    'biomarkers': 'Troponin at 0, 3, 6 hours',
                    'aspirin': '325mg chewable if no contraindications',
                    'nitroglycerin': 'Sublingual if systolic BP >90'
                },
                'stemi_protocol': {
                    'activation': 'Immediate cardiac catheterization',
                    'medications': 'Aspirin, clopidogrel, heparin',
                    'door_to_balloon': '<90 minutes',
                    'fibrinolytics': 'If PCI not available within 120 minutes'
                },
                'monitoring_requirements': {
                    'continuous_ecg': 'Until stable',
                    'vital_signs': 'Every 15 minutes initially',
                    'cardiac_enzymes': 'Serial measurements',
                    'echocardiogram': 'Within 24 hours'
                }
            }
        }
        
        return protocols.get(condition, {})
    
    def get_guideline_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded guidelines"""
        stats = {
            'total_guidelines': len(self.guidelines)
        }
        
        # Count by category
        categories = {}
        for guideline in self.guidelines.values():
            category = guideline.category
            categories[category] = categories.get(category, 0) + 1
        
        stats.update(categories)
        
        # Count by evidence level
        evidence_levels = {}
        for guideline in self.guidelines.values():
            level = guideline.evidence_level
            evidence_levels[f'evidence_level_{level}'] = evidence_levels.get(f'evidence_level_{level}', 0) + 1
        
        stats.update(evidence_levels)
        
        return stats
    
    def search_guidelines(self, query: str) -> List[Dict[str, Any]]:
        """Search guidelines by title or description"""
        results = []
        query_lower = query.lower()
        
        for guideline_id, guideline in self.guidelines.items():
            if (query_lower in guideline.title.lower() or 
                query_lower in guideline.description.lower()):
                results.append({
                    'guideline_id': guideline_id,
                    'title': guideline.title,
                    'description': guideline.description,
                    'category': guideline.category,
                    'evidence_level': guideline.evidence_level,
                    'source': guideline.source
                })
        
        return results
