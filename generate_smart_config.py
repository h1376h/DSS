#!/usr/bin/env python3
"""
Smart Dataset Configuration Generator
Creates a focused configuration file for smarter DSS functionalities
based on the analyzed dataset targets
"""

import json
import pandas as pd
from pathlib import Path

def create_smart_config():
    """Create a smart configuration file for DSS functionalities"""
    
    # Load the analysis results
    with open('dataset_target_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    # Create smart configuration
    smart_config = {
        "dataset_configurations": {},
        "ml_models": {
            "classification": {
                "binary": {
                    "models": ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "SVM"],
                    "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
                    "visualizations": ["confusion_matrix", "roc_curve", "precision_recall_curve"]
                },
                "multiclass": {
                    "models": ["RandomForestClassifier", "XGBClassifier", "SVM", "LogisticRegression"],
                    "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                    "visualizations": ["confusion_matrix", "classification_report"]
                }
            },
            "regression": {
                "models": ["RandomForestRegressor", "XGBRegressor", "LinearRegression", "SVR", "Ridge"],
                "metrics": ["rmse", "mae", "r2_score", "mape"],
                "visualizations": ["residual_plot", "prediction_vs_actual", "feature_importance"]
            }
        },
        "smart_features": {
            "risk_stratification": [],
            "outcome_prediction": [],
            "performance_monitoring": [],
            "resource_optimization": [],
            "clinical_decision_support": []
        },
        "dashboard_configurations": {},
        "preprocessing_pipelines": {}
    }
    
    # Process each dataset
    for filename, dataset_info in analysis["dataset_targets"].items():
        dataset_name = filename.replace('.csv', '')
        
        # Dataset configuration
        dataset_config = {
            "file_path": f"datasets/raw/{filename}",
            "dataset_type": dataset_info["dataset_type"],
            "ml_task_type": dataset_info["ml_task_type"],
            "shape": dataset_info["shape"],
            "primary_targets": [],
            "feature_variables": dataset_info["feature_variables"],
            "recommended_models": dataset_info["recommended_models"],
            "preprocessing_needed": dataset_info["preprocessing_needed"],
            "smart_functionalities": []
        }
        
        # Process primary targets
        for target in dataset_info["primary_targets"]:
            target_config = {
                "column": target["column"],
                "target_type": target["target_type"],
                "data_type": target["data_type"],
                "unique_values": target["unique_values"],
                "missing_values": target["missing_values"],
                "business_meaning": get_business_meaning(target["column"], dataset_name),
                "smart_features": get_smart_features(target["column"], dataset_name, target["target_type"])
            }
            dataset_config["primary_targets"].append(target_config)
            
            # Add to smart features
            add_to_smart_features(smart_config, target_config, dataset_name)
        
        # Add dataset-specific smart functionalities
        add_dataset_smart_functionalities(dataset_config, dataset_name, dataset_info)
        
        smart_config["dataset_configurations"][dataset_name] = dataset_config
    
    # Create dashboard configurations
    create_dashboard_configurations(smart_config)
    
    # Create preprocessing pipelines
    create_preprocessing_pipelines(smart_config)
    
    return smart_config

def get_business_meaning(column_name, dataset_name):
    """Get business meaning for target variables"""
    meanings = {
        "treatment_success": "Binary indicator of successful treatment outcome (1=success, 0=failure)",
        "quality_of_life_score": "Continuous score measuring patient quality of life (0-10 scale)",
        "readmission_risk": "Probability of patient readmission (0-1 scale)",
        "patient_satisfaction_score": "Patient satisfaction rating (0-10 scale)",
        "performance_rating": "Staff performance rating (0-5 scale)",
        "quality_score": "Overall quality metric (0-100 scale)",
        "target": "Primary prediction target (varies by dataset)",
        "diagnosis": "Medical diagnosis classification",
        "effectiveness": "Medication effectiveness score",
        "capacity_value": "Hospital capacity utilization value",
        "Waist": "Waist measurement (physiological metric)",
        "target_name": "Named classification target"
    }
    
    return meanings.get(column_name, f"Target variable: {column_name}")

def get_smart_features(column_name, dataset_name, target_type):
    """Get smart features for target variables"""
    features = []
    
    # Risk stratification features
    if "risk" in column_name.lower() or "readmission" in column_name.lower():
        features.extend([
            "risk_score_calculation",
            "risk_category_assignment",
            "risk_trend_analysis",
            "high_risk_patient_alerts"
        ])
    
    # Outcome prediction features
    if "success" in column_name.lower() or "outcome" in column_name.lower():
        features.extend([
            "outcome_probability_prediction",
            "success_rate_monitoring",
            "outcome_trend_analysis",
            "intervention_recommendations"
        ])
    
    # Performance monitoring features
    if "performance" in column_name.lower() or "satisfaction" in column_name.lower():
        features.extend([
            "performance_score_tracking",
            "satisfaction_trend_analysis",
            "performance_benchmarking",
            "improvement_recommendations"
        ])
    
    # Quality monitoring features
    if "quality" in column_name.lower():
        features.extend([
            "quality_score_monitoring",
            "quality_trend_analysis",
            "quality_benchmarking",
            "quality_improvement_alerts"
        ])
    
    # Clinical decision support features
    if dataset_name in ["breast_cancer_scikit", "diabetes_scikit", "medication_effectiveness_scikit"]:
        features.extend([
            "clinical_prediction_support",
            "diagnosis_assistance",
            "treatment_recommendation",
            "clinical_alert_system"
        ])
    
    return features

def add_to_smart_features(smart_config, target_config, dataset_name):
    """Add target to appropriate smart feature categories"""
    column_name = target_config["column"].lower()
    
    # Risk stratification
    if "risk" in column_name or "readmission" in column_name:
        smart_config["smart_features"]["risk_stratification"].append({
            "dataset": dataset_name,
            "target": target_config["column"],
            "target_type": target_config["target_type"],
            "features": target_config["smart_features"]
        })
    
    # Outcome prediction
    if "success" in column_name or "outcome" in column_name or "effectiveness" in column_name:
        smart_config["smart_features"]["outcome_prediction"].append({
            "dataset": dataset_name,
            "target": target_config["column"],
            "target_type": target_config["target_type"],
            "features": target_config["smart_features"]
        })
    
    # Performance monitoring
    if "performance" in column_name or "satisfaction" in column_name:
        smart_config["smart_features"]["performance_monitoring"].append({
            "dataset": dataset_name,
            "target": target_config["column"],
            "target_type": target_config["target_type"],
            "features": target_config["smart_features"]
        })
    
    # Clinical decision support
    if dataset_name in ["breast_cancer_scikit", "diabetes_scikit", "medication_effectiveness_scikit"]:
        smart_config["smart_features"]["clinical_decision_support"].append({
            "dataset": dataset_name,
            "target": target_config["column"],
            "target_type": target_config["target_type"],
            "features": target_config["smart_features"]
        })

def add_dataset_smart_functionalities(dataset_config, dataset_name, dataset_info):
    """Add dataset-specific smart functionalities"""
    functionalities = []
    
    if dataset_name == "breast_cancer_scikit":
        functionalities.extend([
            "cancer_diagnosis_prediction",
            "malignancy_probability_calculation",
            "treatment_recommendation_system",
            "patient_risk_assessment"
        ])
    
    elif dataset_name == "diabetes_scikit":
        functionalities.extend([
            "diabetes_progression_prediction",
            "risk_factor_analysis",
            "treatment_effectiveness_monitoring",
            "patient_monitoring_alerts"
        ])
    
    elif dataset_name == "clinical_outcomes_synthetic":
        functionalities.extend([
            "treatment_success_prediction",
            "complication_risk_assessment",
            "patient_satisfaction_monitoring",
            "quality_of_life_tracking"
        ])
    
    elif dataset_name == "patient_demographics_synthetic":
        functionalities.extend([
            "readmission_risk_prediction",
            "patient_segmentation",
            "demographic_risk_analysis",
            "admission_planning"
        ])
    
    elif dataset_name == "staff_performance_synthetic":
        functionalities.extend([
            "staff_performance_evaluation",
            "satisfaction_monitoring",
            "training_recommendations",
            "workload_optimization"
        ])
    
    elif dataset_name == "department_performance_synthetic":
        functionalities.extend([
            "department_efficiency_monitoring",
            "resource_utilization_analysis",
            "quality_score_tracking",
            "performance_benchmarking"
        ])
    
    elif dataset_name == "financial_metrics_synthetic":
        functionalities.extend([
            "financial_performance_monitoring",
            "cost_optimization_analysis",
            "revenue_prediction",
            "budget_planning_support"
        ])
    
    elif dataset_name == "hospital_capacity_scikit":
        functionalities.extend([
            "capacity_utilization_prediction",
            "resource_allocation_optimization",
            "demand_forecasting",
            "capacity_planning"
        ])
    
    elif dataset_name == "medication_effectiveness_scikit":
        functionalities.extend([
            "medication_effectiveness_prediction",
            "drug_response_analysis",
            "treatment_optimization",
            "adverse_event_prediction"
        ])
    
    dataset_config["smart_functionalities"] = functionalities

def create_dashboard_configurations(smart_config):
    """Create dashboard configurations for different user types"""
    
    # Clinical Staff Dashboard
    smart_config["dashboard_configurations"]["clinical_staff"] = {
        "title": "Clinical Decision Support Dashboard",
        "datasets": ["breast_cancer_scikit", "diabetes_scikit", "clinical_outcomes_synthetic", "medication_effectiveness_scikit"],
        "widgets": [
            {
                "type": "prediction_card",
                "title": "Patient Risk Assessment",
                "targets": ["target", "readmission_risk", "treatment_success"],
                "visualization": "gauge"
            },
            {
                "type": "outcome_monitor",
                "title": "Treatment Outcomes",
                "targets": ["treatment_success", "quality_of_life_score"],
                "visualization": "line_chart"
            },
            {
                "type": "diagnosis_support",
                "title": "Diagnostic Support",
                "targets": ["diagnosis", "target"],
                "visualization": "confusion_matrix"
            }
        ]
    }
    
    # Administrative Staff Dashboard
    smart_config["dashboard_configurations"]["administrative"] = {
        "title": "Administrative Performance Dashboard",
        "datasets": ["staff_performance_synthetic", "department_performance_synthetic", "financial_metrics_synthetic"],
        "widgets": [
            {
                "type": "performance_monitor",
                "title": "Staff Performance",
                "targets": ["performance_rating", "patient_satisfaction_score"],
                "visualization": "bar_chart"
            },
            {
                "type": "financial_overview",
                "title": "Financial Metrics",
                "targets": ["revenue", "expenses", "quality_score"],
                "visualization": "line_chart"
            },
            {
                "type": "department_efficiency",
                "title": "Department Performance",
                "targets": ["quality_score", "patient_satisfaction"],
                "visualization": "heatmap"
            }
        ]
    }
    
    # Management Dashboard
    smart_config["dashboard_configurations"]["management"] = {
        "title": "Executive Management Dashboard",
        "datasets": ["financial_metrics_synthetic", "hospital_capacity_scikit", "patient_demographics_synthetic"],
        "widgets": [
            {
                "type": "kpi_overview",
                "title": "Key Performance Indicators",
                "targets": ["quality_score", "patient_satisfaction", "readmission_risk"],
                "visualization": "kpi_cards"
            },
            {
                "type": "capacity_analysis",
                "title": "Hospital Capacity",
                "targets": ["capacity_value", "occupancy_rate"],
                "visualization": "gauge"
            },
            {
                "type": "trend_analysis",
                "title": "Performance Trends",
                "targets": ["quality_score", "patient_satisfaction"],
                "visualization": "trend_chart"
            }
        ]
    }

def create_preprocessing_pipelines(smart_config):
    """Create preprocessing pipelines for different data types"""
    
    smart_config["preprocessing_pipelines"] = {
        "classification": {
            "steps": [
                "handle_missing_values",
                "encode_categorical_variables",
                "scale_numerical_features",
                "balance_dataset",
                "feature_selection"
            ],
            "parameters": {
                "missing_value_strategy": "median",
                "encoding_method": "label_encoding",
                "scaling_method": "standard_scaler",
                "balancing_method": "smote",
                "feature_selection_method": "mutual_info"
            }
        },
        "regression": {
            "steps": [
                "handle_missing_values",
                "encode_categorical_variables",
                "scale_numerical_features",
                "outlier_detection",
                "feature_selection"
            ],
            "parameters": {
                "missing_value_strategy": "mean",
                "encoding_method": "one_hot_encoding",
                "scaling_method": "standard_scaler",
                "outlier_method": "isolation_forest",
                "feature_selection_method": "correlation"
            }
        },
        "healthcare_specific": {
            "steps": [
                "validate_medical_values",
                "handle_missing_values",
                "encode_categorical_variables",
                "scale_numerical_features",
                "feature_engineering",
                "feature_selection"
            ],
            "parameters": {
                "medical_validation": True,
                "missing_value_strategy": "medical_imputation",
                "encoding_method": "label_encoding",
                "scaling_method": "robust_scaler",
                "feature_engineering": "domain_specific",
                "feature_selection_method": "mutual_info"
            }
        }
    }

def main():
    """Main function to generate smart configuration"""
    print("=" * 80)
    print("GENERATING SMART DATASET CONFIGURATION")
    print("=" * 80)
    
    # Create smart configuration
    smart_config = create_smart_config()
    
    # Save configuration
    with open('smart_dataset_config.json', 'w') as f:
        json.dump(smart_config, f, indent=2, default=str)
    
    print(f"Smart configuration saved to 'smart_dataset_config.json'")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    
    print(f"Total datasets configured: {len(smart_config['dataset_configurations'])}")
    print(f"Smart feature categories: {len(smart_config['smart_features'])}")
    print(f"Dashboard configurations: {len(smart_config['dashboard_configurations'])}")
    print(f"Preprocessing pipelines: {len(smart_config['preprocessing_pipelines'])}")
    
    print("\nSmart Features by Category:")
    for category, features in smart_config['smart_features'].items():
        print(f"  {category}: {len(features)} features")
    
    print("\nDashboard Configurations:")
    for dashboard_type, config in smart_config['dashboard_configurations'].items():
        print(f"  {dashboard_type}: {len(config['widgets'])} widgets")
    
    print("\n" + "=" * 80)
    print("Smart configuration generation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
