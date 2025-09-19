#!/usr/bin/env python3
"""
Smart Dataset Target Manager
Provides easy access to dataset target configurations for smarter DSS functionalities
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

class SmartDatasetTargetManager:
    """Manages dataset target configurations for smarter DSS functionalities"""
    
    def __init__(self, config_file: str = "smart_dataset_config.json"):
        """Initialize the target manager with configuration file"""
        self.config_file = config_file
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file {self.config_file} not found!")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            return {}
    
    def _get_dataset_name_variations(self, dataset_name: str) -> List[str]:
        """Get common variations of dataset names"""
        variations = []
        
        # Common mappings
        name_mappings = {
            'diabetes': 'diabetes_scikit',
            'breast_cancer': 'breast_cancer_scikit',
            'wine': 'wine_dataset',
            'linnerud': 'linnerud_dataset',
            'hospital_capacity': 'hospital_capacity_scikit',
            'medication_effectiveness': 'medication_effectiveness_scikit',
            'clinical_outcomes': 'clinical_outcomes_synthetic',
            'patient_demographics': 'patient_demographics_synthetic',
            'staff_performance': 'staff_performance_synthetic',
            'department_performance': 'department_performance_synthetic',
            'financial_metrics': 'financial_metrics_synthetic'
        }
        
        # Add direct mapping if exists
        if dataset_name in name_mappings:
            variations.append(name_mappings[dataset_name])
        
        # Add variations with common suffixes
        suffixes = ['_scikit', '_synthetic', '_dataset']
        for suffix in suffixes:
            if not dataset_name.endswith(suffix):
                variations.append(dataset_name + suffix)
        
        # Add variations without common suffixes
        for suffix in suffixes:
            if dataset_name.endswith(suffix):
                variations.append(dataset_name.replace(suffix, ''))
        
        return variations
    
    def get_dataset_targets(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get target variables for a specific dataset"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("primary_targets", [])
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                self.logger.info(f"Found dataset '{variation}' for query '{dataset_name}'")
                return self.config["dataset_configurations"][variation].get("primary_targets", [])
        
        self.logger.warning(f"Dataset {dataset_name} not found in configuration. Available datasets: {list(self.config.get('dataset_configurations', {}).keys())}")
        return []
    
    def get_dataset_features(self, dataset_name: str) -> List[str]:
        """Get feature variables for a specific dataset"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("feature_variables", [])
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                return self.config["dataset_configurations"][variation].get("feature_variables", [])
        
        return []
    
    def get_recommended_models(self, dataset_name: str) -> List[str]:
        """Get recommended ML models for a dataset"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("recommended_models", [])
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                return self.config["dataset_configurations"][variation].get("recommended_models", [])
        
        return []
    
    def get_smart_functionalities(self, dataset_name: str) -> List[str]:
        """Get smart functionalities available for a dataset"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("smart_functionalities", [])
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                return self.config["dataset_configurations"][variation].get("smart_functionalities", [])
        
        return []
    
    def get_ml_task_type(self, dataset_name: str) -> str:
        """Get the ML task type for a dataset"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("ml_task_type", "unknown")
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                return self.config["dataset_configurations"][variation].get("ml_task_type", "unknown")
        
        return "unknown"
    
    def get_dataset_type(self, dataset_name: str) -> str:
        """Get the dataset type (healthcare, business, general)"""
        # Try exact match first
        if dataset_name in self.config.get("dataset_configurations", {}):
            return self.config["dataset_configurations"][dataset_name].get("dataset_type", "unknown")
        
        # Try common name variations
        name_variations = self._get_dataset_name_variations(dataset_name)
        for variation in name_variations:
            if variation in self.config.get("dataset_configurations", {}):
                return self.config["dataset_configurations"][variation].get("dataset_type", "unknown")
        
        return "unknown"
    
    def get_smart_features_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get smart features by category (risk_stratification, outcome_prediction, etc.)"""
        return self.config.get("smart_features", {}).get(category, [])
    
    def get_dashboard_config(self, dashboard_type: str) -> Dict[str, Any]:
        """Get dashboard configuration for a specific user type"""
        return self.config.get("dashboard_configurations", {}).get(dashboard_type, {})
    
    def get_preprocessing_pipeline(self, pipeline_type: str) -> Dict[str, Any]:
        """Get preprocessing pipeline configuration"""
        return self.config.get("preprocessing_pipelines", {}).get(pipeline_type, {})
    
    def get_target_business_meaning(self, dataset_name: str, target_column: str) -> str:
        """Get business meaning for a specific target variable"""
        targets = self.get_dataset_targets(dataset_name)
        for target in targets:
            if target["column"] == target_column:
                return target.get("business_meaning", f"Target variable: {target_column}")
        return f"Target variable: {target_column}"
    
    def get_target_smart_features(self, dataset_name: str, target_column: str) -> List[str]:
        """Get smart features for a specific target variable"""
        targets = self.get_dataset_targets(dataset_name)
        for target in targets:
            if target["column"] == target_column:
                return target.get("smart_features", [])
        return []
    
    def get_all_datasets_by_type(self, dataset_type: str) -> List[str]:
        """Get all datasets of a specific type (healthcare, business, general)"""
        datasets = []
        for dataset_name, config in self.config.get("dataset_configurations", {}).items():
            if config.get("dataset_type") == dataset_type:
                datasets.append(dataset_name)
        return datasets
    
    def get_all_datasets_by_ml_task(self, ml_task_type: str) -> List[str]:
        """Get all datasets for a specific ML task type"""
        datasets = []
        for dataset_name, config in self.config.get("dataset_configurations", {}).items():
            if config.get("ml_task_type") == ml_task_type:
                datasets.append(dataset_name)
        return datasets
    
    def get_classification_datasets(self) -> List[str]:
        """Get all classification datasets"""
        return self.get_all_datasets_by_ml_task("binary_classification") + \
               self.get_all_datasets_by_ml_task("multiclass_classification")
    
    def get_regression_datasets(self) -> List[str]:
        """Get all regression datasets"""
        return self.get_all_datasets_by_ml_task("regression")
    
    def get_healthcare_datasets(self) -> List[str]:
        """Get all healthcare datasets"""
        return self.get_all_datasets_by_type("healthcare")
    
    def get_business_datasets(self) -> List[str]:
        """Get all business datasets"""
        return self.get_all_datasets_by_type("business")
    
    def get_general_datasets(self) -> List[str]:
        """Get all general datasets"""
        return self.get_all_datasets_by_type("general")
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of all datasets"""
        summary = {
            "total_datasets": len(self.config.get("dataset_configurations", {})),
            "healthcare_datasets": len(self.get_healthcare_datasets()),
            "business_datasets": len(self.get_business_datasets()),
            "general_datasets": len(self.get_general_datasets()),
            "classification_datasets": len(self.get_classification_datasets()),
            "regression_datasets": len(self.get_regression_datasets()),
            "smart_feature_categories": len(self.config.get("smart_features", {})),
            "dashboard_configurations": len(self.config.get("dashboard_configurations", {})),
            "preprocessing_pipelines": len(self.config.get("preprocessing_pipelines", {}))
        }
        return summary
    
    def get_model_recommendations(self, dataset_name: str, target_column: str = None) -> Dict[str, Any]:
        """Get model recommendations for a specific dataset and target"""
        # Try exact match first
        dataset_config = self.config.get("dataset_configurations", {}).get(dataset_name, {})
        
        # Try common name variations if exact match fails
        if not dataset_config:
            name_variations = self._get_dataset_name_variations(dataset_name)
            for variation in name_variations:
                if variation in self.config.get("dataset_configurations", {}):
                    dataset_config = self.config["dataset_configurations"][variation]
                    break
        
        ml_task_type = dataset_config.get("ml_task_type", "unknown")
        
        # Get base model recommendations
        base_models = dataset_config.get("recommended_models", [])
        
        # Get ML-specific recommendations
        ml_models = self.config.get("ml_models", {})
        
        recommendations = {
            "dataset": dataset_name,
            "target": target_column,
            "ml_task_type": ml_task_type,
            "recommended_models": base_models,
            "metrics": [],
            "visualizations": []
        }
        
        if ml_task_type in ["binary_classification", "multiclass_classification"]:
            classification_config = ml_models.get("classification", {})
            if ml_task_type == "binary_classification":
                binary_config = classification_config.get("binary", {})
                recommendations["metrics"] = binary_config.get("metrics", [])
                recommendations["visualizations"] = binary_config.get("visualizations", [])
            else:
                multiclass_config = classification_config.get("multiclass", {})
                recommendations["metrics"] = multiclass_config.get("metrics", [])
                recommendations["visualizations"] = multiclass_config.get("visualizations", [])
        
        elif ml_task_type == "regression":
            regression_config = ml_models.get("regression", {})
            recommendations["metrics"] = regression_config.get("metrics", [])
            recommendations["visualizations"] = regression_config.get("visualizations", [])
        
        return recommendations
    
    def get_preprocessing_recommendations(self, dataset_name: str) -> Dict[str, Any]:
        """Get preprocessing recommendations for a dataset"""
        # Try exact match first
        dataset_config = self.config.get("dataset_configurations", {}).get(dataset_name, {})
        
        # Try common name variations if exact match fails
        if not dataset_config:
            name_variations = self._get_dataset_name_variations(dataset_name)
            for variation in name_variations:
                if variation in self.config.get("dataset_configurations", {}):
                    dataset_config = self.config["dataset_configurations"][variation]
                    break
        
        ml_task_type = dataset_config.get("ml_task_type", "unknown")
        dataset_type = dataset_config.get("dataset_type", "unknown")
        
        # Determine preprocessing pipeline
        if dataset_type == "healthcare":
            pipeline_type = "healthcare_specific"
        elif ml_task_type in ["binary_classification", "multiclass_classification"]:
            pipeline_type = "classification"
        elif ml_task_type == "regression":
            pipeline_type = "regression"
        else:
            pipeline_type = "classification"  # Default
        
        pipeline_config = self.get_preprocessing_pipeline(pipeline_type)
        
        return {
            "dataset": dataset_name,
            "pipeline_type": pipeline_type,
            "steps": pipeline_config.get("steps", []),
            "parameters": pipeline_config.get("parameters", {}),
            "dataset_specific_needs": dataset_config.get("preprocessing_needed", [])
        }
    
    def validate_dataset_targets(self, dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that dataset matches expected target configuration"""
        dataset_config = self.config.get("dataset_configurations", {}).get(dataset_name, {})
        
        validation_result = {
            "dataset": dataset_name,
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "targets_found": [],
            "targets_missing": []
        }
        
        if not dataset_config:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Dataset {dataset_name} not found in configuration")
            return validation_result
        
        # Check expected targets
        expected_targets = [target["column"] for target in dataset_config.get("primary_targets", [])]
        
        for target in expected_targets:
            if target in df.columns:
                validation_result["targets_found"].append(target)
            else:
                validation_result["targets_missing"].append(target)
                validation_result["warnings"].append(f"Expected target {target} not found in dataset")
        
        # Check shape
        expected_shape = dataset_config.get("shape")
        if expected_shape and df.shape != tuple(expected_shape):
            validation_result["warnings"].append(
                f"Dataset shape {df.shape} doesn't match expected shape {expected_shape}"
            )
        
        return validation_result
    
    def get_smart_insights(self, dataset_name: str) -> Dict[str, Any]:
        """Get smart insights and recommendations for a dataset"""
        # Try exact match first
        dataset_config = self.config.get("dataset_configurations", {}).get(dataset_name, {})
        
        # Try common name variations if exact match fails
        if not dataset_config:
            name_variations = self._get_dataset_name_variations(dataset_name)
            for variation in name_variations:
                if variation in self.config.get("dataset_configurations", {}):
                    dataset_config = self.config["dataset_configurations"][variation]
                    break
        
        insights = {
            "dataset": dataset_name,
            "dataset_type": dataset_config.get("dataset_type", "unknown"),
            "ml_task_type": dataset_config.get("ml_task_type", "unknown"),
            "smart_functionalities": dataset_config.get("smart_functionalities", []),
            "business_value": [],
            "recommended_use_cases": [],
            "integration_suggestions": []
        }
        
        # Generate business value insights
        dataset_type = dataset_config.get("dataset_type", "unknown")
        ml_task_type = dataset_config.get("ml_task_type", "unknown")
        
        if dataset_type == "healthcare":
            insights["business_value"].extend([
                "Improve patient outcomes",
                "Reduce healthcare costs",
                "Enhance clinical decision making",
                "Optimize resource allocation"
            ])
            
            if ml_task_type in ["binary_classification", "multiclass_classification"]:
                insights["recommended_use_cases"].extend([
                    "Disease diagnosis support",
                    "Risk stratification",
                    "Treatment outcome prediction",
                    "Patient classification"
                ])
            elif ml_task_type == "regression":
                insights["recommended_use_cases"].extend([
                    "Outcome prediction",
                    "Risk scoring",
                    "Performance monitoring",
                    "Resource optimization"
                ])
        
        elif dataset_type == "business":
            insights["business_value"].extend([
                "Improve operational efficiency",
                "Optimize resource allocation",
                "Enhance performance monitoring",
                "Reduce operational costs"
            ])
        
        # Integration suggestions
        smart_features = dataset_config.get("smart_functionalities", [])
        if smart_features:
            insights["integration_suggestions"].extend([
                "Implement real-time prediction capabilities",
                "Create interactive dashboards",
                "Add alert and notification systems",
                "Integrate with existing healthcare systems"
            ])
        
        return insights

# Convenience functions for easy access
def get_target_manager(config_file: str = "smart_dataset_config.json") -> SmartDatasetTargetManager:
    """Get a SmartDatasetTargetManager instance"""
    return SmartDatasetTargetManager(config_file)

def get_dataset_targets(dataset_name: str, config_file: str = "smart_dataset_config.json") -> List[Dict[str, Any]]:
    """Quick access to dataset targets"""
    manager = get_target_manager(config_file)
    return manager.get_dataset_targets(dataset_name)

def get_recommended_models(dataset_name: str, config_file: str = "smart_dataset_config.json") -> List[str]:
    """Quick access to recommended models"""
    manager = get_target_manager(config_file)
    return manager.get_recommended_models(dataset_name)

def get_smart_functionalities(dataset_name: str, config_file: str = "smart_dataset_config.json") -> List[str]:
    """Quick access to smart functionalities"""
    manager = get_target_manager(config_file)
    return manager.get_smart_functionalities(dataset_name)

if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("SMART DATASET TARGET MANAGER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Initialize manager
    manager = SmartDatasetTargetManager()
    
    # Get summary
    summary = manager.get_dataset_summary()
    print(f"Dataset Summary: {summary}")
    
    # Example: Get targets for breast cancer dataset
    breast_cancer_targets = manager.get_dataset_targets("breast_cancer_scikit")
    print(f"\nBreast Cancer Dataset Targets: {len(breast_cancer_targets)}")
    for target in breast_cancer_targets:
        print(f"  - {target['column']}: {target['target_type']}")
    
    # Example: Get smart functionalities
    smart_features = manager.get_smart_functionalities("breast_cancer_scikit")
    print(f"\nBreast Cancer Smart Features: {len(smart_features)}")
    for feature in smart_features:
        print(f"  - {feature}")
    
    # Example: Get model recommendations
    model_recs = manager.get_model_recommendations("breast_cancer_scikit")
    print(f"\nModel Recommendations for Breast Cancer:")
    print(f"  Models: {model_recs['recommended_models']}")
    print(f"  Metrics: {model_recs['metrics']}")
    print(f"  Visualizations: {model_recs['visualizations']}")
    
    print("\n" + "=" * 80)
    print("Example usage complete!")
    print("=" * 80)
