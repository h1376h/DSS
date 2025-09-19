#!/usr/bin/env python3
"""
Intelligent Data Analysis Utilities
Provides smart data analysis functions for the Healthcare DSS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IntelligentDataAnalyzer:
    """Intelligent data analysis utilities for healthcare DSS"""
    
    @staticmethod
    def detect_task_type(target_data: pd.Series, dataset_name: str = None) -> Dict[str, Any]:
        """
        Intelligently detect the most appropriate ML task type for a target variable
        
        Args:
            target_data: The target column data
            dataset_name: Optional dataset name for context
            
        Returns:
            Dictionary containing task type recommendations and confidence scores
        """
        analysis = {
            'target_type': 'unknown',
            'confidence': 0,
            'reasons': [],
            'recommendations': [],
            'data_characteristics': {}
        }
        
        # Basic data characteristics
        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        unique_values = target_data.nunique()
        total_values = len(target_data)
        unique_ratio = unique_values / total_values if total_values > 0 else 0
        missing_count = target_data.isnull().sum()
        missing_ratio = missing_count / total_values if total_values > 0 else 0
        
        analysis['data_characteristics'] = {
            'is_numeric': is_numeric,
            'unique_values': unique_values,
            'total_values': total_values,
            'unique_ratio': unique_ratio,
            'missing_count': missing_count,
            'missing_ratio': missing_ratio,
            'data_type': str(target_data.dtype)
        }
        
        # Classification analysis
        classification_score = 0
        classification_reasons = []
        
        if not is_numeric:
            # Non-numeric data is almost always classification
            classification_score = 95
            classification_reasons.append("Non-numeric data type")
        elif unique_values == 2:
            # Binary classification
            classification_score = 100
            classification_reasons.append("Binary classification (2 unique values)")
        elif unique_values <= 10 and unique_ratio < 0.05:
            # Very low cardinality categorical (strict threshold)
            classification_score = 90
            classification_reasons.append(f"Very low cardinality ({unique_values} unique values)")
            classification_reasons.append(f"Very low unique ratio ({unique_ratio:.1%})")
        elif unique_values <= 20 and unique_ratio < 0.1:
            # Low cardinality categorical (stricter threshold)
            classification_score = 75
            classification_reasons.append(f"Low cardinality ({unique_values} unique values)")
            classification_reasons.append(f"Low unique ratio ({unique_ratio:.1%})")
        elif unique_values <= 50 and unique_ratio < 0.15:
            # Medium cardinality categorical (very strict threshold)
            classification_score = 60
            classification_reasons.append(f"Medium cardinality ({unique_values} unique values)")
            classification_reasons.append(f"Moderate unique ratio ({unique_ratio:.1%})")
        
        # Regression analysis
        regression_score = 0
        regression_reasons = []
        
        if is_numeric and unique_ratio > 0.3:
            # High cardinality numeric (lowered threshold)
            regression_score = 95
            regression_reasons.append("High cardinality numeric data")
            regression_reasons.append(f"High unique ratio ({unique_ratio:.1%})")
        elif is_numeric and unique_values > 100:
            # Many unique numeric values (lowered threshold)
            regression_score = 90
            regression_reasons.append("Many unique numeric values")
            regression_reasons.append(f"Cardinality: {unique_values}")
        elif is_numeric and unique_ratio > 0.2:
            # Moderate cardinality numeric (lowered threshold)
            regression_score = 80
            regression_reasons.append("Moderate cardinality numeric data")
            regression_reasons.append(f"Unique ratio: {unique_ratio:.1%}")
        elif is_numeric and unique_values > 50:
            # Moderate number of unique values
            regression_score = 70
            regression_reasons.append("Moderate number of unique values")
            regression_reasons.append(f"Cardinality: {unique_values}")
        
        # Additional domain-specific analysis
        if dataset_name:
            # Diabetes dataset specific logic
            if 'diabetes' in dataset_name.lower():
                if is_numeric and unique_values > 50:
                    regression_score += 20  # Boost regression score for diabetes
                    regression_reasons.append("Diabetes dataset typically uses continuous target")
            
            # Other healthcare datasets
            if any(keyword in dataset_name.lower() for keyword in ['expenditure', 'cost', 'price', 'amount', 'value', 'score', 'rating']):
                if is_numeric:
                    regression_score += 15
                    regression_reasons.append("Financial/continuous metrics suggest regression")
        
        # Statistical analysis for better decision making
        if is_numeric and total_values > 0:
            # Check if values are evenly distributed (suggests continuous)
            value_counts = target_data.value_counts()
            max_count = value_counts.max()
            min_count = value_counts.min()
            count_ratio = min_count / max_count if max_count > 0 else 0
            
            if count_ratio > 0.1:  # Relatively even distribution
                regression_score += 10
                regression_reasons.append("Even distribution suggests continuous data")
            elif count_ratio < 0.01:  # Very uneven distribution
                classification_score += 10
                classification_reasons.append("Uneven distribution suggests categorical data")
        
        # Determine primary recommendation
        if classification_score > regression_score:
            analysis['target_type'] = 'classification'
            analysis['confidence'] = classification_score
            analysis['reasons'] = classification_reasons
            analysis['recommendations'].append("Use classification algorithms")
            analysis['recommendations'].append("Consider label encoding for categorical targets")
        elif regression_score > classification_score:
            analysis['target_type'] = 'regression'
            analysis['confidence'] = regression_score
            analysis['reasons'] = regression_reasons
            analysis['recommendations'].append("Use regression algorithms")
            analysis['recommendations'].append("Consider feature scaling")
        else:
            # Tie or both suitable
            analysis['target_type'] = 'mixed'
            analysis['confidence'] = max(classification_score, regression_score)
            analysis['reasons'] = classification_reasons + regression_reasons
            analysis['recommendations'].append("Both classification and regression are suitable")
            analysis['recommendations'].append("Choose based on business objective")
        
        # Add dataset-specific insights if available
        if dataset_name:
            try:
                from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
                smart_manager = SmartDatasetTargetManager()
                smart_targets = smart_manager.get_dataset_targets(dataset_name)
                
                # Find matching target in smart configuration
                for smart_target in smart_targets:
                    if smart_target['column'] == target_data.name:
                        analysis['smart_insights'] = {
                            'business_meaning': smart_target.get('business_meaning', ''),
                            'target_type': smart_target.get('target_type', ''),
                            'smart_features': smart_target.get('smart_features', [])
                        }
                        break
            except Exception as e:
                logger.debug(f"Could not get smart insights: {e}")
        
        return analysis
    
    @staticmethod
    def get_missing_value_strategies(target_data: pd.Series, task_type: str = None) -> Dict[str, Any]:
        """
        Get intelligent missing value handling strategies
        
        Args:
            target_data: The target column data
            task_type: Optional task type for context
            
        Returns:
            Dictionary containing missing value handling strategies
        """
        strategies = {
            'has_missing': False,
            'missing_count': 0,
            'missing_ratio': 0.0,
            'recommended_strategies': [],
            'strategy_details': {}
        }
        
        missing_count = target_data.isnull().sum()
        total_count = len(target_data)
        missing_ratio = missing_count / total_count if total_count > 0 else 0
        
        strategies['missing_count'] = missing_count
        strategies['missing_ratio'] = missing_ratio
        strategies['has_missing'] = missing_count > 0
        
        if missing_count == 0:
            strategies['recommended_strategies'].append("No missing values - no action needed")
            return strategies
        
        # Determine data type
        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        is_categorical = not is_numeric or target_data.nunique() < 20
        
        # Strategy recommendations based on data characteristics
        if is_numeric:
            if missing_ratio < 0.05:  # Less than 5% missing
                strategies['recommended_strategies'].append("Mean imputation (low missing ratio)")
                strategies['strategy_details']['mean'] = {
                    'description': 'Replace missing values with the mean',
                    'suitable_for': 'Numeric data with low missing ratio',
                    'implementation': 'target_data.fillna(target_data.mean())',
                    'confidence': 90
                }
            elif missing_ratio < 0.2:  # Less than 20% missing
                strategies['recommended_strategies'].append("Median imputation (moderate missing ratio)")
                strategies['strategy_details']['median'] = {
                    'description': 'Replace missing values with the median',
                    'suitable_for': 'Numeric data with moderate missing ratio',
                    'implementation': 'target_data.fillna(target_data.median())',
                    'confidence': 85
                }
            else:  # High missing ratio
                strategies['recommended_strategies'].append("Advanced imputation (high missing ratio)")
                strategies['strategy_details']['advanced'] = {
                    'description': 'Use KNN or iterative imputation',
                    'suitable_for': 'Numeric data with high missing ratio',
                    'implementation': 'from sklearn.impute import KNNImputer',
                    'confidence': 80
                }
        
        if is_categorical:
            strategies['recommended_strategies'].append("Mode imputation (categorical data)")
            strategies['strategy_details']['mode'] = {
                'description': 'Replace missing values with the most frequent value',
                'suitable_for': 'Categorical data',
                'implementation': 'target_data.fillna(target_data.mode()[0])',
                'confidence': 90
            }
        
        # Task-specific strategies
        if task_type == 'classification':
            strategies['recommended_strategies'].append("Create 'Missing' category")
            strategies['strategy_details']['missing_category'] = {
                'description': 'Treat missing values as a separate category',
                'suitable_for': 'Classification tasks',
                'implementation': 'target_data.fillna("Missing")',
                'confidence': 75
            }
        
        # Always include drop option
        strategies['recommended_strategies'].append("Drop missing values")
        strategies['strategy_details']['drop'] = {
            'description': 'Remove rows with missing values',
            'suitable_for': 'All data types',
            'implementation': 'target_data.dropna()',
            'confidence': 60
        }
        
        return strategies
    
    @staticmethod
    def get_preprocessing_recommendations(target_data: pd.Series, task_type: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        Get intelligent preprocessing recommendations
        
        Args:
            target_data: The target column data
            task_type: The selected ML task type
            dataset_name: Optional dataset name for context
            
        Returns:
            Dictionary containing preprocessing recommendations
        """
        recommendations = {
            'task_type': task_type,
            'preprocessing_steps': [],
            'encoding_needed': False,
            'scaling_needed': False,
            'balancing_needed': False,
            'details': {}
        }
        
        # Basic data analysis
        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        unique_values = target_data.nunique()
        is_categorical = not is_numeric or unique_values < 20
        
        # Task-specific preprocessing
        if task_type == 'classification':
            if is_categorical:
                recommendations['preprocessing_steps'].append("Label encoding")
                recommendations['encoding_needed'] = True
                recommendations['details']['label_encoding'] = {
                    'description': 'Convert categorical labels to numeric',
                    'implementation': 'from sklearn.preprocessing import LabelEncoder'
                }
            
            # Check for class imbalance
            if unique_values > 1:
                value_counts = target_data.value_counts()
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 3:  # Significant imbalance
                    recommendations['preprocessing_steps'].append("Class balancing")
                    recommendations['balancing_needed'] = True
                    recommendations['details']['balancing'] = {
                        'description': 'Balance class distribution',
                        'imbalance_ratio': imbalance_ratio,
                        'implementation': 'from imblearn.over_sampling import SMOTE'
                    }
        
        elif task_type == 'regression':
            if is_numeric:
                recommendations['preprocessing_steps'].append("Feature scaling")
                recommendations['scaling_needed'] = True
                recommendations['details']['scaling'] = {
                    'description': 'Scale numeric features',
                    'implementation': 'from sklearn.preprocessing import StandardScaler'
                }
            
            # Check for outliers
            if is_numeric and unique_values > 10:
                q1 = target_data.quantile(0.25)
                q3 = target_data.quantile(0.75)
                iqr = q3 - q1
                outlier_threshold = 1.5 * iqr
                outliers = ((target_data < q1 - outlier_threshold) | (target_data > q3 + outlier_threshold)).sum()
                
                if outliers > len(target_data) * 0.05:  # More than 5% outliers
                    recommendations['preprocessing_steps'].append("Outlier handling")
                    recommendations['details']['outliers'] = {
                        'description': 'Handle outliers in target variable',
                        'outlier_count': outliers,
                        'implementation': 'Consider outlier removal or transformation'
                    }
        
        # Dataset-specific recommendations
        if dataset_name:
            try:
                from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
                smart_manager = SmartDatasetTargetManager()
                preprocessing_recs = smart_manager.get_preprocessing_recommendations(dataset_name)
                
                if preprocessing_recs.get('steps'):
                    recommendations['smart_steps'] = preprocessing_recs['steps']
                    recommendations['smart_parameters'] = preprocessing_recs.get('parameters', {})
            except Exception as e:
                logger.debug(f"Could not get smart preprocessing recommendations: {e}")
        
        return recommendations
    
    @staticmethod
    def detect_and_fix_categorical_data(data: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Detect if categorical data is stored as continuous values and fix it
        
        Args:
            data: The data series to analyze
            column_name: Name of the column for context
            
        Returns:
            Dictionary with analysis results and fixed data
        """
        analysis = {
            'original_data': data.copy(),
            'is_categorical': False,
            'needs_conversion': False,
            'conversion_method': None,
            'fixed_data': data.copy(),
            'unique_values': data.nunique(),
            'data_type': str(data.dtype),
            'conversion_details': {}
        }
        
        # Check if this looks like categorical data stored as continuous
        unique_count = data.nunique()
        total_count = len(data)
        unique_ratio = unique_count / total_count
        
        # Heuristics for detecting categorical data stored as continuous
        categorical_indicators = []
        
        # 1. Low cardinality (few unique values relative to total)
        if unique_ratio < 0.1:  # Less than 10% unique values
            categorical_indicators.append(f"Low cardinality: {unique_count}/{total_count} unique values")
        
        # 2. Check if values are close to common categorical patterns
        if unique_count <= 10:  # Small number of unique values
            # Check if values cluster around common categorical patterns
            value_counts = data.value_counts()
            top_values = value_counts.head(5)
            
            # Check for binary patterns (0/1, -1/1, etc.)
            if unique_count == 2:
                values = sorted(data.unique())
                if abs(values[0] - values[1]) > 0.1:  # Values are not close to 0/1
                    categorical_indicators.append(f"Binary-like data with values: {values}")
            
            # Check for normalized categorical data (values around 0 with small variations)
            if data.min() < 0.1 and data.max() > -0.1:  # Values close to 0
                categorical_indicators.append("Values clustered around 0 (possibly normalized)")
            
            # Check for integer-like patterns in float data
            if data.dtype in ['float64', 'float32']:
                # Check if values are close to integers
                rounded_values = data.round().unique()
                if len(rounded_values) <= unique_count * 1.5:  # Most values are close to integers
                    categorical_indicators.append("Float values close to integers")
        
        # 3. Check for common categorical column names
        if column_name:
            categorical_names = ['sex', 'gender', 'male', 'female', 'category', 'type', 'class', 'label']
            if any(name in column_name.lower() for name in categorical_names):
                categorical_indicators.append(f"Column name '{column_name}' suggests categorical data")
        
        # Determine if this is categorical data that needs conversion
        if len(categorical_indicators) >= 2 or (len(categorical_indicators) >= 1 and unique_count <= 5):
            analysis['is_categorical'] = True
            analysis['needs_conversion'] = True
            
            # Determine conversion method
            if unique_count == 2:
                # Binary categorical data
                analysis['conversion_method'] = 'binary'
                values = sorted(data.unique())
                analysis['conversion_details'] = {
                    'original_values': values,
                    'mapped_values': [0, 1],
                    'mapping': {values[0]: 0, values[1]: 1}
                }
                
                # Apply conversion
                analysis['fixed_data'] = data.map({values[0]: 0, values[1]: 1})
                
            elif unique_count <= 10:
                # Multi-class categorical data
                analysis['conversion_method'] = 'multiclass'
                unique_vals = sorted(data.unique())
                mapping = {val: i for i, val in enumerate(unique_vals)}
                analysis['conversion_details'] = {
                    'original_values': unique_vals,
                    'mapped_values': list(range(len(unique_vals))),
                    'mapping': mapping
                }
                
                # Apply conversion
                analysis['fixed_data'] = data.map(mapping)
            
            else:
                # Too many unique values, might not be categorical
                analysis['needs_conversion'] = False
                analysis['conversion_method'] = None
        
        analysis['categorical_indicators'] = categorical_indicators
        return analysis
    
    @staticmethod
    def validate_task_type_selection(target_data: pd.Series, selected_task_type: str) -> Dict[str, Any]:
        """
        Validate if the selected task type is appropriate for the target data
        
        Args:
            target_data: The target column data
            selected_task_type: The selected task type
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            'is_valid': True,
            'confidence': 0,
            'warnings': [],
            'suggestions': []
        }
        
        # Get intelligent task type detection
        analysis = IntelligentDataAnalyzer.detect_task_type(target_data)
        detected_type = analysis['target_type']
        confidence = analysis['confidence']
        
        # Validate selection
        if selected_task_type == 'classification':
            if detected_type == 'regression':
                validation['is_valid'] = False
                validation['warnings'].append("Target appears to be continuous (regression) but classification was selected")
                validation['suggestions'].append("Consider using regression instead")
            elif detected_type == 'mixed':
                validation['confidence'] = confidence
                validation['warnings'].append("Target could be either classification or regression")
                validation['suggestions'].append("Classification is acceptable but regression might be more appropriate")
            else:
                validation['confidence'] = confidence
        
        elif selected_task_type == 'regression':
            if detected_type == 'classification':
                validation['is_valid'] = False
                validation['warnings'].append("Target appears to be categorical (classification) but regression was selected")
                validation['suggestions'].append("Consider using classification instead")
            elif detected_type == 'mixed':
                validation['confidence'] = confidence
                validation['warnings'].append("Target could be either classification or regression")
                validation['suggestions'].append("Regression is acceptable but classification might be more appropriate")
            else:
                validation['confidence'] = confidence
        
        return validation

# Convenience functions
def detect_task_type(target_data: pd.Series, dataset_name: str = None) -> Dict[str, Any]:
    """Convenience function for task type detection"""
    return IntelligentDataAnalyzer.detect_task_type(target_data, dataset_name)

def get_missing_value_strategies(target_data: pd.Series, task_type: str = None) -> Dict[str, Any]:
    """Convenience function for missing value strategies"""
    return IntelligentDataAnalyzer.get_missing_value_strategies(target_data, task_type)

def get_preprocessing_recommendations(target_data: pd.Series, task_type: str, dataset_name: str = None) -> Dict[str, Any]:
    """Convenience function for preprocessing recommendations"""
    return IntelligentDataAnalyzer.get_preprocessing_recommendations(target_data, task_type, dataset_name)

def validate_task_type_selection(target_data: pd.Series, selected_task_type: str) -> Dict[str, Any]:
    """Convenience function for task type validation"""
    return IntelligentDataAnalyzer.validate_task_type_selection(target_data, selected_task_type)
