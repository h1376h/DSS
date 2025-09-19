"""
Preprocessing Engine for Healthcare DSS
======================================

This module handles all data preprocessing, analysis, and transformation
capabilities for the Healthcare Decision Support System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """
    Advanced preprocessing engine for healthcare data analysis
    
    Provides intelligent data preprocessing, feature engineering,
    and data quality assessment capabilities.
    """
    
    def __init__(self):
        """Initialize the preprocessing engine"""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
    def detect_and_remove_data_leakage(self, dataset: pd.DataFrame, target_column: str, 
                                     correlation_threshold: float = 0.95,
                                     feature_importance_threshold: float = 0.9,
                                     auto_remove: bool = True) -> Dict[str, Any]:
        """
        Detect potential data leakage and automatically remove problematic features
        
        Args:
            dataset: The dataset to analyze
            target_column: Name of the target column
            correlation_threshold: Threshold for high correlation (default: 0.95)
            feature_importance_threshold: Threshold for suspicious feature importance (default: 0.9)
            auto_remove: Whether to automatically remove leaked features (default: True)
            
        Returns:
            Dictionary containing cleaned dataset and leakage analysis results
        """
        try:
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Separate features and target
            features = dataset.drop(columns=[target_column])
            target = dataset[target_column]
            
            leakage_analysis = {
                'leakage_detected': False,
                'leakage_type': None,
                'suspicious_features': [],
                'removed_features': [],
                'correlation_matrix': None,
                'recommendations': [],
                'confidence_score': 0,
                'risk_level': 'low',
                'cleaned_dataset': dataset.copy(),
                'features_removed_count': 0
            }
            
            # 1. Check for target column accidentally included in features
            if target_column in features.columns:
                leakage_analysis['leakage_detected'] = True
                leakage_analysis['leakage_type'] = 'target_in_features'
                leakage_analysis['suspicious_features'].append(target_column)
                leakage_analysis['recommendations'].append(f"🚨 CRITICAL: Target column '{target_column}' found in features!")
                leakage_analysis['confidence_score'] = 100
                leakage_analysis['risk_level'] = 'critical'
                
                if auto_remove:
                    # Remove target column from features
                    features = features.drop(columns=[target_column])
                    leakage_analysis['removed_features'].append(target_column)
                    leakage_analysis['features_removed_count'] += 1
                    leakage_analysis['recommendations'].append(f"Automatically removed target column '{target_column}' from features")
                else:
                    return leakage_analysis
            
            # 2. Calculate correlation matrix
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) == 0:
                leakage_analysis['recommendations'].append("No numeric features found for correlation analysis")
                return leakage_analysis
            
            # Add target to correlation analysis
            correlation_data = pd.concat([numeric_features, target], axis=1)
            correlation_matrix = correlation_data.corr()
            
            leakage_analysis['correlation_matrix'] = correlation_matrix
            
            # 3. Check for high correlations with target
            target_correlations = correlation_matrix[target_column].drop(target_column)
            high_correlations = target_correlations[abs(target_correlations) > correlation_threshold]
            
            if len(high_correlations) > 0:
                leakage_analysis['leakage_detected'] = True
                leakage_analysis['leakage_type'] = 'high_correlation'
                
                features_to_remove = []
                for feature, corr in high_correlations.items():
                    leakage_analysis['suspicious_features'].append({
                        'feature': feature,
                        'correlation': corr,
                        'risk': 'high' if abs(corr) > 0.98 else 'medium'
                    })
                    
                    if auto_remove:
                        features_to_remove.append(feature)
                
                leakage_analysis['confidence_score'] = min(95, 70 + len(high_correlations) * 10)
                leakage_analysis['risk_level'] = 'high' if any(abs(corr) > 0.98 for corr in high_correlations.values) else 'medium'
                
                leakage_analysis['recommendations'].append(f"🚨 HIGH CORRELATION DETECTED: {len(high_correlations)} features with correlation > {correlation_threshold}")
                for feature, corr in high_correlations.items():
                    leakage_analysis['recommendations'].append(f"  • {feature}: {corr:.3f}")
                
                # Automatically remove highly correlated features
                if auto_remove and features_to_remove:
                    features = features.drop(columns=features_to_remove)
                    leakage_analysis['removed_features'].extend(features_to_remove)
                    leakage_analysis['features_removed_count'] += len(features_to_remove)
                    leakage_analysis['recommendations'].append(f"Automatically removed {len(features_to_remove)} highly correlated features: {', '.join(features_to_remove)}")
                    
                    # Recalculate correlation matrix after removal
                    numeric_features = features.select_dtypes(include=[np.number])
                    if len(numeric_features.columns) > 0:
                        correlation_data = pd.concat([numeric_features, target], axis=1)
                        correlation_matrix = correlation_data.corr()
                        leakage_analysis['correlation_matrix'] = correlation_matrix
            
            # 4. Check for perfect or near-perfect correlations between features
            feature_correlations = correlation_matrix.drop(target_column, axis=0).drop(target_column, axis=1)
            perfect_correlations = []
            
            for i, col1 in enumerate(feature_correlations.columns):
                for j, col2 in enumerate(feature_correlations.columns):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = feature_correlations.loc[col1, col2]
                        if abs(corr_value) > 0.99:
                            perfect_correlations.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': corr_value
                            })
            
            if perfect_correlations:
                leakage_analysis['leakage_detected'] = True
                if leakage_analysis['leakage_type'] is None:
                    leakage_analysis['leakage_type'] = 'perfect_feature_correlation'
                leakage_analysis['confidence_score'] = max(leakage_analysis['confidence_score'], 80)
                leakage_analysis['risk_level'] = 'high' if leakage_analysis['risk_level'] == 'low' else leakage_analysis['risk_level']
                
                leakage_analysis['recommendations'].append(f"WARNING: PERFECT CORRELATION: {len(perfect_correlations)} feature pairs with correlation > 0.99")
                for pair in perfect_correlations:
                    leakage_analysis['recommendations'].append(f"  • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
                
                # Automatically remove duplicate features (keep the first one)
                if auto_remove:
                    features_to_remove = []
                    for pair in perfect_correlations:
                        # Remove the second feature in each pair (keep the first one)
                        if pair['feature2'] not in features_to_remove and pair['feature2'] in features.columns:
                            features_to_remove.append(pair['feature2'])
                    
                    if features_to_remove:
                        features = features.drop(columns=features_to_remove)
                        leakage_analysis['removed_features'].extend(features_to_remove)
                        leakage_analysis['features_removed_count'] += len(features_to_remove)
                        leakage_analysis['recommendations'].append(f"Automatically removed {len(features_to_remove)} duplicate features: {', '.join(features_to_remove)}")
            
            # 5. Check for suspicious data patterns
            suspicious_patterns = []
            
            # Check for features that are perfect predictors
            for col in numeric_features.columns:
                if pd.Series(numeric_features[col]).nunique() == len(numeric_features):
                    suspicious_patterns.append(f"Feature '{col}' has unique values for every row (potential ID column)")
            
            # Check for features with very low variance
            low_variance_features = []
            for col in numeric_features.columns:
                if numeric_features[col].std() < 1e-10:
                    low_variance_features.append(col)
            
            if low_variance_features:
                suspicious_patterns.append(f"Features with zero variance: {low_variance_features}")
            
            if suspicious_patterns:
                leakage_analysis['leakage_detected'] = True
                if leakage_analysis['leakage_type'] is None:
                    leakage_analysis['leakage_type'] = 'suspicious_patterns'
                leakage_analysis['confidence_score'] = max(leakage_analysis['confidence_score'], 60)
                leakage_analysis['risk_level'] = 'medium' if leakage_analysis['risk_level'] == 'low' else leakage_analysis['risk_level']
                
                leakage_analysis['recommendations'].extend([f"WARNING: SUSPICIOUS PATTERN: {pattern}" for pattern in suspicious_patterns])
                
                # Automatically remove suspicious features
                if auto_remove:
                    features_to_remove = []
                    for pattern in suspicious_patterns:
                        if "unique values for every row" in pattern:
                            # Extract feature name from pattern
                            feature_name = pattern.split("'")[1] if "'" in pattern else None
                            if feature_name and feature_name in features.columns:
                                features_to_remove.append(feature_name)
                        elif "zero variance" in pattern:
                            # Extract feature names from pattern
                            if "Features with zero variance:" in pattern:
                                var_features = pattern.split("Features with zero variance: ")[1]
                                var_features = var_features.strip("[]").replace("'", "").split(", ")
                                features_to_remove.extend([f.strip() for f in var_features if f.strip() in features.columns])
                    
                    if features_to_remove:
                        features = features.drop(columns=features_to_remove)
                        leakage_analysis['removed_features'].extend(features_to_remove)
                        leakage_analysis['features_removed_count'] += len(features_to_remove)
                        leakage_analysis['recommendations'].append(f"Automatically removed {len(features_to_remove)} suspicious features: {', '.join(features_to_remove)}")
            
            # 6. Generate final recommendations and create cleaned dataset
            if not leakage_analysis['leakage_detected']:
                leakage_analysis['recommendations'].append("No obvious data leakage detected")
                leakage_analysis['confidence_score'] = 85  # High confidence in clean data
            else:
                # Add mitigation recommendations
                leakage_analysis['recommendations'].append("")
                leakage_analysis['recommendations'].append("INFO: AUTOMATIC ACTIONS TAKEN:")
                
                if leakage_analysis['features_removed_count'] > 0:
                    leakage_analysis['recommendations'].append(f"  • Removed {leakage_analysis['features_removed_count']} problematic features")
                    leakage_analysis['recommendations'].append(f"  • Features removed: {', '.join(leakage_analysis['removed_features'])}")
                
                leakage_analysis['recommendations'].append("  • Dataset cleaned and ready for training")
                leakage_analysis['recommendations'].append("  • Model training can proceed safely")
            
            # Create cleaned dataset
            cleaned_dataset = pd.concat([features, target], axis=1)
            leakage_analysis['cleaned_dataset'] = cleaned_dataset
            
            # Add summary information
            leakage_analysis['original_shape'] = dataset.shape
            leakage_analysis['cleaned_shape'] = cleaned_dataset.shape
            leakage_analysis['features_removed_percentage'] = (leakage_analysis['features_removed_count'] / (dataset.shape[1] - 1)) * 100
            
            return leakage_analysis
            
        except Exception as e:
            logger.error(f"Error in data leakage detection: {str(e)}")
            return {
                'leakage_detected': False,
                'leakage_type': None,
                'suspicious_features': [],
                'removed_features': [],
                'correlation_matrix': None,
                'recommendations': [f"Error in leakage detection: {str(e)}"],
                'confidence_score': 0,
                'risk_level': 'low',
                'cleaned_dataset': dataset,
                'features_removed_count': 0,
                'original_shape': dataset.shape,
                'cleaned_shape': dataset.shape,
                'features_removed_percentage': 0,
                'error': str(e)
            }
    
    def analyze_target_column(self, dataset: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze target column to suggest appropriate task type and provide insights
        
        Args:
            dataset: The dataset containing the target column
            target_column: Name of the target column
            
        Returns:
            Dictionary containing analysis results and suggestions
        """
        try:
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            target_data = dataset[target_column]

            # Analyze data characteristics
            analysis = {
                'column_name': target_column,
                'data_type': str(target_data.dtype),
                'unique_values': target_data.nunique(),
                'total_values': len(target_data),
                'missing_values': target_data.isnull().sum(),
                'min_value': target_data.min() if pd.api.types.is_numeric_dtype(target_data) else None,
                'max_value': target_data.max() if pd.api.types.is_numeric_dtype(target_data) else None,
                'mean_value': target_data.mean() if pd.api.types.is_numeric_dtype(target_data) else None,
                'std_value': target_data.std() if pd.api.types.is_numeric_dtype(target_data) else None,
            }

            # Determine if suitable for classification
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            unique_count = target_data.nunique()
            total_count = len(target_data)
            unique_ratio = unique_count / total_count

            # Enhanced classification suitability analysis
            classification_suitable = False
            classification_reasons = []
            classification_confidence = 0

            if not is_numeric:
                # Categorical data is suitable for classification
                classification_suitable = True
                classification_confidence = 100
                classification_reasons.append("Column contains categorical data")
            else:
                # For numeric data, analyze the distribution
                if unique_count <= 2:
                    # Binary classification
                    classification_suitable = True
                    classification_confidence = 95
                    classification_reasons.append(f"Binary data ({unique_count} unique values)")
                    classification_reasons.append("Perfect for binary classification")
                elif unique_count <= 5 and unique_ratio < 0.05:
                    # Low cardinality - likely categorical
                    classification_suitable = True
                    classification_confidence = 85
                    classification_reasons.append(f"Low cardinality ({unique_count} unique values)")
                    classification_reasons.append("Likely represents distinct categories")
                elif unique_count <= 10 and unique_ratio < 0.1:
                    # Medium cardinality - might be categorical
                    classification_suitable = True
                    classification_confidence = 60
                    classification_reasons.append(f"Medium cardinality ({unique_count} unique values)")
                    classification_reasons.append("Could represent categories, but verify data meaning")
                else:
                    # High cardinality - likely continuous
                    classification_suitable = False
                    classification_confidence = 10
                    classification_reasons.append(f"High cardinality ({unique_count} unique values)")
                    classification_reasons.append("Continuous numerical data - not suitable for classification")
                    classification_reasons.append("Consider data preprocessing or use regression instead")
            
            # Check for sufficient samples per class (if classification)
            if classification_suitable and unique_count > 1:
                class_counts = target_data.value_counts()
                min_class_size = class_counts.min()
                if min_class_size < 2:
                    classification_suitable = False
                    classification_reasons.append(f"Insufficient samples per class (minimum: {min_class_size})")
                elif min_class_size < 10:
                    classification_reasons.append(f"Warning: Some classes have very few samples (minimum: {min_class_size})")
            
            # Enhanced regression suitability analysis
            regression_suitable = is_numeric and unique_count > 1
            regression_reasons = []
            regression_confidence = 0
            
            if is_numeric:
                regression_confidence = 90
                regression_reasons.append("Numeric data type")
                if unique_count > 20:
                    regression_confidence = 95
                    regression_reasons.append("High cardinality suggests continuous values")
                    regression_reasons.append("Excellent for regression analysis")
                elif unique_count > 10:
                    regression_confidence = 80
                    regression_reasons.append("Medium cardinality - suitable for regression")
                else:
                    regression_confidence = 40
                    regression_reasons.append("Low cardinality - might be categorical")
                    regression_reasons.append("Consider if values represent discrete categories")
            else:
                regression_confidence = 5
                regression_reasons.append("Non-numeric data - not suitable for regression")
                regression_reasons.append("Consider data preprocessing or use classification instead")
            
            # Generate intelligent recommendations
            recommendations = []
            primary_recommendation = None
            
            if classification_suitable and regression_suitable:
                # Both are possible - recommend based on confidence
                if classification_confidence > regression_confidence:
                    primary_recommendation = "classification"
                    recommendations.append(f"RECOMMENDED: Classification (Confidence: {classification_confidence}%)")
                    recommendations.append("Data appears to represent distinct categories")
                    recommendations.append(f"Alternative: Regression (Confidence: {regression_confidence}%)")
                else:
                    primary_recommendation = "regression"
                    recommendations.append(f"RECOMMENDED: Regression (Confidence: {regression_confidence}%)")
                    recommendations.append("Data appears to be continuous numerical values")
                    recommendations.append(f"Alternative: Classification (Confidence: {classification_confidence}%)")
            elif classification_suitable:
                primary_recommendation = "classification"
                recommendations.append(f"RECOMMENDED: Classification (Confidence: {classification_confidence}%)")
                recommendations.append("Data appears to represent distinct categories")
                if regression_confidence > 30:
                    recommendations.append(f"WARNING: Regression possible but not ideal (Confidence: {regression_confidence}%)")
            elif regression_suitable:
                primary_recommendation = "regression"
                recommendations.append(f"RECOMMENDED: Regression (Confidence: {regression_confidence}%)")
                recommendations.append("Data appears to be continuous numerical values")
                if classification_confidence > 30:
                    recommendations.append(f"WARNING: Classification possible but not ideal (Confidence: {classification_confidence}%)")
            else:
                primary_recommendation = "neither"
                recommendations.append("❌ Neither classification nor regression is suitable")
                recommendations.append("Consider data preprocessing or different target column")
                recommendations.append("SUGGESTION: Try: Converting to categorical, binning continuous values, or selecting a different target")
            
            analysis.update({
                'classification_suitable': classification_suitable,
                'classification_reasons': classification_reasons,
                'classification_confidence': classification_confidence,
                'regression_suitable': regression_suitable,
                'regression_reasons': regression_reasons,
                'regression_confidence': regression_confidence,
                'recommendations': recommendations,
                'primary_recommendation': primary_recommendation,
                'suggested_task_type': primary_recommendation if primary_recommendation != 'neither' else 'unknown'
            })
            
            return analysis

        except ValueError as e:
            logger.error(f"Error analyzing target column: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error analyzing target column: {e}")
            return {
                'error': str(e),
                'column_name': target_column,
                'suggested_task_type': 'unknown'
            }
    
    def get_preprocessing_options(self, dataset: pd.DataFrame, target_column: str, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Analyze data and suggest intelligent preprocessing options
        
        Args:
            dataset: The dataset to analyze
            target_column: Name of the target column
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary containing preprocessing options and recommendations
        """
        try:
            target_data = dataset[target_column]
            
            # Analyze data characteristics
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            unique_count = target_data.nunique()
            missing_count = target_data.isnull().sum()
            missing_ratio = missing_count / len(target_data)
            
            # Analyze feature columns
            feature_cols = [col for col in dataset.columns if col != target_column]
            numeric_features = dataset[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = dataset[feature_cols].select_dtypes(include=['object']).columns.tolist()
            
            preprocessing_options = {
                'target_preprocessing': [],
                'feature_preprocessing': [],
                'scaling_options': [],
                'encoding_options': [],
                'feature_engineering': [],
                'data_quality_issues': [],
                'recommendations': []
            }
            
            # Target preprocessing options
            if task_type == 'classification' and is_numeric and unique_count > 10:
                preprocessing_options['target_preprocessing'].extend([
                    {
                        'name': 'binary_classification',
                        'description': 'Convert to binary (above/below median)',
                        'confidence': 85,
                        'suitable': True
                    },
                    {
                        'name': 'quartile_binning',
                        'description': 'Create 4 categories (quartiles)',
                        'confidence': 75,
                        'suitable': True
                    },
                    {
                        'name': 'decile_binning',
                        'description': 'Create 10 categories (deciles)',
                        'confidence': 60,
                        'suitable': unique_count > 20
                    }
                ])
            
            # Feature preprocessing options
            if len(numeric_features) > 0:
                preprocessing_options['feature_preprocessing'].extend([
                    {
                        'name': 'handle_outliers',
                        'description': 'Detect and handle outliers using IQR method',
                        'confidence': 90,
                        'suitable': True
                    },
                    {
                        'name': 'feature_scaling',
                        'description': 'Scale numeric features for better model performance',
                        'confidence': 95,
                        'suitable': True
                    }
                ])
            
            if len(categorical_features) > 0:
                preprocessing_options['feature_preprocessing'].extend([
                    {
                        'name': 'categorical_encoding',
                        'description': 'Encode categorical variables',
                        'confidence': 90,
                        'suitable': True
                    }
                ])
            
            # Scaling options
            preprocessing_options['scaling_options'] = [
                {
                    'name': 'standard_scaler',
                    'description': 'StandardScaler (mean=0, std=1)',
                    'confidence': 90,
                    'suitable': True,
                    'best_for': 'Most algorithms, especially SVM, neural networks'
                },
                {
                    'name': 'minmax_scaler',
                    'description': 'MinMaxScaler (0-1 range)',
                    'confidence': 85,
                    'suitable': True,
                    'best_for': 'Neural networks, algorithms sensitive to scale'
                },
                {
                    'name': 'robust_scaler',
                    'description': 'RobustScaler (median and IQR)',
                    'confidence': 80,
                    'suitable': missing_ratio > 0.05,
                    'best_for': 'Data with outliers'
                },
                {
                    'name': 'normalizer',
                    'description': 'Normalizer (unit norm)',
                    'confidence': 60,
                    'suitable': len(numeric_features) > 5,
                    'best_for': 'Text data, high-dimensional data'
                }
            ]
            
            # Encoding options
            preprocessing_options['encoding_options'] = [
                {
                    'name': 'one_hot_encoding',
                    'description': 'One-hot encoding for categorical variables',
                    'confidence': 90,
                    'suitable': len(categorical_features) > 0,
                    'best_for': 'Tree-based algorithms'
                },
                {
                    'name': 'label_encoding',
                    'description': 'Label encoding for ordinal variables',
                    'confidence': 70,
                    'suitable': len(categorical_features) > 0,
                    'best_for': 'Ordinal categorical data'
                },
                {
                    'name': 'target_encoding',
                    'description': 'Target encoding for high-cardinality categoricals',
                    'confidence': 75,
                    'suitable': self._has_high_cardinality_categoricals(dataset, categorical_features),
                    'best_for': 'High-cardinality categorical features'
                }
            ]
            
            # Feature engineering options
            preprocessing_options['feature_engineering'] = [
                {
                    'name': 'polynomial_features',
                    'description': 'Generate polynomial features',
                    'confidence': 60,
                    'suitable': len(numeric_features) > 2,
                    'best_for': 'Linear models, small datasets'
                },
                {
                    'name': 'feature_interactions',
                    'description': 'Create feature interactions',
                    'confidence': 70,
                    'suitable': len(numeric_features) > 3,
                    'best_for': 'Complex relationships'
                },
                {
                    'name': 'dimensionality_reduction',
                    'description': 'PCA for dimensionality reduction',
                    'confidence': 65,
                    'suitable': len(numeric_features) > 10,
                    'best_for': 'High-dimensional data'
                }
            ]
            
            # Data quality issues
            if missing_ratio > 0:
                preprocessing_options['data_quality_issues'].append({
                    'issue': 'missing_values',
                    'severity': 'high' if missing_ratio > 0.1 else 'medium',
                    'description': f'{missing_ratio:.1%} missing values in target',
                    'solutions': ['imputation', 'drop_missing', 'indicator_variables']
                })
            
            # Generate intelligent recommendations
            recommendations = []
            if preprocessing_options['target_preprocessing']:
                recommendations.append("RECOMMENDED: Target preprocessing recommended for classification")
            if len(numeric_features) > 0:
                recommendations.append("RECOMMENDED: Feature scaling recommended for numeric features")
            if len(categorical_features) > 0:
                recommendations.append("REQUIRED: Categorical encoding required")
            if missing_ratio > 0:
                recommendations.append("WARNING: Missing value handling needed")
            
            preprocessing_options['recommendations'] = recommendations
            
            return preprocessing_options
            
        except Exception as e:
            logger.error(f"Error analyzing preprocessing options: {e}")
            return {'error': str(e)}
    
    def preprocess_data(self, dataset: pd.DataFrame, target_column: str, 
                       preprocessing_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply preprocessing transformations to the dataset
        
        Args:
            dataset: The dataset to preprocess
            target_column: Name of the target column
            preprocessing_config: Configuration for preprocessing steps
            
        Returns:
            Tuple of (features, target) after preprocessing
        """
        try:
            # Separate features and target
            # For datasets with multiple target-related columns, exclude all of them
            target_related_columns = [col for col in dataset.columns if 'target' in col.lower()]
            
            # If we're using a specific target column, exclude all other target-related columns
            if target_column in target_related_columns:
                features = dataset.drop(columns=target_related_columns)
            else:
                # If target column is not in target-related columns, just drop the specified column
                features = dataset.drop(columns=[target_column])
            
            target = dataset[target_column].copy()
            
            # Apply preprocessing based on configuration
            if preprocessing_config:
                # Handle missing values
                if 'handle_missing' in preprocessing_config:
                    features = self._handle_missing_values(features)
                
                # Apply scaling
                if 'scaling_method' in preprocessing_config:
                    features = self._apply_scaling(features, preprocessing_config['scaling_method'])
                
                # Apply encoding
                if 'encoding_method' in preprocessing_config:
                    features = self._apply_encoding(features, preprocessing_config['encoding_method'])
                
                # Apply feature engineering
                for key, value in preprocessing_config.items():
                    if key.startswith('feature_') and value:
                        features = self._apply_feature_engineering(features, key)
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def _has_high_cardinality_categoricals(self, dataset: pd.DataFrame, categorical_features: List[str]) -> bool:
        """Check if dataset has high-cardinality categorical features"""
        try:
            if len(categorical_features) == 0:
                return False
            
            for col in categorical_features:
                if col in dataset.columns:
                    unique_count = dataset[col].nunique()
                    if unique_count > 10:  # Threshold for high cardinality
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking high cardinality categoricals: {e}")
            return False
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Simple imputation for now - can be enhanced
        numeric_features = features.select_dtypes(include=[np.number]).columns
        categorical_features = features.select_dtypes(include=['object']).columns
        
        if len(numeric_features) > 0:
            imputer = SimpleImputer(strategy='mean')
            features.loc[:, numeric_features] = imputer.fit_transform(features[numeric_features])
        
        if len(categorical_features) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            features.loc[:, categorical_features] = imputer.fit_transform(features[categorical_features])
        
        return features
    
    def _apply_scaling(self, features: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply scaling to numeric features"""
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            return features
        
        if method in ['standard_scaler', 'standard']:
            scaler = StandardScaler()
        elif method in ['minmax_scaler', 'minmax']:
            scaler = MinMaxScaler()
        elif method in ['robust_scaler', 'robust']:
            scaler = RobustScaler()
        elif method in ['normalizer', 'normalize']:
            scaler = Normalizer()
        else:
            return features
        
        # Apply scaling and ensure proper dtype handling
        scaled_data = scaler.fit_transform(features[numeric_features])
        
        # Convert to DataFrame and ensure proper dtypes
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_features, index=features.index)
        
        # Ensure all scaled columns are float64 to avoid dtype warnings
        scaled_df = scaled_df.astype('float64')
        
        # Update the features DataFrame with proper dtype handling
        features = features.copy()  # Avoid SettingWithCopyWarning
        for col in numeric_features:
            features[col] = scaled_df[col].astype('float64')
        
        return features
    
    def _apply_encoding(self, features: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply encoding to categorical features"""
        categorical_features = features.select_dtypes(include=['object']).columns
        
        if len(categorical_features) == 0:
            return features
        
        if method in ['one_hot_encoding', 'onehot']:
            features = pd.get_dummies(features, columns=categorical_features)
        elif method in ['label_encoding', 'label']:
            for col in categorical_features:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
        
        return features
    
    def _apply_feature_engineering(self, features: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        if method == 'polynomial_features':
            # Simple polynomial features - can be enhanced
            numeric_features = features.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 1:
                # Create interaction features
                for i, col1 in enumerate(numeric_features):
                    for col2 in numeric_features[i+1:]:
                        features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
        
        elif method == 'dimensionality_reduction':
            numeric_features = features.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 10:
                pca = PCA(n_components=min(10, len(numeric_features)))
                pca_features = pca.fit_transform(features[numeric_features])
                pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])
                features = pd.concat([features.drop(columns=numeric_features), pca_df], axis=1)
        
        return features
