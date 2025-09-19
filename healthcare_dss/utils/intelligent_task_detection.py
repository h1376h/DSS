"""
Intelligent Task Type Detection Module
=====================================

This module provides robust task type detection that doesn't rely on hardcoded logic
but instead uses statistical and data analysis techniques to determine if a target
variable represents classification or regression.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from healthcare_dss.utils.debug_manager import debug_manager


def detect_task_type_intelligently(y: np.ndarray, feature_name: str = None) -> Tuple[str, float, Dict[str, Any]]:
    """
    Intelligently detect task type using statistical analysis without hardcoded logic.
    
    Args:
        y: Target variable array
        feature_name: Optional name of the feature for debugging
        
    Returns:
        Tuple of (task_type, confidence, analysis_details)
    """
    
    analysis_details = {
        'feature_name': feature_name,
        'data_type': str(y.dtype),
        'unique_values': len(np.unique(y)),
        'unique_values_list': np.unique(y).tolist(),
        'is_numeric': pd.api.types.is_numeric_dtype(y),
        'is_categorical': isinstance(y.dtype, pd.CategoricalDtype),
        'is_object': pd.api.types.is_object_dtype(y)
    }
    
    debug_manager.log_debug(f"Intelligent task type detection for {feature_name}:")
    debug_manager.log_debug(f"  - Data type: {y.dtype}")
    debug_manager.log_debug(f"  - Unique values: {len(np.unique(y))}")
    debug_manager.log_debug(f"  - Unique values: {np.unique(y)}")
    
    # Convert to numpy array if not already
    y_array = np.array(y)
    
    # Get unique values
    unique_values = np.unique(y_array)
    n_unique = len(unique_values)
    
    # Rule 1: If only 1 unique value, it's not a valid target (edge case)
    if n_unique == 1:
        debug_manager.log_debug(f"  - Only 1 unique value, invalid target")
        return 'invalid', 0.0, analysis_details
    
    # Rule 2: If 2 unique values, it's binary classification (high confidence)
    if n_unique == 2:
        debug_manager.log_debug(f"  - Binary classification detected (2 unique values)")
        analysis_details['detection_reason'] = 'binary_classification'
        analysis_details['confidence_factors'] = ['exactly_two_values']
        return 'classification', 0.95, analysis_details
    
    # Rule 3: Check if values are non-numeric strings (categorical)
    if not pd.api.types.is_numeric_dtype(y_array):
        debug_manager.log_debug(f"  - Non-numeric data, classification detected")
        analysis_details['detection_reason'] = 'non_numeric_data'
        analysis_details['confidence_factors'] = ['non_numeric_type']
        return 'classification', 0.9, analysis_details
    
    # Rule 4: For numeric data, use statistical analysis
    confidence_factors = []
    classification_score = 0.0
    regression_score = 0.0
    
    # Factor 1: Number of unique values relative to total samples
    total_samples = len(y_array)
    unique_ratio = n_unique / total_samples
    
    if unique_ratio <= 0.05:  # Less than 5% unique values (stricter threshold)
        classification_score += 0.4
        confidence_factors.append('very_low_unique_ratio')
        debug_manager.log_debug(f"  - Very low unique ratio: {unique_ratio:.3f}")
    elif unique_ratio <= 0.1:  # Less than 10% unique values
        classification_score += 0.2
        confidence_factors.append('low_unique_ratio')
        debug_manager.log_debug(f"  - Low unique ratio: {unique_ratio:.3f}")
    elif unique_ratio >= 0.3:  # More than 30% unique values (lowered threshold)
        regression_score += 0.4
        confidence_factors.append('high_unique_ratio')
        debug_manager.log_debug(f"  - High unique ratio: {unique_ratio:.3f}")
    elif unique_ratio >= 0.2:  # More than 20% unique values (lowered threshold)
        regression_score += 0.2
        confidence_factors.append('moderate_unique_ratio')
        debug_manager.log_debug(f"  - Moderate unique ratio: {unique_ratio:.3f}")
    
    # Factor 2: Check if values are integers (more likely classification)
    if np.allclose(unique_values, np.round(unique_values), atol=1e-10):
        classification_score += 0.2
        confidence_factors.append('integer_values')
        debug_manager.log_debug(f"  - Integer values detected")
    
    # Factor 3: Check if values are consecutive integers starting from 0 or 1
    if np.allclose(unique_values, np.round(unique_values), atol=1e-10):
        sorted_values = np.sort(unique_values)
        if len(sorted_values) <= 20:  # Only check for small number of classes
            # Check if values start from 0 or 1 and are consecutive
            if (sorted_values[0] in [0, 1] and 
                np.allclose(np.diff(sorted_values), 1, atol=1e-10)):
                classification_score += 0.3
                confidence_factors.append('consecutive_integers')
                debug_manager.log_debug(f"  - Consecutive integers starting from {sorted_values[0]}")
    
    # Factor 4: Check distribution characteristics
    if n_unique <= 20:  # Small number of unique values
        # Check if values are evenly distributed (more likely classification)
        # Handle negative values for bincount by shifting to non-negative range
        if np.min(y_array) >= 0:
            value_counts = np.bincount(y_array.astype(int))
        else:
            # Shift values to non-negative range for bincount
            min_val = np.min(y_array)
            shifted_array = y_array - min_val
            value_counts = np.bincount(shifted_array.astype(int))
        
        if len(value_counts) > 0:
            # Calculate coefficient of variation of counts
            if np.mean(value_counts) > 0:
                cv_counts = np.std(value_counts) / np.mean(value_counts)
                if cv_counts < 0.5:  # Low variation in counts
                    classification_score += 0.2
                    confidence_factors.append('even_distribution')
                    debug_manager.log_debug(f"  - Even distribution (CV: {cv_counts:.3f})")
    
    # Factor 5: Check for common classification patterns
    if n_unique <= 10:  # Very small number of classes
        classification_score += 0.2
        confidence_factors.append('few_classes')
        debug_manager.log_debug(f"  - Few classes ({n_unique})")
    
    # Factor 6: Check for continuous distribution patterns
    if n_unique > 50:  # Many unique values
        # Check if values follow a continuous distribution
        if np.issubdtype(y_array.dtype, np.floating):
            # Check for continuous-like patterns
            sorted_values = np.sort(unique_values)
            gaps = np.diff(sorted_values)
            if len(gaps) > 0:
                # If gaps are relatively small and consistent, likely continuous
                gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                if gap_cv < 1.0:  # Low variation in gaps
                    regression_score += 0.3
                    confidence_factors.append('continuous_distribution')
                    debug_manager.log_debug(f"  - Continuous distribution pattern")
    
    # Factor 6b: Check for truly continuous data (many unique values with float type)
    if n_unique > 50 and np.issubdtype(y_array.dtype, np.floating):  # Lowered threshold
        # Check if the data has a continuous distribution
        sorted_values = np.sort(unique_values)
        if len(sorted_values) > 1:
            # Calculate the range and check if values are spread across it
            data_range = sorted_values[-1] - sorted_values[0]
            if data_range > 0:
                # Check if values are relatively evenly distributed
                gaps = np.diff(sorted_values)
                if len(gaps) > 0:
                    mean_gap = np.mean(gaps)
                    gap_std = np.std(gaps)
                    if mean_gap > 0 and gap_std / mean_gap < 2.0:  # Relatively consistent gaps
                        regression_score += 0.5  # Increased weight
                        confidence_factors.append('high_cardinality_continuous')
                        debug_manager.log_debug(f"  - High cardinality continuous data")
    
    # Factor 7: Domain-specific heuristics (without hardcoding specific values)
    # Check if the feature name suggests classification
    if feature_name:
        classification_keywords = ['sex', 'gender', 'type', 'category', 'class', 'label', 'status', 'group']
        if any(keyword in feature_name.lower() for keyword in classification_keywords):
            classification_score += 0.2
            confidence_factors.append('semantic_hint')
            debug_manager.log_debug(f"  - Semantic hint from feature name: {feature_name}")
    
    # Determine final task type based on scores
    total_classification_score = classification_score
    total_regression_score = regression_score
    
    debug_manager.log_debug(f"  - Classification score: {total_classification_score:.3f}")
    debug_manager.log_debug(f"  - Regression score: {total_regression_score:.3f}")
    
    # Decision logic
    if total_classification_score > total_regression_score:
        confidence = min(0.9, 0.5 + total_classification_score)
        task_type = 'classification'
        debug_manager.log_debug(f"  - Decision: Classification (confidence: {confidence:.3f})")
    else:
        confidence = min(0.9, 0.5 + total_regression_score)
        task_type = 'regression'
        debug_manager.log_debug(f"  - Decision: Regression (confidence: {confidence:.3f})")
    
    analysis_details.update({
        'detection_reason': 'statistical_analysis',
        'confidence_factors': confidence_factors,
        'classification_score': total_classification_score,
        'regression_score': total_regression_score,
        'unique_ratio': unique_ratio,
        'final_confidence': confidence
    })
    
    return task_type, confidence, analysis_details


def validate_task_type_with_data(y_true: np.ndarray, y_pred: np.ndarray, 
                                detected_task_type: str, confidence: float) -> Tuple[str, float]:
    """
    Validate the detected task type by analyzing prediction patterns.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        detected_task_type: Initially detected task type
        confidence: Initial confidence
        
    Returns:
        Tuple of (validated_task_type, updated_confidence)
    """
    
    debug_manager.log_debug(f"Validating task type {detected_task_type} with prediction analysis")
    
    # Check prediction patterns
    unique_true = len(np.unique(y_true))
    unique_pred = len(np.unique(y_pred))
    
    debug_manager.log_debug(f"  - Unique true values: {unique_true}")
    debug_manager.log_debug(f"  - Unique predicted values: {unique_pred}")
    
    # If predictions have very few unique values but true values have many, 
    # it might be a classification model predicting regression
    if detected_task_type == 'regression' and unique_pred <= 10 and unique_true > 20:
        debug_manager.log_debug(f"  - Prediction pattern suggests classification model")
        return 'classification', confidence * 0.8
    
    # If predictions have many unique values but true values have few,
    # it might be a regression model predicting classification
    if detected_task_type == 'classification' and unique_pred > 20 and unique_true <= 10:
        debug_manager.log_debug(f"  - Prediction pattern suggests regression model")
        return 'regression', confidence * 0.8
    
    return detected_task_type, confidence
