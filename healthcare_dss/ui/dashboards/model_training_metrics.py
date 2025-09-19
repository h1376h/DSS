"""
Model Training Metrics Module
============================

This module handles metrics calculation and evaluation:
- Comprehensive metrics calculation for classification and regression
- Feature importance analysis
- Cross-validation score analysis
- Data leakage detection in metrics
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_percentage_error, explained_variance_score

from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics based on task type"""
    
    metrics = {}
    
    try:
        # Determine task type dynamically using intelligent detection
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_true)
        
        debug_manager.log_debug(f"Metrics calculation - Intelligent task type detection:")
        debug_manager.log_debug(f"  - Input task_type: {task_type}")
        debug_manager.log_debug(f"  - Detected task_type: {detected_task_type}")
        debug_manager.log_debug(f"  - Confidence: {confidence:.3f}")
        
        # Always use detected task type if confidence is reasonable, otherwise use input
        if confidence > 0.5:  # Lower threshold to be more aggressive
            task_type = detected_task_type
            debug_manager.log_debug(f"  - Using detected task type: {task_type} (confidence: {confidence:.3f})")
        else:
            debug_manager.log_debug(f"  - Using input task type: {task_type} (low confidence: {confidence:.3f})")
        
        # Normalize task type input
        if task_type in ['continuous', 'regression']:
            task_type = 'regression'
        elif task_type in ['discrete', 'classification']:
            task_type = 'classification'
        
        debug_manager.log_debug(f"Calculating metrics for task_type: {task_type}")
        debug_manager.log_debug(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        debug_manager.log_debug(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        debug_manager.log_debug(f"y_true sample: {y_true[:5]}")
        debug_manager.log_debug(f"y_pred sample: {y_pred[:5]}")
        debug_manager.log_debug(f"y_true min/max: {np.min(y_true)}/{np.max(y_true)}")
        debug_manager.log_debug(f"y_pred min/max: {np.min(y_pred)}/{np.max(y_pred)}")
        
        # Check prediction distribution
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        debug_manager.log_debug(f"Unique true values: {unique_true}")
        debug_manager.log_debug(f"Unique pred values: {unique_pred}")
        debug_manager.log_debug(f"Prediction distribution: {np.bincount(y_pred.astype(int)) if len(unique_pred) <= 10 else 'Too many unique values'}")
        
        # Check if predictions are all the same
        if len(unique_pred) == 1:
            debug_manager.log_debug(f"WARNING: All predictions are the same value: {unique_pred[0]}")
        if len(unique_true) == 1:
            debug_manager.log_debug(f"WARNING: All true values are the same: {unique_true[0]}")
        
        unique_values = len(np.unique(y_true))
    
        if task_type == "classification":
            # Classification metrics using intelligent task detection
            debug_manager.log_debug(f"Calculating classification metrics with intelligent detection")
            
            # Initialize discrete predictions variable
            y_pred_discrete = None
            
            # Only proceed if we're confident this is classification
            if detected_task_type == 'classification' and confidence > 0.5:
                try:
                    # Convert predictions to discrete classes
                    unique_true_values = np.unique(y_true)
                    debug_manager.log_debug(f"Unique true values: {unique_true_values}")
                    
                    # Convert continuous predictions to discrete classes
                    y_pred_discrete = np.zeros_like(y_pred)
                    for i, pred_val in enumerate(y_pred):
                        # Find closest true value
                        closest_idx = np.argmin(np.abs(unique_true_values - pred_val))
                        y_pred_discrete[i] = unique_true_values[closest_idx]
                    
                    debug_manager.log_debug(f"Converted predictions to discrete classes")
                    debug_manager.log_debug(f"y_pred_discrete unique values: {np.unique(y_pred_discrete)}")
                    
                    # Calculate accuracy with discrete predictions
                    correct_predictions = np.sum(y_true == y_pred_discrete)
                    total_predictions = len(y_true)
                    metrics['accuracy'] = float(correct_predictions / total_predictions) if total_predictions > 0 else 0.0
                    
                    # Get unique classes
                    unique_classes = np.unique(np.concatenate([y_true, y_pred_discrete]))
                    debug_manager.log_debug(f"Unique classes: {unique_classes}")
                    
                    # Use discrete predictions for all metrics
                    y_pred_for_metrics = y_pred_discrete
                    
                    # Calculate precision, recall, and F1 for each class
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for class_label in unique_classes:
                        # True positives, false positives, false negatives
                        tp = np.sum((y_true == class_label) & (y_pred_for_metrics == class_label))
                        fp = np.sum((y_true != class_label) & (y_pred_for_metrics == class_label))
                        fn = np.sum((y_true == class_label) & (y_pred_for_metrics != class_label))
                    
                        # Calculate precision, recall, F1
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                    
                    # Calculate weighted averages
                    class_counts = np.array([np.sum(y_true == cls) for cls in unique_classes])
                    total_samples = np.sum(class_counts)
                    
                    if total_samples > 0:
                        metrics['precision'] = float(np.average(precisions, weights=class_counts))
                        metrics['recall'] = float(np.average(recalls, weights=class_counts))
                        metrics['f1_score'] = float(np.average(f1_scores, weights=class_counts))
                    else:
                        metrics['precision'] = 0.0
                        metrics['recall'] = 0.0
                        metrics['f1_score'] = 0.0
                    
                    debug_manager.log_debug(f"Classification metrics calculated successfully")
                    debug_manager.log_debug(f"Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1_score']}")
                    
                except Exception as e:
                    debug_manager.log_debug(f"Error calculating classification metrics: {str(e)}")
                    # Set default values
                    metrics['accuracy'] = 0.0
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1_score'] = 0.0
                    metrics['error'] = f"Failed to calculate classification metrics: {str(e)}"
            else:
                debug_manager.log_debug(f"Skipping classification metrics: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
                # Set default values
                metrics['accuracy'] = 0.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
            
            # Additional classification metrics with intelligent task detection
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                
                # Only generate confusion matrix if we're confident this is classification
                if detected_task_type == 'classification' and confidence > 0.5:
                    try:
                        # Ensure both y_true and y_pred are discrete for confusion matrix
                        unique_true_values = np.unique(y_true)
                        
                        if len(unique_true_values) <= 20:  # Reasonable number of classes
                            # Convert predictions to discrete classes if needed
                            y_pred_discrete = np.zeros_like(y_pred)
                            for i, pred_val in enumerate(y_pred):
                                # Find closest true value
                                closest_idx = np.argmin(np.abs(unique_true_values - pred_val))
                                y_pred_discrete[i] = unique_true_values[closest_idx]
                            
                            debug_manager.log_debug(f"Converted predictions to discrete classes for confusion matrix")
                            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_discrete).tolist()
                        else:
                            debug_manager.log_debug(f"Cannot generate confusion matrix: Too many unique values ({len(unique_true_values)})")
                            metrics['confusion_matrix'] = None
                            
                    except Exception as e:
                        debug_manager.log_debug(f"Could not generate confusion matrix: {str(e)}")
                        metrics['confusion_matrix'] = None
                else:
                    debug_manager.log_debug(f"Skipping confusion matrix: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
                    metrics['confusion_matrix'] = None
                
                # Use discrete predictions for classification report if available
                y_pred_for_report = y_pred_discrete if y_pred_discrete is not None else y_pred
                metrics['classification_report'] = classification_report(y_true, y_pred_for_report, output_dict=True)
                
                metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                
            except Exception as e:
                debug_manager.log_debug(f"Error calculating additional classification metrics: {str(e)}")
                
        else:
            # Regression metrics - always calculate manually to avoid sklearn issues
            debug_manager.log_debug(f"Calculating regression metrics manually")
            debug_manager.log_debug(f"Task type for metrics: {task_type}")
            debug_manager.log_debug(f"y_true range: {np.min(y_true)} to {np.max(y_true)}")
            debug_manager.log_debug(f"y_pred range: {np.min(y_pred)} to {np.max(y_pred)}")
            
            try:
                # Calculate basic metrics manually (more reliable)
                metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
                
                # Calculate R² manually
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                metrics['r2_score'] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
                
                debug_manager.log_debug(f"Manual regression metrics calculated successfully")
                debug_manager.log_debug(f"R²: {metrics['r2_score']}, MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAE: {metrics['mae']}")
                
            except Exception as e:
                debug_manager.log_debug(f"Error calculating manual regression metrics: {str(e)}")
                # Set default values
                metrics['r2_score'] = 0.0
                metrics['mse'] = 0.0
                metrics['rmse'] = 0.0
                metrics['mae'] = 0.0
                metrics['error'] = f"Failed to calculate regression metrics: {str(e)}"
            
            # Additional regression metrics
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
                metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
                
                # Calculate relative metrics
                y_mean = np.mean(y_true)
                metrics['cv_rmse'] = metrics['rmse'] / y_mean if y_mean != 0 else float('inf')
                metrics['cv_mae'] = metrics['mae'] / y_mean if y_mean != 0 else float('inf')
                
            except Exception as e:
                debug_manager.log_debug(f"Error calculating additional regression metrics: {str(e)}")
        
        # Store the actual task type used and analysis details
        metrics['actual_task_type'] = task_type
        metrics['detection_confidence'] = confidence
        metrics['detection_analysis'] = analysis_details
        
        debug_manager.log_debug(f"Calculated metrics for {task_type} task with {unique_values} unique target values")
        
    except Exception as e:
        debug_manager.log_debug(f"Error calculating metrics: {str(e)}")
        # Return comprehensive fallback metrics
        if task_type == 'classification':
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
        else:
            metrics = {
                'r2_score': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'error': str(e)
            }
    
    return metrics


def get_feature_importance(model, feature_names: List[str], feature_selector) -> Dict[str, float]:
    """Get feature importance from trained model"""
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return {}
        
        # If feature selection was applied, map back to original features
        if feature_selector is not None:
            selected_features = feature_selector.get_support()
            full_importances = np.zeros(len(feature_names))
            full_importances[selected_features] = importances
            importances = full_importances
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                feature_importance[feature_names[i]] = float(importance)
        
        # Check for suspiciously high importance (potential data leakage)
        max_importance = max(importances) if len(importances) > 0 else 0
        
        # More aggressive threshold for post-training detection
        if max_importance > 0.7:  # Lowered from 0.9 to catch more cases
            debug_manager.log_debug(f"WARNING: Very high feature importance detected ({max_importance:.3f}). Possible data leakage.")
            
            # Find all features with suspiciously high importance
            suspicious_features = []
            for i, importance in enumerate(importances):
                if importance > 0.7 and i < len(feature_names):
                    suspicious_features.append((feature_names[i], importance))
                    debug_manager.log_debug(f"High importance feature: {feature_names[i]} = {importance:.3f}")
                    
                    # Add to leakage detection if not already detected
                    if feature_names[i] not in st.session_state.get('leakage_detected', {}):
                        if not hasattr(st.session_state, 'leakage_detected'):
                            st.session_state.leakage_detected = {}
                        st.session_state.leakage_detected[feature_names[i]] = {
                            'type': 'extremely_high_importance',
                            'importance': importance,
                            'severity': 'critical' if importance > 0.9 else 'high'
                        }
            
            # Log summary
            if suspicious_features:
                debug_manager.log_debug(f"Found {len(suspicious_features)} features with suspiciously high importance: {[f[0] for f in suspicious_features]}")
        
        return feature_importance
        
    except Exception as e:
        debug_manager.log_debug(f"Error getting feature importance: {str(e)}")
        return {}


def display_training_summary(results: Dict):
    """Display training summary"""
    
    st.subheader("Training Summary")
    
    # Data leakage warning
    if results.get('leakage_detected'):
        st.error("**WARNING: Potential Data Leakage Detected!**")
        
        leakage_info = results['leakage_detected']
        for feature, details in leakage_info.items():
            severity_color = {
                'critical': 'Critical',
                'high': 'High', 
                'medium': 'Medium'
            }.get(details['severity'], 'Unknown')
            
            st.warning(f"**{feature}**: {details['type']} (Severity: {severity_color})")
            
            if details['type'] == 'perfect_correlation':
                st.write(f"   - Training correlation: {details.get('train_corr', 'N/A'):.3f}")
                st.write(f"   - Test correlation: {details.get('test_corr', 'N/A'):.3f}")
            elif details['type'] == 'identical_to_target':
                st.write(f"   - Identical to target in training: {details.get('train_identical', False)}")
                st.write(f"   - Identical to target in test: {details.get('test_identical', False)}")
            elif details['type'] == 'extremely_predictive':
                st.write(f"   - Training R²: {details.get('train_r2', 'N/A'):.3f}")
                st.write(f"   - Test R²: {details.get('test_r2', 'N/A'):.3f}")
            elif details['type'] == 'suspicious_naming_pattern':
                st.write(f"   - Suspicious pattern detected: {details.get('pattern', 'N/A')}")
        
        st.info("**Recommendation**: Review your dataset for features that may contain target information or are derived from the target variable.")
    
    # Feature filtering information
    if results.get('features_filtered'):
        st.success("**Automatic Feature Filtering Applied**")
        filtered_features = results['features_filtered']
        st.write(f"**Removed {len(filtered_features)} suspicious features:**")
        for feature in filtered_features:
            st.write(f"• {feature}")
        st.info("These features were automatically removed to prevent data leakage.")
    
    # Binning information
    if results.get('binning_applied'):
        st.success("**Intelligent Binning Applied**")
        binning_info = results.get('binning_info', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Binning Details:**")
            st.write(f"Strategy: {binning_info.get('strategy', 'Unknown')}")
            st.write(f"Number of bins: {binning_info.get('n_bins', 'Unknown')}")
            st.write(f"Bin counts: {binning_info.get('bin_counts', 'Unknown')}")
        
        with col2:
            if binning_info.get('bin_labels'):
                st.write("**Bin Labels:**")
                for i, label in enumerate(binning_info['bin_labels']):
                    count = binning_info['bin_counts'][i] if i < len(binning_info['bin_counts']) else 'N/A'
                    st.write(f"• {label}: {count} samples")
            
            if binning_info.get('class_balance'):
                balance = binning_info['class_balance']
                if balance > 0.5:
                    st.success(f"Good class balance: {balance:.2f}")
                elif balance > 0.2:
                    st.warning(f"Moderate class imbalance: {balance:.2f}")
                else:
                    st.error(f"Poor class balance: {balance:.2f}")
        
        st.info("Continuous target variable was converted to discrete classes using intelligent binning.")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", results['model_type'])
    
    with col2:
        st.metric("Task Type", results['task_type'])
    
    with col3:
        st.metric("Training Time", f"{results['training_time']:.2f}s")
    
    with col4:
        cv_mean = np.mean(results['cv_scores'])
        cv_std = np.std(results['cv_scores'])
        st.metric("CV Score", f"{cv_mean:.3f} ± {cv_std:.3f}")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics = results['metrics']
    actual_task_type = metrics.get('actual_task_type', results['task_type'])
    unique_values = metrics.get('unique_target_values', 'Unknown')
    
    # Show task type information
    st.info(f"**Task Type**: {actual_task_type} (Target has {unique_values} unique values)")
    
    if actual_task_type == 'classification':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 'N/A'):.3f}" if metrics.get('accuracy') is not None else "N/A")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 'N/A'):.3f}" if metrics.get('precision') is not None else "N/A")
        
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 'N/A'):.3f}" if metrics.get('recall') is not None else "N/A")
        
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.3f}" if metrics.get('f1_score') is not None else "N/A")
        
        # Additional classification metrics
        if any(key in metrics for key in ['precision_macro', 'recall_macro', 'f1_macro']):
            st.subheader("Additional Classification Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precision (Macro)", f"{metrics.get('precision_macro', 'N/A'):.3f}" if metrics.get('precision_macro') is not None else "N/A")
            
            with col2:
                st.metric("Recall (Macro)", f"{metrics.get('recall_macro', 'N/A'):.3f}" if metrics.get('recall_macro') is not None else "N/A")
            
            with col3:
                st.metric("F1-Score (Macro)", f"{metrics.get('f1_macro', 'N/A'):.3f}" if metrics.get('f1_macro') is not None else "N/A")
    
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{metrics.get('r2_score', 'N/A'):.3f}" if metrics.get('r2_score') is not None else "N/A")
        
        with col2:
            st.metric("MSE", f"{metrics.get('mse', 'N/A'):.3f}" if metrics.get('mse') is not None else "N/A")
        
        with col3:
            st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.3f}" if metrics.get('rmse') is not None else "N/A")
        
        with col4:
            st.metric("MAE", f"{metrics.get('mae', 'N/A'):.3f}" if metrics.get('mae') is not None else "N/A")
        
        # Additional regression metrics
        if any(key in metrics for key in ['mape', 'explained_variance', 'cv_rmse']):
            st.subheader("Additional Regression Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MAPE", f"{metrics.get('mape', 'N/A'):.3f}" if metrics.get('mape') is not None else "N/A")
            
            with col2:
                st.metric("Explained Variance", f"{metrics.get('explained_variance', 'N/A'):.3f}" if metrics.get('explained_variance') is not None else "N/A")
            
            with col3:
                st.metric("CV-RMSE", f"{metrics.get('cv_rmse', 'N/A'):.3f}" if metrics.get('cv_rmse') is not None else "N/A")
    
    # Show error if any
    if 'error' in metrics:
        st.error(f"Error calculating metrics: {metrics['error']}")
    
    # Cross-validation scores
    st.subheader("Cross-Validation Results")
    
    cv_scores = results['cv_scores']
    fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(cv_scores) + 1)),
        y=cv_scores,
        mode='lines+markers',
        name='CV Scores'
    ))
    
    fig.add_hline(y=np.mean(cv_scores), line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {np.mean(cv_scores):.3f}")
    
    fig.update_layout(
        title="Cross-Validation Scores",
        xaxis_title="Fold",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Feature importance
    if results['feature_importance']:
        st.subheader("Feature Importance")
        
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_features:
            features, importances = zip(*sorted_features)
            
            fig = go.Figure(data=go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h'
            ))
            
            fig.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")
    
    # Model parameters
    st.subheader("Model Parameters")
    st.json(results['parameters'])
