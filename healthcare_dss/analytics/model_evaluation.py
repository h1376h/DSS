"""
Model Evaluation Engine for Healthcare DSS
==========================================

This module handles model evaluation, performance analysis, and
explainability features for the Healthcare Decision Support System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)

# Import intelligent task detection
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently


class ModelEvaluationEngine:
    """
    Advanced model evaluation engine for healthcare data analysis
    
    Provides comprehensive model evaluation, performance analysis,
    and explainability capabilities.
    """
    
    def __init__(self):
        """Initialize the model evaluation engine"""
        self.explainers = {}
        
    def evaluate_model_performance(self, model, X_test, y_test, y_pred, task_type: str) -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test labels
            y_pred: Predicted labels
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        try:
            evaluation = {
                'basic_metrics': self._calculate_basic_metrics(y_test, y_pred, task_type),
                'detailed_metrics': self._calculate_detailed_metrics(y_test, y_pred, task_type),
                'visualizations': self._create_performance_visualizations(y_test, y_pred, task_type),
                'feature_analysis': self._analyze_feature_importance(model, X_test.columns.tolist()),
                'error_analysis': self._analyze_prediction_errors(y_test, y_pred, X_test)
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            raise
    
    def _calculate_basic_metrics(self, y_test, y_pred, task_type: str) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Additional classification metrics
            try:
                # Check if we have multiple classes and positive samples for ROC AUC
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1:
                    # For binary classification, check if we have both classes
                    if len(unique_classes) == 2:
                        # Check if both classes are present in both y_test and y_pred
                        test_classes = set(y_test)
                        pred_classes = set(y_pred)
                        if len(test_classes) == 2 and len(pred_classes) == 2:
                            metrics['auc_roc'] = roc_auc_score(y_test, y_pred)
                        else:
                            metrics['auc_roc'] = 0.5  # Random performance
                    else:
                        # Multi-class ROC AUC
                        metrics['auc_roc'] = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
                else:
                    metrics['auc_roc'] = 0.5  # Random performance for single class
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['auc_roc'] = 0.5
                
        else:  # regression
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            
            # Additional regression metrics
            metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
            metrics['explained_variance'] = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        
        return metrics
    
    def _calculate_detailed_metrics(self, y_test, y_pred, task_type: str) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        detailed = {}
        
        # Use intelligent task detection to determine actual task type
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_test)
        
        logger.debug(f"Model evaluation - Task type detection:")
        logger.debug(f"  - Input task_type: {task_type}")
        logger.debug(f"  - Detected task_type: {detected_task_type}")
        logger.debug(f"  - Confidence: {confidence:.3f}")
        
        # Use detected task type if confidence is high, otherwise use input
        if confidence > 0.7:
            actual_task_type = detected_task_type
            logger.debug(f"  - Using detected task type: {actual_task_type} (confidence: {confidence:.3f})")
        else:
            actual_task_type = task_type
            logger.debug(f"  - Using input task type: {actual_task_type} (low confidence: {confidence:.3f})")
        
        # Store detection information
        detailed['detected_task_type'] = detected_task_type
        detailed['detection_confidence'] = confidence
        detailed['detection_analysis'] = analysis_details
        detailed['actual_task_type'] = actual_task_type
        
        if actual_task_type == 'classification':
            # Confusion matrix - only generate if we're confident this is classification
            if detected_task_type == 'classification' and confidence > 0.6:
                try:
                    # Ensure both y_test and y_pred are discrete for confusion matrix
                    unique_true = np.unique(y_test)
                    
                    if len(unique_true) <= 20:  # Reasonable number of classes
                        # Convert predictions to discrete classes if needed
                        y_pred_discrete = np.zeros_like(y_pred)
                        for i, pred_val in enumerate(y_pred):
                            # Find closest true value
                            closest_idx = np.argmin(np.abs(unique_true - pred_val))
                            y_pred_discrete[i] = unique_true[closest_idx]
                        
                        logger.debug(f"Converted predictions to discrete classes for confusion matrix")
                        cm = confusion_matrix(y_test, y_pred_discrete)
                        detailed['confusion_matrix'] = {
                            'matrix': cm.tolist(),
                            'labels': sorted(list(set(y_test) | set(y_pred_discrete)))
                        }
                    else:
                        logger.warning(f"Cannot generate confusion matrix: Too many unique values ({len(unique_true)})")
                        detailed['confusion_matrix'] = None
                        
                except Exception as e:
                    logger.warning(f"Could not generate confusion matrix: {str(e)}")
                    detailed['confusion_matrix'] = None
            else:
                logger.info(f"Skipping confusion matrix: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
                detailed['confusion_matrix'] = None
            
            # Classification report
            try:
                # Check if we have multiple classes for classification report
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1:
                    detailed['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                else:
                    detailed['classification_report'] = None
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
                detailed['classification_report'] = None
            
            # ROC curve data
            try:
                # Check if we have multiple classes for ROC curve
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1:
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                    detailed['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist()
                    }
                else:
                    detailed['roc_curve'] = None
            except Exception as e:
                logger.warning(f"Could not generate ROC curve: {e}")
                detailed['roc_curve'] = None
            
            # Precision-Recall curve
            try:
                # Check if we have multiple classes for precision-recall curve
                unique_classes = np.unique(y_test)
                if len(unique_classes) > 1:
                    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
                    detailed['precision_recall_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': thresholds.tolist()
                    }
                else:
                    detailed['precision_recall_curve'] = None
            except Exception as e:
                logger.warning(f"Could not generate precision-recall curve: {e}")
                detailed['precision_recall_curve'] = None
                
        else:  # regression
            # Residual analysis
            residuals = y_test - y_pred
            detailed['residual_analysis'] = {
                'residuals': residuals.tolist(),
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': self._calculate_skewness(residuals),
                'residual_kurtosis': self._calculate_kurtosis(residuals)
            }
            
            # Prediction intervals
            detailed['prediction_intervals'] = self._calculate_prediction_intervals(y_test, y_pred)
        
        return detailed
    
    def _create_performance_visualizations(self, y_test, y_pred, task_type: str) -> Dict[str, Any]:
        """Create performance visualization data"""
        visualizations = {}
        
        # Use intelligent task detection to determine actual task type
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_test)
        
        # Use detected task type if confidence is high, otherwise use input
        if confidence > 0.7:
            actual_task_type = detected_task_type
        else:
            actual_task_type = task_type
        
        if actual_task_type == 'classification':
            # Confusion matrix heatmap - only generate if we're confident this is classification
            if detected_task_type == 'classification' and confidence > 0.6:
                try:
                    # Ensure both y_test and y_pred are discrete for confusion matrix
                    unique_true = np.unique(y_test)
                    
                    if len(unique_true) <= 20:  # Reasonable number of classes
                        # Convert predictions to discrete classes if needed
                        y_pred_discrete = np.zeros_like(y_pred)
                        for i, pred_val in enumerate(y_pred):
                            # Find closest true value
                            closest_idx = np.argmin(np.abs(unique_true - pred_val))
                            y_pred_discrete[i] = unique_true[closest_idx]
                        
                        logger.debug(f"Converted predictions to discrete classes for visualization")
                        
                        # Validate data before creating confusion matrix
                        unique_test = np.unique(y_test)
                        unique_pred = np.unique(y_pred_discrete)
                        
                        if len(unique_test) > 1 and len(unique_pred) > 1:
                            cm = confusion_matrix(y_test, y_pred_discrete)
                        else:
                            logger.warning("Insufficient class diversity for confusion matrix")
                            cm = np.array([[len(y_test), 0], [0, 0]])  # Fallback matrix
                        visualizations['confusion_matrix'] = {
                            'data': cm.tolist(),
                            'labels': sorted(list(set(y_test) | set(y_pred_discrete))),
                            'type': 'heatmap'
                        }
                    else:
                        logger.warning(f"Cannot generate confusion matrix visualization: Too many unique values ({len(unique_true)})")
                        visualizations['confusion_matrix'] = None
                        
                except Exception as e:
                    logger.warning(f"Could not generate confusion matrix visualization: {str(e)}")
                    visualizations['confusion_matrix'] = None
            else:
                logger.info(f"Skipping confusion matrix visualization: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
                visualizations['confusion_matrix'] = None
            
            # ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                visualizations['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'type': 'line'
                }
            except:
                visualizations['roc_curve'] = None
                
        else:  # regression
            # Actual vs Predicted scatter plot
            visualizations['actual_vs_predicted'] = {
                'actual': y_test.tolist(),
                'predicted': y_pred.tolist(),
                'type': 'scatter'
            }
            
            # Residual plot
            residuals = y_test - y_pred
            visualizations['residual_plot'] = {
                'predicted': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'type': 'scatter'
            }
        
        return visualizations
    
    def _analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from trained model"""
        try:
            importance_analysis = {}
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                
                importance_analysis['feature_importance'] = sorted_importance
                importance_analysis['top_features'] = list(sorted_importance.keys())[:10]
                importance_analysis['importance_scores'] = list(sorted_importance.values())[:10]
                
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    coef_dict = dict(zip(feature_names, abs(model.coef_)))
                else:
                    coef_dict = dict(zip(feature_names, abs(model.coef_).mean(axis=0)))
                
                sorted_coef = dict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True))
                importance_analysis['feature_importance'] = sorted_coef
                importance_analysis['top_features'] = list(sorted_coef.keys())[:10]
                importance_analysis['importance_scores'] = list(sorted_coef.values())[:10]
                
            else:
                importance_analysis['feature_importance'] = {}
                importance_analysis['top_features'] = []
                importance_analysis['importance_scores'] = []
            
            return importance_analysis
            
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {e}")
            return {'feature_importance': {}, 'top_features': [], 'importance_scores': []}
    
    def _analyze_prediction_errors(self, y_test, y_pred, X_test) -> Dict[str, Any]:
        """Analyze prediction errors and identify patterns"""
        try:
            errors = y_test - y_pred
            abs_errors = np.abs(errors)
            
            error_analysis = {
                'error_statistics': {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'mean_absolute_error': float(np.mean(abs_errors)),
                    'max_error': float(np.max(abs_errors)),
                    'min_error': float(np.min(abs_errors))
                },
                'error_distribution': {
                    'percentiles': {
                        '25th': float(np.percentile(abs_errors, 25)),
                        '50th': float(np.percentile(abs_errors, 50)),
                        '75th': float(np.percentile(abs_errors, 75)),
                        '90th': float(np.percentile(abs_errors, 90)),
                        '95th': float(np.percentile(abs_errors, 95))
                    }
                },
                'worst_predictions': self._identify_worst_predictions(y_test, y_pred, X_test)
            }
            
            return error_analysis
            
        except Exception as e:
            logger.warning(f"Could not analyze prediction errors: {e}")
            return {'error_statistics': {}, 'error_distribution': {}, 'worst_predictions': []}
    
    def _identify_worst_predictions(self, y_test, y_pred, X_test, n_worst: int = 5) -> List[Dict[str, Any]]:
        """Identify the worst predictions for analysis"""
        try:
            errors = np.abs(y_test - y_pred)
            worst_indices = np.argsort(errors)[-n_worst:][::-1]
            
            worst_predictions = []
            for idx in worst_indices:
                worst_predictions.append({
                    'index': int(idx),
                    'actual': float(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else float(y_test[idx]),
                    'predicted': float(y_pred[idx]),
                    'error': float(errors[idx]),
                    'features': X_test.iloc[idx].to_dict() if hasattr(X_test, 'iloc') else {}
                })
            
            return worst_predictions
            
        except Exception as e:
            logger.warning(f"Could not identify worst predictions: {e}")
            return []
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5, 
                           task_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform cross-validation evaluation of a model
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target values
            cv: Number of cross-validation folds
            task_type: Type of task ('classification', 'regression', or 'auto')
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, r2_score
            
            # Detect task type if auto
            if task_type == 'auto':
                task_type = self._detect_task_type(y)
            
            # Prepare scoring metrics based on task type
            if task_type == 'classification':
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
                
                # Adaptive CV strategy for classification - handle class imbalance
                cv_strategy = self._get_adaptive_cv_strategy(y, cv, task_type)
            else:  # regression
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Perform cross-validation for each metric
            cv_results = {}
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=metric, n_jobs=-1)
                    cv_results[metric] = {
                        'scores': scores.tolist(),
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'min': float(scores.min()),
                        'max': float(scores.max())
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate {metric}: {e}")
                    cv_results[metric] = {'error': str(e)}
            
            # Calculate overall performance
            if 'accuracy' in cv_results:
                overall_score = cv_results['accuracy']['mean']
            elif 'r2' in cv_results:
                overall_score = cv_results['r2']['mean']
            elif 'neg_mean_squared_error' in cv_results:
                overall_score = -cv_results['neg_mean_squared_error']['mean']  # Convert back to positive
            else:
                overall_score = 0.0
            
            # Determine model stability
            if 'accuracy' in cv_results:
                stability_score = 1.0 - cv_results['accuracy']['std']
            elif 'r2' in cv_results:
                stability_score = 1.0 - abs(cv_results['r2']['std'])
            else:
                stability_score = 0.5
            
            results = {
                'task_type': task_type,
                'cv_folds': cv,
                'overall_score': overall_score,
                'stability_score': stability_score,
                'cv_results': cv_results,
                'model_performance': {
                    'excellent': overall_score > 0.9,
                    'good': 0.8 <= overall_score <= 0.9,
                    'fair': 0.7 <= overall_score < 0.8,
                    'poor': overall_score < 0.7
                },
                'stability_assessment': {
                    'very_stable': stability_score > 0.9,
                    'stable': 0.8 <= stability_score <= 0.9,
                    'moderate': 0.7 <= stability_score < 0.8,
                    'unstable': stability_score < 0.7
                }
            }
            
            logger.info(f"Cross-validation completed: Overall score = {overall_score:.3f}, Stability = {stability_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {
                'error': str(e),
                'task_type': task_type,
                'cv_folds': cv
            }
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect task type based on target values using intelligent analysis"""
        try:
            from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently
            
            # Use intelligent task detection
            task_type, confidence, analysis_details = detect_task_type_intelligently(y.values, 'target')
            
            logger.info(f"Task type detection: {task_type} (confidence: {confidence:.3f})")
            logger.debug(f"Analysis details: {analysis_details}")
            
            return task_type
        except Exception as e:
            logger.warning(f"Intelligent task detection failed: {e}, using fallback")
            # Fallback to simple heuristic
            try:
                unique_values = y.nunique()
                total_values = len(y)
                unique_ratio = unique_values / total_values if total_values > 0 else 0
                
                # More sophisticated fallback logic
                if unique_values <= 2:
                    return 'classification'  # Binary classification
                elif unique_values <= 10 and unique_ratio < 0.1:
                    return 'classification'  # Low cardinality categorical
                elif pd.api.types.is_numeric_dtype(y) and unique_ratio > 0.1:
                    return 'regression'  # High cardinality numeric
                else:
                    return 'regression'  # Default to regression for continuous data
            except:
                return 'regression'  # Safe default
    
    def _get_adaptive_cv_strategy(self, y: pd.Series, cv: int, task_type: str):
        """Get adaptive cross-validation strategy based on data characteristics"""
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
            
            if task_type == 'classification':
                # Check for class imbalance issues
                unique_values = y.nunique()
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                
                logger.info(f"Classification CV strategy analysis:")
                logger.info(f"  - Unique classes: {unique_values}")
                logger.info(f"  - Min class count: {min_class_count}")
                logger.info(f"  - Class distribution: {class_counts.to_dict()}")
                
                # If minimum class count is too small for stratified CV, use regular KFold
                if min_class_count < 2:
                    logger.warning(f"Minimum class count ({min_class_count}) too small for StratifiedKFold, using KFold")
                    return KFold(n_splits=cv, shuffle=True, random_state=42)
                
                # If we have enough samples per class, use StratifiedKFold
                if min_class_count >= cv:
                    logger.info(f"Using StratifiedKFold with {cv} folds")
                    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                else:
                    # Reduce CV folds to accommodate class distribution
                    adaptive_cv = min(cv, min_class_count)
                    logger.warning(f"Reducing CV folds from {cv} to {adaptive_cv} due to class imbalance")
                    return StratifiedKFold(n_splits=adaptive_cv, shuffle=True, random_state=42)
            else:
                # For regression, always use KFold
                return KFold(n_splits=cv, shuffle=True, random_state=42)
                
        except Exception as e:
            logger.warning(f"Error in adaptive CV strategy: {e}, using default KFold")
            return KFold(n_splits=cv, shuffle=True, random_state=42)
    
    def create_explainers(self, model_key: str, X_train: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Create explainability tools for the model"""
        try:
            explainers = {}
            
            # SHAP explainer
            try:
                # Try different explainer types based on model
                if hasattr(model_key, 'tree_') or hasattr(model_key, 'estimators_'):
                    # Tree-based models
                    explainer = shap.TreeExplainer(model_key)
                elif hasattr(model_key, 'coef_'):
                    # Linear models
                    explainer = shap.LinearExplainer(model_key, X_train)
                else:
                    # Generic explainer
                    explainer = shap.Explainer(model_key, X_train)
                
                explainers['shap'] = {
                    'explainer': explainer,
                    'type': 'shap'
                }
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
            
            # LIME explainer
            try:
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    class_names=['class_0', 'class_1'] if len(np.unique(X_train)) == 2 else None,
                    mode='classification' if hasattr(model_key, 'predict_proba') else 'regression'
                )
                
                explainers['lime'] = {
                    'explainer': lime_explainer,
                    'type': 'lime'
                }
            except Exception as e:
                logger.warning(f"Could not create LIME explainer: {e}")
            
            self.explainers[model_key] = explainers
            return explainers
            
        except Exception as e:
            logger.error(f"Error creating explainers: {e}")
            return {}
    
    def explain_prediction(self, model_key: str, features: Union[pd.DataFrame, np.ndarray], 
                          instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a specific prediction using available explainers"""
        try:
            if model_key not in self.explainers:
                return {'error': 'No explainers available for this model'}
            
            explanation = {}
            explainers = self.explainers[model_key]
            
            # Prepare instance
            if isinstance(features, pd.DataFrame):
                instance = features.iloc[instance_idx].values
            else:
                instance = features[instance_idx]
            
            # SHAP explanation
            if 'shap' in explainers:
                try:
                    shap_values = explainers['shap']['explainer'].shap_values(instance)
                    explanation['shap'] = {
                        'values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                        'base_value': explainers['shap']['explainer'].expected_value
                    }
                except Exception as e:
                    explanation['shap'] = {'error': str(e)}
            
            # LIME explanation
            if 'lime' in explainers:
                try:
                    lime_exp = explainers['lime']['explainer'].explain_instance(
                        instance, 
                        lambda x: explainers['lime']['explainer'].predict(x)
                    )
                    explanation['lime'] = {
                        'explanation': lime_exp.as_list(),
                        'score': lime_exp.score
                    }
                except Exception as e:
                    explanation['lime'] = {'error': str(e)}
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            skewness = np.mean(((data - mean) / std) ** 3)
            return float(skewness)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3
            return float(kurtosis)
        except:
            return 0.0
    
    def _calculate_prediction_intervals(self, y_test, y_pred, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate prediction intervals for regression"""
        try:
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            
            # Calculate prediction intervals
            z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            margin_of_error = z_score * residual_std
            
            return {
                'lower_bound': float(np.mean(y_pred) - margin_of_error),
                'upper_bound': float(np.mean(y_pred) + margin_of_error),
                'confidence_level': confidence
            }
        except:
            return {'lower_bound': 0, 'upper_bound': 0, 'confidence_level': confidence}
