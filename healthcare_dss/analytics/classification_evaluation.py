"""
Classification Model Evaluation Module for Healthcare DSS

This module implements comprehensive classification model evaluation including
confusion matrices, ROC curves, and other classification metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import intelligent task detection
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently

class ClassificationEvaluator:
    """
    Comprehensive Classification Model Evaluation for Healthcare DSS
    
    Provides detailed evaluation metrics, visualizations, and analysis
    for classification models in healthcare applications.
    """
    
    def __init__(self):
        """Initialize Classification Evaluator"""
        self.evaluation_results = {}
        self.visualizations = {}
        
    def evaluate_classification_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                    model_name: str = "Model", class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a classification model
        
        Args:
            model: Trained classification model
            X_test: Test features
            y_test: Test target values
            model_name: Name of the model for reporting
            class_names: Optional class names for better interpretation
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate confusion matrix with intelligent task detection
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_test)
        
        logger.debug(f"Classification evaluation - Task type detection:")
        logger.debug(f"  - Detected task_type: {detected_task_type}")
        logger.debug(f"  - Confidence: {confidence:.3f}")
        
        # Only generate confusion matrix if we're confident this is classification
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
                else:
                    logger.warning(f"Cannot generate confusion matrix: Too many unique values ({len(unique_true)})")
                    cm = None
                    
            except Exception as e:
                logger.warning(f"Could not generate confusion matrix: {str(e)}")
                cm = None
        else:
            logger.info(f"Skipping confusion matrix: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
            cm = None
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Calculate ROC AUC if binary classification
        roc_auc = None
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Calculate average precision
        avg_precision = None
        if y_pred_proba is not None:
            if len(np.unique(y_test)) == 2:
                avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
            else:
                avg_precision = average_precision_score(y_test, y_pred_proba, average='weighted')
        
        # Cross-validation scores
        cv_scores = None
        if hasattr(model, 'fit'):
            try:
                cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            except:
                cv_scores = None
        
        # Compile results
        results = {
            'model_name': model_name,
            'basic_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'average_precision': avg_precision
            },
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist()
            },
            'cross_validation': {
                'scores': cv_scores.tolist() if cv_scores is not None else None,
                'mean': cv_scores.mean() if cv_scores is not None else None,
                'std': cv_scores.std() if cv_scores is not None else None
            },
            'class_names': class_names or [f"Class {i}" for i in range(len(np.unique(y_test)))],
            'test_data_info': {
                'n_samples': len(y_test),
                'n_classes': len(np.unique(y_test)),
                'class_distribution': dict(pd.Series(y_test).value_counts().sort_index())
            }
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}: Accuracy = {accuracy:.3f}")
        return results
    
    def evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                           model_name: str = "Model", class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate classification predictions directly without requiring a model
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for reporting
            class_names: Optional class names for better interpretation
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Evaluating classification predictions for: {model_name}")
        
        # Convert to numpy arrays for consistency
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Get unique classes
        unique_classes = np.unique(y_true)
        
        # Prepare results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'precision_per_class': precision_per_class.tolist() if len(precision_per_class) > 0 else [],
            'recall_per_class': recall_per_class.tolist() if len(recall_per_class) > 0 else [],
            'f1_per_class': f1_per_class.tolist() if len(f1_per_class) > 0 else [],
            'unique_classes': unique_classes.tolist(),
            'class_names': class_names if class_names else [f"Class {i}" for i in unique_classes],
            'n_classes': len(unique_classes),
            'n_samples': len(y_true)
        }
        
        # Store results for later visualization
        self.evaluation_results[model_name] = results
        
        logger.info(f"Prediction evaluation completed for {model_name}: Accuracy = {accuracy:.3f}")
        return results
    
    def create_confusion_matrix_plot(self, model_name: str = None, y_true: pd.Series = None, 
                                   y_pred: np.ndarray = None, figsize: Tuple[int, int] = (8, 6)) -> go.Figure:
        """
        Create interactive confusion matrix plot
        
        Args:
            model_name: Name of the model to plot (if using stored results)
            y_true: True target values (if creating plot directly)
            y_pred: Predicted values (if creating plot directly)
            figsize: Figure size (for matplotlib compatibility)
            
        Returns:
            Plotly figure with confusion matrix
        """
        if y_true is not None and y_pred is not None:
            # Create plot directly from predictions
            cm = confusion_matrix(y_true, y_pred)
            unique_classes = np.unique(y_true)
            class_names = [f"Class {i}" for i in unique_classes]
            model_name = model_name or "Direct Predictions"
        elif model_name and model_name in self.evaluation_results:
            # Use stored results
            results = self.evaluation_results[model_name]
            cm = np.array(results['confusion_matrix'])
            class_names = results['class_names']
        else:
            raise ValueError("Either provide model_name with stored results, or y_true and y_pred for direct plotting")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        return fig
    
    def create_roc_curve_plot(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = "Model") -> Optional[go.Figure]:
        """
        Create ROC curve plot for binary classification
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_name: Name of the model
            
        Returns:
            Plotly figure with ROC curve or None if not binary classification
        """
        if len(np.unique(y_test)) != 2:
            logger.warning("ROC curve only available for binary classification")
            return None
        
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction")
            return None
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Create plot
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_precision_recall_plot(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                   model_name: str = "Model") -> Optional[go.Figure]:
        """
        Create Precision-Recall curve plot
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_name: Name of the model
            
        Returns:
            Plotly figure with Precision-Recall curve
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction")
            return None
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_test)
        
        if len(np.unique(y_test)) == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        else:
            # Multi-class classification
            precision, recall, thresholds = precision_recall_curve(
                y_test, y_pred_proba, average='weighted'
            )
            avg_precision = average_precision_score(y_test, y_pred_proba, average='weighted')
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_metrics_comparison_plot(self, models: Optional[List[str]] = None) -> go.Figure:
        """
        Create comparison plot of metrics across multiple models
        
        Args:
            models: List of model names to compare (if None, uses all available)
            
        Returns:
            Plotly figure with metrics comparison
        """
        if models is None:
            models = list(self.evaluation_results.keys())
        
        if not models:
            raise ValueError("No evaluation results available")
        
        # Extract metrics
        metrics_data = []
        for model_name in models:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                metrics = results['basic_metrics']
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
        
        if not metrics_data:
            raise ValueError("No valid evaluation results found")
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=df['Model'],
                    y=df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """
        Generate detailed evaluation report for a model
        
        Args:
            model_name: Name of the model to report on
            
        Returns:
            Formatted evaluation report string
        """
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for model: {model_name}"
        
        results = self.evaluation_results[model_name]
        
        report = []
        report.append("=" * 60)
        report.append(f"CLASSIFICATION MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        metrics = results['basic_metrics']
        report.append("BASIC METRICS")
        report.append("-" * 20)
        report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall:    {metrics['recall']:.4f}")
        report.append(f"F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            report.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        if metrics['average_precision'] is not None:
            report.append(f"Avg Precision: {metrics['average_precision']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS")
        report.append("-" * 20)
        class_names = results['class_names']
        precision_per_class = results['per_class_metrics']['precision']
        recall_per_class = results['per_class_metrics']['recall']
        f1_per_class = results['per_class_metrics']['f1_score']
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                report.append(f"{class_name}:")
                report.append(f"  Precision: {precision_per_class[i]:.4f}")
                report.append(f"  Recall:    {recall_per_class[i]:.4f}")
                report.append(f"  F1-Score:  {f1_per_class[i]:.4f}")
        report.append("")
        
        # Confusion matrix
        report.append("CONFUSION MATRIX")
        report.append("-" * 20)
        cm = np.array(results['confusion_matrix'])
        report.append("Predicted ->")
        report.append("Actual")
        
        # Header row
        header = "     " + "".join([f"{name:>8}" for name in class_names])
        report.append(header)
        
        # Data rows
        for i, class_name in enumerate(class_names):
            row = f"{class_name:>4} " + "".join([f"{cm[i][j]:>8}" for j in range(len(class_names))])
            report.append(row)
        report.append("")
        
        # Cross-validation results
        cv_results = results['cross_validation']
        if cv_results['scores'] is not None:
            report.append("CROSS-VALIDATION RESULTS")
            report.append("-" * 30)
            report.append(f"Mean Score: {cv_results['mean']:.4f}")
            report.append(f"Std Dev:    {cv_results['std']:.4f}")
            report.append(f"Scores:     {[f'{score:.4f}' for score in cv_results['scores']]}")
            report.append("")
        
        # Test data information
        test_info = results['test_data_info']
        report.append("TEST DATA INFORMATION")
        report.append("-" * 25)
        report.append(f"Number of samples: {test_info['n_samples']}")
        report.append(f"Number of classes: {test_info['n_classes']}")
        report.append("Class distribution:")
        for class_name, count in test_info['class_distribution'].items():
            report.append(f"  {class_name}: {count}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on accuracy
        
        Returns:
            Tuple of (model_name, results) for the best model
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_model = None
        best_accuracy = 0
        
        for model_name, results in self.evaluation_results.items():
            accuracy = results['basic_metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        return best_model, self.evaluation_results[best_model]
