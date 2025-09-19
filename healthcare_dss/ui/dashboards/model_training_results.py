"""
Model Training Results Display Module
====================================

This module handles training results display and visualization:
- Training results visualization
- Model performance metrics
- Prediction analysis
- Model comparison tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from healthcare_dss.ui.utils.common import (
    display_error_message,
    display_success_message,
    display_warning_message,
    safe_dataframe_display,
    safe_display_model_results,
    safe_display_feature_importance
)
from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently


def show_training_results_tab():
    """Show training results tab"""
    
    if 'training_results' not in st.session_state or st.session_state.training_results is None:
        st.warning("No training results available. Please train a model first.")
        return
    
    results = st.session_state.training_results
    
    st.subheader("Training Results")
    
    # Results overview
    _display_results_overview(results)
    
    # Performance metrics
    _display_performance_metrics(results)
    
    # Predictions analysis
    _display_predictions_analysis(results)
    
    # Feature importance
    _display_feature_importance(results)
    
    # Model comparison (if multiple models trained)
    _display_model_comparison()
    
    # Export options
    _display_export_options(results)


def _display_results_overview(results: Dict):
    """Display results overview"""
    
    st.subheader("Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", results['model_type'])
    
    with col2:
        st.metric("Task Type", results['task_type'])
    
    with col3:
        st.metric("Training Time", f"{results['training_time']:.2f}s")
    
    with col4:
        cv_mean = np.mean(results['cv_scores'])
        st.metric("CV Score", f"{cv_mean:.3f}")
    
    # Cross-validation scores chart
    st.subheader("Cross-Validation Scores")
    
    cv_scores = results['cv_scores']
    fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(cv_scores) + 1)),
        y=cv_scores,
        mode='lines+markers',
        name='CV Scores',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add mean line
    mean_score = np.mean(cv_scores)
    fig.add_hline(
        y=mean_score, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {mean_score:.3f}",
        annotation_position="top right"
    )
    
    # Add confidence interval
    std_score = np.std(cv_scores)
    fig.add_hline(y=mean_score + std_score, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=mean_score - std_score, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Cross-Validation Scores Across Folds",
        xaxis_title="Fold Number",
        yaxis_title="Score",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, width="stretch")


def _display_performance_metrics(results: Dict):
    """Display performance metrics"""
    
    st.subheader("Performance Metrics")
    
    metrics = results['metrics']
    task_type = results['task_type']
    
    if task_type == 'classification':
        _display_classification_metrics(metrics, results)
    else:
        _display_regression_metrics(metrics, results)


def _display_classification_metrics(metrics: Dict, results: Dict):
    """Display classification metrics"""
    
    # Show intelligent task detection information
    actual_task_type = metrics.get('actual_task_type', results['task_type'])
    detection_confidence = metrics.get('detection_confidence', None)
    detection_analysis = metrics.get('detection_analysis', {})
    
    if detection_confidence is not None:
        st.info(f"**Task Type**: {actual_task_type} (Detection Confidence: {detection_confidence:.1%})")
        if detection_analysis.get('confidence_factors'):
            st.caption(f"Detection factors: {', '.join(detection_analysis['confidence_factors'])}")
    else:
        st.info(f"**Task Type**: {actual_task_type}")
    
    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics.get('accuracy', 0.0)
        st.metric("Accuracy", f"{accuracy:.3f}" if accuracy is not None else "N/A")
    
    with col2:
        precision = metrics.get('precision', 0.0)
        st.metric("Precision", f"{precision:.3f}" if precision is not None else "N/A")
    
    with col3:
        recall = metrics.get('recall', 0.0)
        st.metric("Recall", f"{recall:.3f}" if recall is not None else "N/A")
    
    with col4:
        f1_score = metrics.get('f1_score', 0.0)
        st.metric("F1 Score", f"{f1_score:.3f}" if f1_score is not None else "N/A")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    # Use intelligent task detection to determine if this is actually classification
    detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_test)
    
    debug_manager.log_debug(f"Confusion matrix generation - Task type detection:")
    debug_manager.log_debug(f"  - Detected task_type: {detected_task_type}")
    debug_manager.log_debug(f"  - Confidence: {confidence:.3f}")
    debug_manager.log_debug(f"  - y_test unique values: {np.unique(y_test)}")
    debug_manager.log_debug(f"  - y_pred unique values: {np.unique(y_pred)}")
    
    # Only generate confusion matrix if we're confident this is classification
    if detected_task_type == 'classification' and confidence > 0.6:
        try:
            # Ensure both y_test and y_pred are discrete for confusion matrix
            # Convert continuous predictions to discrete classes if needed
            unique_true = np.unique(y_test)
            
            if len(unique_true) <= 20:  # Reasonable number of classes
                # Convert predictions to discrete classes
                y_pred_discrete = np.zeros_like(y_pred)
                for i, pred_val in enumerate(y_pred):
                    # Find closest true value
                    closest_idx = np.argmin(np.abs(unique_true - pred_val))
                    y_pred_discrete[i] = unique_true[closest_idx]
                
                debug_manager.log_debug(f"Converted predictions to discrete classes")
                debug_manager.log_debug(f"y_pred_discrete unique values: {np.unique(y_pred_discrete)}")
                
                cm = confusion_matrix(y_test, y_pred_discrete)
                
                # Create confusion matrix heatmap
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    color_continuous_scale="Blues"
                )
                
                fig.update_layout(
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Classification report
                st.subheader("Classification Report")
                
                report = classification_report(y_test, y_pred_discrete, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                # Format the dataframe for better display
                report_df = report_df.round(3)
                safe_dataframe_display(report_df, width="stretch")
                
            else:
                st.warning(f"Cannot generate confusion matrix: Too many unique values ({len(unique_true)})")
                
        except Exception as e:
            st.warning(f"Could not generate confusion matrix: {str(e)}")
            debug_manager.log_debug(f"Confusion matrix error: {str(e)}")
    else:
        st.info(f"Skipping confusion matrix: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
        if detected_task_type == 'regression':
            st.caption("Confusion matrices are only available for classification tasks.")


def _display_regression_metrics(metrics: Dict, results: Dict):
    """Display regression metrics"""
    
    # Show intelligent task detection information
    actual_task_type = metrics.get('actual_task_type', results['task_type'])
    detection_confidence = metrics.get('detection_confidence', None)
    detection_analysis = metrics.get('detection_analysis', {})
    
    if detection_confidence is not None:
        st.info(f"**Task Type**: {actual_task_type} (Detection Confidence: {detection_confidence:.1%})")
        if detection_analysis.get('confidence_factors'):
            st.caption(f"Detection factors: {', '.join(detection_analysis['confidence_factors'])}")
    else:
        st.info(f"**Task Type**: {actual_task_type}")
    
    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r2_score = metrics.get('r2_score', 0.0)
        st.metric("R² Score", f"{r2_score:.3f}" if r2_score is not None else "N/A")
    
    with col2:
        mse = metrics.get('mse', 0.0)
        st.metric("MSE", f"{mse:.3f}" if mse is not None else "N/A")
    
    with col3:
        rmse = metrics.get('rmse', 0.0)
        st.metric("RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")
    
    with col4:
        mae = metrics.get('mae', 0.0)
        st.metric("MAE", f"{mae:.3f}" if mae is not None else "N/A")
    
    # Prediction vs Actual scatter plot
    st.subheader("Predictions vs Actual")
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=6, opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Predictions vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Residuals plot
    st.subheader("Residuals Plot")
    
    residuals = y_test - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='green', size=6, opacity=0.6)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residuals vs Predicted Values",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def _display_predictions_analysis(results: Dict):
    """Display predictions analysis"""
    
    st.subheader("Predictions Analysis")
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    task_type = results['task_type']
    
    # Prediction statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Samples", len(y_test))
    
    with col2:
        if task_type == 'classification':
            correct_predictions = np.sum(y_test == y_pred)
            st.metric("Correct Predictions", correct_predictions)
        else:
            mae = np.mean(np.abs(y_test - y_pred))
            st.metric("Mean Absolute Error", f"{mae:.3f}")
    
    with col3:
        if task_type == 'classification':
            accuracy = np.sum(y_test == y_pred) / len(y_test)
            st.metric("Test Accuracy", f"{accuracy:.3f}")
        else:
            r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
            st.metric("Test R²", f"{r2:.3f}")
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    
    if task_type == 'classification':
        # Show prediction counts
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        actual_counts = pd.Series(y_test).value_counts().sort_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pred_counts.index,
            y=pred_counts.values,
            name='Predicted',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=actual_counts.index,
            y=actual_counts.values,
            name='Actual',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Prediction vs Actual Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, width="stretch")
    
    else:
        # Show prediction distribution
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual Distribution', 'Predicted Distribution'))
        
        fig.add_trace(go.Histogram(x=y_test, name='Actual', marker_color='lightblue'), row=1, col=1)
        fig.add_trace(go.Histogram(x=y_pred, name='Predicted', marker_color='lightcoral'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")


def _display_feature_importance(results: Dict):
    """Display feature importance"""
    
    if not results.get('feature_importance'):
        st.info("Feature importance not available for this model type.")
        return
    
    st.subheader("Feature Importance")
    
    feature_importance = results['feature_importance']
    
    # Check for extremely high importance (potential data leakage)
    max_importance = max(feature_importance.values()) if feature_importance else 0
    if max_importance > 0.9:
        st.error(f"**WARNING: Extremely high feature importance detected ({max_importance:.3f})!**")
        st.warning("This may indicate data leakage. Please review your dataset for features that might contain target information.")
        
        # Find and highlight the problematic feature
        max_feature = max(feature_importance.items(), key=lambda x: x[1])
        st.error(f"**Suspicious feature**: {max_feature[0]} with importance {max_feature[1]:.3f}")
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_features:
        st.info("No feature importance data available.")
        return
    
    # Top features
    top_n = min(15, len(sorted_features))
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    # Create horizontal bar chart
    fig = go.Figure(data=go.Bar(
        x=list(importances),
        y=list(features),
        orientation='h',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Feature importance table
    st.subheader("Feature Importance Table")
    
    importance_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    importance_df['Importance'] = importance_df['Importance'].round(4)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    safe_dataframe_display(importance_df, width="stretch")


def _display_model_comparison():
    """Display model comparison if multiple models are available"""
    
    # Check if we have multiple models in session state
    if 'model_comparison' not in st.session_state:
        return
    
    st.subheader("Model Comparison")
    
    comparison_data = st.session_state.model_comparison
    
    if len(comparison_data) < 2:
        st.info("Train multiple models to see comparison.")
        return
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison metrics
    safe_dataframe_display(comparison_df, width="stretch")
    
    # Comparison chart
    if 'accuracy' in comparison_df.columns:
        metric_col = 'accuracy'
    elif 'r2_score' in comparison_df.columns:
        metric_col = 'r2_score'
    else:
        return
    
    fig = px.bar(
        comparison_df,
        x='model_type',
        y=metric_col,
        title=f"Model Comparison - {metric_col.title()}",
        color=metric_col,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")


def _display_export_options(results: Dict):
    """Display export options for results"""
    
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Metrics"):
            _export_metrics(results)
    
    with col2:
        if st.button("Export Predictions"):
            _export_predictions(results)
    
    with col3:
        if st.button("Export Feature Importance"):
            _export_feature_importance(results)


def _export_metrics(results: Dict):
    """Export metrics to JSON"""
    
    export_data = {
        'model_type': results['model_type'],
        'task_type': results['task_type'],
        'training_time': results['training_time'],
        'cv_scores': results['cv_scores'].tolist(),
        'cv_mean': float(np.mean(results['cv_scores'])),
        'cv_std': float(np.std(results['cv_scores'])),
        'metrics': results['metrics'],
        'parameters': results['parameters'],
        'timestamp': results['timestamp']
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="Download Metrics JSON",
        data=json_str,
        file_name=f"model_metrics_{results['model_type'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def _export_predictions(results: Dict):
    """Export predictions to CSV"""
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'residual': y_test - y_pred if results['task_type'] == 'regression' else None
    })
    
    csv = predictions_df.to_csv(index=False)
    
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name=f"predictions_{results['model_type'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def _export_feature_importance(results: Dict):
    """Export feature importance to CSV"""
    
    if not results.get('feature_importance'):
        st.warning("No feature importance data to export.")
        return
    
    feature_importance = results['feature_importance']
    
    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    
    csv = importance_df.to_csv(index=False)
    
    st.download_button(
        label="Download Feature Importance CSV",
        data=csv,
        file_name=f"feature_importance_{results['model_type'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
