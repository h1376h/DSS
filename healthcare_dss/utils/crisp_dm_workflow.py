"""
CRISP-DM Workflow Implementation for Healthcare DSS

This module implements the Cross-Industry Standard Process for Data Mining (CRISP-DM)
specifically adapted for healthcare data mining projects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Import intelligent task detection
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently

class CRISPDMWorkflow:
    """
    CRISP-DM Workflow Implementation for Healthcare Data Mining
    
    Implements the six-phase CRISP-DM methodology:
    1. Business Understanding
    2. Data Understanding  
    3. Data Preparation
    4. Modeling
    5. Evaluation
    6. Deployment
    """
    
    def __init__(self, data_manager):
        """
        Initialize CRISP-DM Workflow
        
        Args:
            data_manager: DataManager instance with loaded datasets (includes all dataset functionality)
        """
        self.data_manager = data_manager
        self.workflow_results = {}
        self.models = {}
        self.evaluation_results = {}
        
    def _get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get dataset from the consolidated DataManager"""
        if dataset_name in self.data_manager.datasets:
            return self.data_manager.datasets[dataset_name]
        else:
            logger.warning(f"Dataset '{dataset_name}' not found in DataManager")
            return pd.DataFrame()
        
    def execute_full_workflow(self, dataset_name: str, target_column: str, 
                            business_objective: str = "Predict patient outcomes") -> Dict[str, Any]:
        """
        Execute complete CRISP-DM workflow
        
        Args:
            dataset_name: Name of dataset to analyze
            target_column: Target variable column name
            business_objective: Business objective description
            
        Returns:
            Dictionary containing workflow results
        """
        logger.info(f"Starting CRISP-DM workflow for {dataset_name}")
        
        # Phase 1: Business Understanding
        business_results = self._phase1_business_understanding(business_objective)
        
        # Phase 2: Data Understanding
        data_understanding = self._phase2_data_understanding(dataset_name)
        
        # Phase 3: Data Preparation
        prepared_data = self._phase3_data_preparation(dataset_name, target_column)
        
        # Phase 4: Modeling
        modeling_results = self._phase4_modeling(prepared_data, target_column)
        
        # Phase 5: Evaluation
        evaluation_results = self._phase5_evaluation(modeling_results, prepared_data, target_column)
        
        # Phase 6: Deployment
        deployment_results = self._phase6_deployment(evaluation_results)
        
        # Compile results
        self.workflow_results = {
            'business_understanding': business_results,
            'data_understanding': data_understanding,
            'data_preparation': prepared_data,
            'modeling': modeling_results,
            'evaluation': evaluation_results,
            'deployment': deployment_results
        }
        
        logger.info("CRISP-DM workflow completed successfully")
        return self.workflow_results
    
    def _phase1_business_understanding(self, business_objective: str) -> Dict[str, Any]:
        """Phase 1: Business Understanding"""
        logger.info("Phase 1: Business Understanding")
        
        return {
            'objective': business_objective,
            'success_criteria': [
                'Achieve >85% accuracy in predictions',
                'Model interpretability for clinical use',
                'Real-time prediction capability'
            ],
            'stakeholders': [
                'Clinical staff',
                'Healthcare administrators', 
                'Data scientists',
                'IT department'
            ],
            'constraints': [
                'HIPAA compliance required',
                'Real-time performance needed',
                'Model explainability required'
            ]
        }
    
    def _phase2_data_understanding(self, dataset_name: str) -> Dict[str, Any]:
        """Phase 2: Data Understanding"""
        logger.info("Phase 2: Data Understanding")
        
        df = self._get_dataset(dataset_name)
        
        # Basic data description
        data_description = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Statistical summary
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        statistical_summary = df[numeric_columns].describe().to_dict() if len(numeric_columns) > 0 else {}
        
        # Data quality assessment
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        return {
            'data_description': data_description,
            'statistical_summary': statistical_summary,
            'data_quality': {
                'completeness_percentage': completeness,
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(numeric_columns),
                'categorical_columns': len(df.columns) - len(numeric_columns)
            }
        }
    
    def _phase3_data_preparation(self, dataset_name: str, target_column: str) -> Dict[str, Any]:
        """Phase 3: Data Preparation"""
        logger.info("Phase 3: Data Preparation")
        
        df = self.data_manager.datasets[dataset_name].copy()
        
        # Data selection
        selected_data = df.copy()
        
        # Data cleaning
        # Handle missing values
        missing_before = selected_data.isnull().sum().sum()
        selected_data = selected_data.fillna(selected_data.median(numeric_only=True))
        missing_after = selected_data.isnull().sum().sum()
        
        # Remove duplicates
        duplicates_before = selected_data.duplicated().sum()
        selected_data = selected_data.drop_duplicates()
        duplicates_after = selected_data.duplicated().sum()
        
        # Data transformation
        # Encode categorical variables
        categorical_columns = selected_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:
                selected_data[col] = pd.Categorical(selected_data[col]).codes
        
        # Feature engineering
        numeric_columns = selected_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            # Create interaction features
            for i, col1 in enumerate(numeric_columns[:3]):  # Limit to first 3 to avoid too many features
                for col2 in numeric_columns[i+1:4]:
                    if col1 != target_column and col2 != target_column:
                        selected_data[f'{col1}_x_{col2}'] = selected_data[col1] * selected_data[col2]
        
        # Data integration (if multiple datasets available)
        integration_info = {
            'datasets_available': list(self.data_manager.datasets.keys()),
            'primary_dataset': dataset_name
        }
        
        return {
            'cleaned_data': selected_data,
            'data_cleaning': {
                'missing_values_removed': missing_before - missing_after,
                'duplicates_removed': duplicates_before - duplicates_after,
                'final_shape': selected_data.shape
            },
            'feature_engineering': {
                'original_features': len(df.columns),
                'engineered_features': len(selected_data.columns) - len(df.columns),
                'total_features': len(selected_data.columns)
            },
            'integration': integration_info
        }
    
    def _phase4_modeling(self, prepared_data: Dict[str, Any], target_column: str) -> Dict[str, Any]:
        """Phase 4: Modeling"""
        logger.info("Phase 4: Modeling")
        
        df = prepared_data['cleaned_data']
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle target variable encoding if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Validate data for machine learning
        unique_classes = np.unique(y)
        min_class_count = min(np.bincount(y))
        
        if len(unique_classes) < 2:
            raise ValueError(f"Target column '{target_column}' has only {len(unique_classes)} unique class(es). Need at least 2 classes for classification.")
        
        if min_class_count < 2:
            raise ValueError(f"Target column '{target_column}' has classes with insufficient data. Minimum class count is {min_class_count}, need at least 2.")
        
        # Check if we have enough data for train-test split
        if len(X) < 10:
            raise ValueError(f"Dataset has only {len(X)} samples. Need at least 10 samples for meaningful train-test split.")
        
        # Split data with proper stratification
        stratify_param = y if len(unique_classes) > 1 and min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        # Select modeling techniques
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=5000, solver='liblinear'),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"Trained {name} model successfully")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        self.models = trained_models
        
        return {
            'models_trained': list(trained_models.keys()),
            'training_data_shape': X_train.shape,
            'test_data_shape': X_test.shape,
            'feature_names': list(X.columns),
            'target_distribution': {
                'train': dict(pd.Series(y_train).value_counts()),
                'test': dict(pd.Series(y_test).value_counts())
            }
        }
    
    def _phase5_evaluation(self, modeling_results: Dict[str, Any], 
                          prepared_data: Dict[str, Any], target_column: str) -> Dict[str, Any]:
        """Phase 5: Evaluation"""
        logger.info("Phase 5: Evaluation")
        
        df = prepared_data['cleaned_data']
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle target variable encoding if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Validate data for evaluation
        unique_classes = np.unique(y)
        min_class_count = min(np.bincount(y))
        
        if len(unique_classes) < 2:
            raise ValueError(f"Target column '{target_column}' has only {len(unique_classes)} unique class(es). Need at least 2 classes for evaluation.")
        
        if min_class_count < 2:
            raise ValueError(f"Target column '{target_column}' has classes with insufficient data. Minimum class count is {min_class_count}, need at least 2.")
        
        # Check if we have enough data for train-test split
        if len(X) < 10:
            raise ValueError(f"Dataset has only {len(X)} samples. Need at least 10 samples for meaningful evaluation.")
        
        # Split data with proper stratification
        stratify_param = y if len(unique_classes) > 1 and min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Confusion matrix with intelligent task detection
                detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_test)
                
                logger.debug(f"CRISP-DM evaluation - Task type detection:")
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
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                evaluation_results[name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'confusion_matrix': cm.tolist() if cm is not None else None,
                    'classification_report': class_report
                }
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                evaluation_results[name] = {'error': str(e)}
        
        # Select best model
        best_model = None
        best_accuracy = 0
        for name, results in evaluation_results.items():
            if 'accuracy' in results and results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model = name
        
        self.evaluation_results = evaluation_results
        
        return {
            'model_evaluations': evaluation_results,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'evaluation_summary': {
                'models_evaluated': len(evaluation_results),
                'successful_evaluations': len([r for r in evaluation_results.values() if 'accuracy' in r])
            }
        }
    
    def _phase6_deployment(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Deployment"""
        logger.info("Phase 6: Deployment")
        
        best_model = evaluation_results.get('best_model')
        best_accuracy = evaluation_results.get('best_accuracy', 0)
        
        # Deployment planning
        deployment_plan = {
            'selected_model': best_model,
            'model_performance': best_accuracy,
            'deployment_strategy': 'Gradual rollout with A/B testing',
            'monitoring_plan': [
                'Model performance monitoring',
                'Data drift detection',
                'User feedback collection',
                'Regular model retraining'
            ],
            'integration_points': [
                'Healthcare DSS dashboard',
                'Clinical decision support system',
                'Real-time prediction API'
            ]
        }
        
        # Success criteria evaluation
        success_criteria_met = {
            'accuracy_target': best_accuracy >= 0.85,
            'model_interpretability': best_model in ['Random Forest', 'Decision Tree'],
            'real_time_capability': True  # Assuming all models can run in real-time
        }
        
        return {
            'deployment_plan': deployment_plan,
            'success_criteria': success_criteria_met,
            'next_steps': [
                'Deploy model to staging environment',
                'Conduct user acceptance testing',
                'Implement monitoring systems',
                'Plan production deployment'
            ]
        }
    
    def generate_workflow_report(self) -> str:
        """
        Generate comprehensive CRISP-DM workflow report
        
        Returns:
            Formatted workflow report string
        """
        if not self.workflow_results:
            return "No workflow results available. Please run execute_full_workflow() first."
        
        report = []
        report.append("=" * 80)
        report.append("CRISP-DM WORKFLOW REPORT FOR HEALTHCARE DSS")
        report.append("=" * 80)
        report.append("")
        
        # Phase 1: Business Understanding
        business = self.workflow_results['business_understanding']
        report.append("PHASE 1: BUSINESS UNDERSTANDING")
        report.append("-" * 40)
        report.append(f"Objective: {business['objective']}")
        report.append("Success Criteria:")
        for criterion in business['success_criteria']:
            report.append(f"  • {criterion}")
        report.append("")
        
        # Phase 2: Data Understanding
        data_understanding = self.workflow_results['data_understanding']
        report.append("PHASE 2: DATA UNDERSTANDING")
        report.append("-" * 40)
        report.append(f"Dataset Shape: {data_understanding['data_description']['shape']}")
        report.append(f"Data Quality: {data_understanding['data_quality']['completeness_percentage']:.1f}% completeness")
        report.append(f"Missing Values: {data_understanding['data_description']['missing_values']}")
        report.append("")
        
        # Phase 3: Data Preparation
        data_prep = self.workflow_results['data_preparation']
        report.append("PHASE 3: DATA PREPARATION")
        report.append("-" * 40)
        report.append(f"Missing Values Removed: {data_prep['data_cleaning']['missing_values_removed']}")
        report.append(f"Duplicates Removed: {data_prep['data_cleaning']['duplicates_removed']}")
        report.append(f"Features Engineered: {data_prep['feature_engineering']['engineered_features']}")
        report.append(f"Final Dataset Shape: {data_prep['data_cleaning']['final_shape']}")
        report.append("")
        
        # Phase 4: Modeling
        modeling = self.workflow_results['modeling']
        report.append("PHASE 4: MODELING")
        report.append("-" * 40)
        report.append(f"Models Trained: {', '.join(modeling['models_trained'])}")
        report.append(f"Training Data Shape: {modeling['training_data_shape']}")
        report.append(f"Test Data Shape: {modeling['test_data_shape']}")
        report.append("")
        
        # Phase 5: Evaluation
        evaluation = self.workflow_results['evaluation']
        report.append("PHASE 5: EVALUATION")
        report.append("-" * 40)
        report.append(f"Best Model: {evaluation['best_model']}")
        report.append(f"Best Accuracy: {evaluation['best_accuracy']:.3f}")
        report.append("Model Performance Summary:")
        for model_name, results in evaluation['model_evaluations'].items():
            if 'accuracy' in results:
                report.append(f"  {model_name}: {results['accuracy']:.3f} accuracy")
        report.append("")
        
        # Phase 6: Deployment
        deployment = self.workflow_results['deployment']
        report.append("PHASE 6: DEPLOYMENT")
        report.append("-" * 40)
        report.append(f"Selected Model: {deployment['deployment_plan']['selected_model']}")
        report.append("Success Criteria Met:")
        for criterion, met in deployment['success_criteria'].items():
            status = "✓" if met else "✗"
            report.append(f"  {status} {criterion.replace('_', ' ').title()}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
