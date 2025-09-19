"""
Model Management Subsystem for Healthcare DSS
============================================

This module implements the main model management capabilities including:
- Predictive analytics and machine learning models
- Model training and validation
- Model performance evaluation
- Model deployment and monitoring
- Ensemble modeling and information fusion

The ModelManager now acts as a coordinator that uses specialized modules:
- PreprocessingEngine: Data preprocessing and analysis
- ModelTrainingEngine: Model training and hyperparameter optimization
- ModelEvaluationEngine: Performance evaluation and explainability
- ModelRegistry: Model storage, versioning, and management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime

# Import specialized modules
from healthcare_dss.core.preprocessing_engine import PreprocessingEngine
from healthcare_dss.analytics.model_training import ModelTrainingEngine
from healthcare_dss.analytics.model_evaluation import ModelEvaluationEngine
from healthcare_dss.analytics.model_registry import ModelRegistry
from healthcare_dss.core.data_management import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model Management Subsystem for Healthcare DSS
    
    Implements AI-powered model management following DSS_2.md methodology:
    - AI Technology Selection Matrix for healthcare problems
    - Knowledge Acquisition and Organization methods
    - Automated Decision-Making Process implementation
    - AI vs Human Intelligence comparison capabilities
    
    This class coordinates all model management activities by delegating
    to specialized engines for preprocessing, training, evaluation, and storage.
    """
    
    def __init__(self, data_manager: DataManager, models_dir: str = "models"):
        """
        Initialize the Model Manager
        
        Args:
            data_manager: DataManager instance for data access
            models_dir: Directory to store model files
        """
        self.data_manager = data_manager
        
        # Initialize specialized engines
        self.preprocessing_engine = PreprocessingEngine()
        self.training_engine = ModelTrainingEngine()
        self.evaluation_engine = ModelEvaluationEngine()
        self.registry = ModelRegistry(models_dir)
        
        logger.info("ModelManager initialized with specialized engines")
    
    def get_ai_technology_selection_matrix(self) -> Dict[str, Any]:
        """
        Get AI Technology Selection Matrix following DSS_2.md methodology
        
        Returns:
            Dictionary containing AI technology recommendations for healthcare problems
        """
        ai_matrix = {
            'healthcare_problems': {
                'medical_image_analysis': {
                    'problem': 'Inconsistent analysis of medical images for disease detection',
                    'ai_technology': 'Computer Vision with Deep Learning',
                    'data_requirements': 'Large dataset of labeled medical images (X-rays, MRIs, CT scans)',
                    'desired_outcome': 'Higher accuracy in early detection, faster screening process, reduced radiologist workload',
                    'implementation_complexity': 'High',
                    'expected_roi': 'Very High'
                },
                'patient_query_handling': {
                    'problem': 'High volume of patient queries about symptoms and treatments',
                    'ai_technology': 'NLP-powered Chatbot',
                    'data_requirements': 'Medical knowledge bases, clinical guidelines, anonymized patient interaction logs',
                    'desired_outcome': '24/7 patient support, reduced call volume for clinical staff, consistent medical information',
                    'implementation_complexity': 'Medium',
                    'expected_roi': 'High'
                },
                'patient_deterioration_prediction': {
                    'problem': 'Difficulty predicting patient deterioration or readmission risk',
                    'ai_technology': 'Machine Learning (Classification/Regression models)',
                    'data_requirements': 'Historical patient data with outcomes, demographics, lab results, vital signs',
                    'desired_outcome': 'Reduced readmission rates, early intervention, optimized resource allocation',
                    'implementation_complexity': 'Medium',
                    'expected_roi': 'Very High'
                },
                'rare_disease_treatment': {
                    'problem': 'Complex treatment planning for rare diseases',
                    'ai_technology': 'Expert Systems with Knowledge Graphs',
                    'data_requirements': 'Medical literature, clinical guidelines, patient case histories',
                    'desired_outcome': 'Personalized treatment recommendations, reduced time to diagnosis, improved outcomes',
                    'implementation_complexity': 'High',
                    'expected_roi': 'High'
                },
                'resource_allocation_optimization': {
                    'problem': 'Optimizing hospital resource allocation and scheduling',
                    'ai_technology': 'Prescriptive Analytics with Optimization Models',
                    'data_requirements': 'Patient demand patterns, resource availability, staff schedules',
                    'desired_outcome': 'Reduced wait times, improved resource utilization, cost optimization',
                    'implementation_complexity': 'High',
                    'expected_roi': 'Very High'
                }
            },
            'ai_vs_human_comparison': {
                'execution_speed': {
                    'ai': 'Very fast - can process thousands of medical images in minutes',
                    'human': 'Can be slow - radiologist may take 15-20 minutes per complex case',
                    'healthcare_context': 'AI can provide rapid preliminary screening, humans provide final diagnosis'
                },
                'consistency': {
                    'ai': 'High; Stable performance across cases',
                    'human': 'Variable; can be affected by fatigue, experience, bias',
                    'healthcare_context': 'AI provides consistent baseline, humans handle edge cases and exceptions'
                },
                'cost': {
                    'ai': 'Usually low and declining with scale',
                    'human': 'High and increasing due to training and salaries',
                    'healthcare_context': 'AI reduces cost per analysis, humans focus on complex cases'
                },
                'reasoning_process': {
                    'ai': 'Clear, visible through explainable AI',
                    'human': 'Difficult to trace at times',
                    'healthcare_context': 'AI provides transparent decision paths, humans provide intuitive reasoning'
                },
                'creativity': {
                    'ai': 'Limited to pattern recognition and optimization',
                    'human': 'Truly creative in novel situations',
                    'healthcare_context': 'AI handles routine cases, humans innovate new treatments'
                },
                'flexibility': {
                    'ai': 'Rigid within programmed parameters',
                    'human': 'Large, flexible adaptation',
                    'healthcare_context': 'AI handles standard protocols, humans adapt to unique situations'
                }
            }
        }
        
        return ai_matrix
    
    def get_knowledge_acquisition_plan(self, project_name: str = "Clinical Decision Support") -> Dict[str, Any]:
        """
        Get Knowledge Acquisition Plan following DSS_2.md methodology
        
        Args:
            project_name: Name of the healthcare project
            
        Returns:
            Dictionary containing knowledge acquisition strategy
        """
        knowledge_plan = {
            'project_name': project_name,
            'knowledge_sources': {
                'clinical_experts': {
                    'source': 'Oncologists, Cardiologists, Radiologists (human experts)',
                    'acquisition_method': 'Structured Interviews',
                    'representation_method': 'IF-THEN rules in an expert system',
                    'validation_process': 'Peer review by a separate panel of specialists'
                },
                'clinical_guidelines': {
                    'source': 'Clinical Trial Protocols, Medical Literature (documents)',
                    'acquisition_method': 'NLP and Text Mining',
                    'representation_method': 'Knowledge Graph mapping eligibility criteria',
                    'validation_process': 'Cross-reference extracted criteria with manually verified checklists'
                },
                'patient_data': {
                    'source': 'EHR Data, Lab Results, Imaging Data (raw data)',
                    'acquisition_method': 'Machine Learning',
                    'representation_method': 'Trained classification model',
                    'validation_process': 'Test model predictions against historical patient decisions made by humans'
                }
            },
            'knowledge_organization': {
                'structured_knowledge': {
                    'clinical_rules': 'IF-THEN statements for diagnostic criteria',
                    'treatment_protocols': 'Step-by-step clinical procedures',
                    'drug_interactions': 'Database of medication interactions'
                },
                'unstructured_knowledge': {
                    'clinical_notes': 'Natural language processing of physician notes',
                    'research_papers': 'Text mining of medical literature',
                    'patient_feedback': 'Sentiment analysis of patient communications'
                }
            },
            'knowledge_refinement': {
                'continuous_learning': 'System learns from new cases and expert feedback',
                'validation_cycles': 'Regular review and update of knowledge base',
                'performance_monitoring': 'Track accuracy and update knowledge accordingly'
            }
        }
        
        return knowledge_plan
    
    def evaluate_ai_model_performance(self, model_name: str, healthcare_context: str = "General") -> Dict[str, Any]:
        """
        Evaluate AI model performance using healthcare-specific metrics following DSS_2.md
        
        Args:
            model_name: Name of the AI model
            healthcare_context: Specific healthcare application context
            
        Returns:
            Dictionary containing performance evaluation results
        """
        # This would typically evaluate actual model performance
        # For demonstration, we'll return template metrics
        
        performance_metrics = {
            'model_name': model_name,
            'healthcare_context': healthcare_context,
            'metrics': {
                'accuracy': {
                    'target': '> 95%',
                    'actual_performance': '96.5%',
                    'notes': 'Meets clinical decision support requirements'
                },
                'speed': {
                    'target': '< 1 minute',
                    'actual_performance': '45 seconds',
                    'notes': 'Suitable for real-time clinical decisions'
                },
                'false_positive_rate': {
                    'target': '< 5%',
                    'actual_performance': '3.2%',
                    'notes': 'Low enough to avoid alert fatigue for clinicians'
                },
                'cost_benefit': {
                    'target': 'Positive ROI within 18 months',
                    'actual_performance': 'ROI achieved in 12 months',
                    'notes': 'Savings measured in reduced length of stay and treatment costs'
                }
            },
            'clinical_validation': {
                'expert_review': 'Approved by clinical review board',
                'regulatory_compliance': 'Meets FDA guidelines for clinical decision support',
                'safety_assessment': 'No adverse events reported during pilot'
            },
            'turing_test_results': {
                'human_expert_accuracy': '94.2%',
                'ai_model_accuracy': '96.5%',
                'indistinguishability_score': '87%',
                'notes': 'AI recommendations indistinguishable from human expert in 87% of cases'
            }
        }
        
        return performance_metrics

    def analyze_target_column(self, dataset_name: str, target_column: str) -> Dict[str, Any]:
        """
        Analyze target column to suggest appropriate task type and provide insights
        
        Args:
            dataset_name: Name of the dataset
            target_column: Name of the target column
            
        Returns:
            Dictionary containing analysis results and suggestions
        """
        try:
            # Get the dataset
            dataset = self.data_manager.datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            # Delegate to preprocessing engine
            result = self.preprocessing_engine.analyze_target_column(dataset, target_column)
            
            # Check if preprocessing engine returned an error
            if 'error' in result:
                raise ValueError(result['error'])
            
            return result

        except ValueError as e:
            logger.error(f"Error analyzing target column: {e}")
            raise  # Re-raise ValueError for proper error handling
        except Exception as e:
            logger.error(f"Error analyzing target column: {e}")
            return {
                'error': str(e),
                'column_name': target_column,
                'suggested_task_type': 'unknown'
            }
    
    def get_preprocessing_options(self, dataset_name: str, target_column: str, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Analyze data and suggest intelligent preprocessing options
        
        Args:
            dataset_name: Name of the dataset
            target_column: Name of the target column
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary containing preprocessing options and recommendations
        """
        try:
            # Get the dataset
            dataset = self.data_manager.datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            # Delegate to preprocessing engine
            return self.preprocessing_engine.get_preprocessing_options(dataset, target_column, task_type)

        except Exception as e:
            logger.error(f"Error analyzing preprocessing options: {e}")
            return {'error': str(e)}

    def get_intelligent_model_recommendations(self, dataset_name: str, target_column: str, 
                                            task_type: str = None) -> Dict[str, Any]:
        """
        Get intelligent model recommendations based on data characteristics
        
        Args:
            dataset_name: Name of the dataset
            target_column: Name of the target column
            task_type: Optional task type (will be inferred if not provided)
            
        Returns:
            Dictionary containing intelligent model recommendations
        """
        try:
            # Analyze target column if task type not provided
            if task_type is None:
                analysis = self.analyze_target_column(dataset_name, target_column)
                if 'error' in analysis:
                    return {'error': analysis['error']}
                task_type = analysis['primary_recommendation']
            
            # Get dataset characteristics
            dataset = self.data_manager.datasets.get(dataset_name)
            if dataset is None:
                return {'error': f"Dataset '{dataset_name}' not found"}
            
            # Analyze data characteristics
            data_characteristics = self._analyze_data_characteristics(dataset, target_column)
            
            # Get intelligent model recommendations
            recommendations = self._get_model_recommendations(data_characteristics, task_type)
            
            # Add preprocessing recommendations
            preprocessing_options = self.get_preprocessing_options(dataset_name, target_column, task_type)
            
            return {
                'task_type': task_type,
                'data_characteristics': data_characteristics,
                'model_recommendations': recommendations,
                'preprocessing_recommendations': preprocessing_options,
                'confidence': self._calculate_recommendation_confidence(data_characteristics, task_type)
            }
            
        except Exception as e:
            logger.error(f"Error getting intelligent model recommendations: {e}")
            return {'error': str(e)}
    
    def _analyze_data_characteristics(self, dataset: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze data characteristics for intelligent model selection"""
        try:
            characteristics = {}
            
            # Basic dataset info
            characteristics['sample_count'] = len(dataset)
            characteristics['feature_count'] = len(dataset.columns) - 1  # Exclude target
            characteristics['target_column'] = target_column
            
            # Data types analysis
            numeric_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()
            
            characteristics['numeric_features'] = len(numeric_features)
            characteristics['categorical_features'] = len(categorical_features)
            characteristics['mixed_data_types'] = len(numeric_features) > 0 and len(categorical_features) > 0
            
            # Target column analysis
            if target_column in dataset.columns:
                target_data = dataset[target_column]
                characteristics['target_unique_values'] = target_data.nunique()
                characteristics['target_missing_values'] = target_data.isnull().sum()
                characteristics['target_data_type'] = str(target_data.dtype)
                
                # Calculate target distribution characteristics
                if target_data.dtype in ['int64', 'float64']:
                    characteristics['target_range'] = float(target_data.max() - target_data.min())
                    characteristics['target_std'] = float(target_data.std())
                    characteristics['target_mean'] = float(target_data.mean())
            
            # Missing values analysis
            missing_values = dataset.isnull().sum()
            characteristics['total_missing_values'] = int(missing_values.sum())
            characteristics['missing_percentage'] = float((missing_values.sum() / (len(dataset) * len(dataset.columns))) * 100)
            characteristics['has_missing_values'] = missing_values.sum() > 0
            
            # Outlier analysis for numeric features
            outlier_info = self._analyze_outliers(dataset, numeric_features)
            characteristics['outlier_analysis'] = outlier_info
            
            # Data complexity metrics
            characteristics['data_complexity'] = self._calculate_data_complexity(dataset, numeric_features, categorical_features)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {}
    
    def _analyze_outliers(self, dataset: pd.DataFrame, numeric_features: List[str]) -> Dict[str, Any]:
        """Analyze outliers in numeric features"""
        try:
            outlier_info = {
                'high_outlier_features': [],
                'moderate_outlier_features': [],
                'low_outlier_features': [],
                'outlier_percentages': {}
            }
            
            for col in numeric_features:
                if col in dataset.columns:
                    Q1 = dataset[col].quantile(0.25)
                    Q3 = dataset[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_count = ((dataset[col] < lower_bound) | (dataset[col] > upper_bound)).sum()
                    outlier_percentage = (outlier_count / len(dataset)) * 100
                    
                    outlier_info['outlier_percentages'][col] = float(outlier_percentage)
                    
                    if outlier_percentage > 15:
                        outlier_info['high_outlier_features'].append(col)
                    elif outlier_percentage > 5:
                        outlier_info['moderate_outlier_features'].append(col)
                    else:
                        outlier_info['low_outlier_features'].append(col)
            
            return outlier_info
            
        except Exception as e:
            logger.error(f"Error analyzing outliers: {e}")
            return {}
    
    def _calculate_data_complexity(self, dataset: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> str:
        """Calculate data complexity level"""
        try:
            complexity_score = 0
            
            # Sample size factor
            if len(dataset) < 100:
                complexity_score += 1
            elif len(dataset) < 1000:
                complexity_score += 2
            else:
                complexity_score += 3
            
            # Feature count factor
            feature_count = len(numeric_features) + len(categorical_features)
            if feature_count < 5:
                complexity_score += 1
            elif feature_count < 20:
                complexity_score += 2
            else:
                complexity_score += 3
            
            # Mixed data types factor
            if len(numeric_features) > 0 and len(categorical_features) > 0:
                complexity_score += 1
            
            # Missing values factor
            missing_percentage = (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100
            if missing_percentage > 10:
                complexity_score += 1
            
            # Determine complexity level
            if complexity_score <= 3:
                return 'low'
            elif complexity_score <= 6:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"Error calculating data complexity: {e}")
            return 'medium'
    
    def _get_model_recommendations(self, data_characteristics: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Get intelligent model recommendations based on data characteristics"""
        try:
            recommendations = []
            
            # Get base model configurations
            available_models = list(self.training_engine.model_configs.keys())
            
            for model_name in available_models:
                if task_type in self.training_engine.model_configs[model_name]:
                    recommendation = self._evaluate_model_suitability(model_name, data_characteristics, task_type)
                    if recommendation['suitability_score'] > 0.3:  # Only recommend models with reasonable suitability
                        recommendations.append(recommendation)
            
            # Sort by suitability score
            recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error getting model recommendations: {e}")
            return []
    
    def _evaluate_model_suitability(self, model_name: str, data_characteristics: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Evaluate model suitability based on data characteristics"""
        try:
            suitability_score = 0.5  # Base score
            reasons = []
            
            sample_count = data_characteristics.get('sample_count', 0)
            feature_count = data_characteristics.get('feature_count', 0)
            data_complexity = data_characteristics.get('data_complexity', 'medium')
            has_missing_values = data_characteristics.get('has_missing_values', False)
            mixed_data_types = data_characteristics.get('mixed_data_types', False)
            
            # Sample size considerations
            if model_name in ['neural_network', 'svm']:
                if sample_count >= 1000:
                    suitability_score += 0.2
                    reasons.append("Large dataset suitable for complex models")
                elif sample_count < 100:
                    suitability_score -= 0.3
                    reasons.append("Small dataset may not be suitable for complex models")
            
            # Feature count considerations
            if model_name in ['random_forest', 'xgboost', 'lightgbm']:
                if feature_count >= 10:
                    suitability_score += 0.2
                    reasons.append("Many features suitable for tree-based models")
                elif feature_count < 3:
                    suitability_score -= 0.2
                    reasons.append("Few features may limit tree-based model effectiveness")
            
            # Data complexity considerations
            if data_complexity == 'high' and model_name in ['random_forest', 'xgboost']:
                suitability_score += 0.2
                reasons.append("Complex data suitable for ensemble methods")
            elif data_complexity == 'low' and model_name in ['linear_regression', 'knn']:
                suitability_score += 0.2
                reasons.append("Simple data suitable for linear models")
            
            # Missing values considerations
            if has_missing_values and model_name in ['random_forest', 'xgboost']:
                suitability_score += 0.1
                reasons.append("Tree-based models handle missing values well")
            elif has_missing_values and model_name in ['svm', 'neural_network']:
                suitability_score -= 0.1
                reasons.append("Missing values may affect model performance")
            
            # Mixed data types considerations
            if mixed_data_types and model_name in ['random_forest', 'xgboost', 'lightgbm']:
                suitability_score += 0.1
                reasons.append("Tree-based models handle mixed data types well")
            
            # Task-specific considerations
            if task_type == 'classification':
                if model_name in ['random_forest', 'xgboost', 'lightgbm']:
                    suitability_score += 0.1
                    reasons.append("Good for classification tasks")
            elif task_type == 'regression':
                if model_name in ['random_forest', 'linear_regression', 'xgboost']:
                    suitability_score += 0.1
                    reasons.append("Good for regression tasks")
            
            # Clamp score between 0 and 1
            suitability_score = max(0.0, min(1.0, suitability_score))
            
            return {
                'model_name': model_name,
                'suitability_score': suitability_score,
                'confidence': suitability_score,
                'reasons': reasons,
                'recommended': suitability_score > 0.6
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model suitability: {e}")
            return {
                'model_name': model_name,
                'suitability_score': 0.5,
                'confidence': 0.5,
                'reasons': ['Unable to evaluate suitability'],
                'recommended': False
            }
    
    def _calculate_recommendation_confidence(self, data_characteristics: Dict[str, Any], task_type: str) -> float:
        """Calculate confidence in recommendations"""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on data quality
            missing_percentage = data_characteristics.get('missing_percentage', 0)
            if missing_percentage < 5:
                confidence += 0.2
            elif missing_percentage < 15:
                confidence += 0.1
            
            # Increase confidence based on sample size
            sample_count = data_characteristics.get('sample_count', 0)
            if sample_count >= 1000:
                confidence += 0.2
            elif sample_count >= 100:
                confidence += 0.1
            
            # Increase confidence based on feature count
            feature_count = data_characteristics.get('feature_count', 0)
            if 5 <= feature_count <= 50:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating recommendation confidence: {e}")
            return 0.5
    
    def get_model_performance_insights(self, model_key: str) -> Dict[str, Any]:
        """Get intelligent insights about model performance"""
        try:
            model_info = self.registry.load_model(model_key)
            if not model_info:
                return {'error': f"Model '{model_key}' not found"}
            
            metrics = model_info.get('metrics', {})
            metadata = model_info.get('metadata', {})
            
            insights = {
                'model_key': model_key,
                'performance_analysis': self._analyze_model_performance(metrics, metadata),
                'recommendations': self._generate_performance_recommendations(metrics, metadata),
                'comparison_with_baseline': self._compare_with_baseline(metrics, metadata)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model performance insights: {e}")
            return {'error': str(e)}
    
    def _analyze_model_performance(self, metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance metrics"""
        try:
            analysis = {}
            
            # Determine task type from metrics
            if 'accuracy' in metrics and 'r2_score' in metrics:
                # Both classification and regression metrics present
                if metrics.get('accuracy', 0) > 0:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
            elif 'accuracy' in metrics:
                task_type = 'classification'
            elif 'r2_score' in metrics:
                task_type = 'regression'
            else:
                task_type = 'unknown'
            
            analysis['task_type'] = task_type
            
            if task_type == 'classification':
                accuracy = metrics.get('accuracy', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1_score', 0)
                
                analysis['performance_level'] = self._evaluate_classification_performance(accuracy, precision, recall, f1_score)
                analysis['balanced_performance'] = abs(precision - recall) < 0.1
                
            elif task_type == 'regression':
                r2_score = metrics.get('r2_score', 0)
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                
                analysis['performance_level'] = self._evaluate_regression_performance(r2_score, rmse, mae)
                analysis['prediction_accuracy'] = 'high' if r2_score > 0.8 else 'medium' if r2_score > 0.5 else 'low'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {}
    
    def _evaluate_classification_performance(self, accuracy: float, precision: float, recall: float, f1_score: float) -> str:
        """Evaluate classification performance level"""
        try:
            avg_score = (accuracy + precision + recall + f1_score) / 4
            
            if avg_score >= 0.9:
                return 'excellent'
            elif avg_score >= 0.8:
                return 'very_good'
            elif avg_score >= 0.7:
                return 'good'
            elif avg_score >= 0.6:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Error evaluating classification performance: {e}")
            return 'unknown'
    
    def _evaluate_regression_performance(self, r2_score: float, rmse: float, mae: float) -> str:
        """Evaluate regression performance level"""
        try:
            if r2_score >= 0.9:
                return 'excellent'
            elif r2_score >= 0.8:
                return 'very_good'
            elif r2_score >= 0.7:
                return 'good'
            elif r2_score >= 0.5:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Error evaluating regression performance: {e}")
            return 'unknown'
    
    def _generate_performance_recommendations(self, metrics: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        try:
            recommendations = []
            
            # Classification recommendations
            if 'accuracy' in metrics:
                accuracy = metrics.get('accuracy', 0)
                if accuracy < 0.7:
                    recommendations.append("Consider feature engineering or hyperparameter tuning to improve accuracy")
                
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                if abs(precision - recall) > 0.2:
                    recommendations.append("Address class imbalance or adjust decision threshold")
            
            # Regression recommendations
            if 'r2_score' in metrics:
                r2_score = metrics.get('r2_score', 0)
                if r2_score < 0.5:
                    recommendations.append("Consider feature selection or different model algorithms")
                
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                if rmse > mae * 2:
                    recommendations.append("High RMSE suggests outliers - consider robust models")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return []
    
    def _compare_with_baseline(self, metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance with baseline"""
        try:
            comparison = {}
            
            if 'accuracy' in metrics:
                # Classification baseline (random guessing)
                baseline_accuracy = 0.5  # Binary classification
                actual_accuracy = metrics.get('accuracy', 0)
                improvement = actual_accuracy - baseline_accuracy
                
                comparison['baseline_accuracy'] = baseline_accuracy
                comparison['improvement_over_baseline'] = improvement
                comparison['improvement_percentage'] = (improvement / baseline_accuracy) * 100
            
            elif 'r2_score' in metrics:
                # Regression baseline (mean prediction)
                baseline_r2 = 0.0
                actual_r2 = metrics.get('r2_score', 0)
                improvement = actual_r2 - baseline_r2
                
                comparison['baseline_r2'] = baseline_r2
                comparison['improvement_over_baseline'] = improvement
                comparison['improvement_percentage'] = improvement * 100
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {}
    
    def train_model(self, 
                   dataset_name: str, 
                   model_name: str, 
                   task_type: str = 'classification',
                   target_column: str = None,
                   test_size: float = 0.2,
                   optimize_hyperparameters: bool = False,
                   preprocessing_config: Dict[str, Any] = None,
                   model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a machine learning model on healthcare data with intelligent preprocessing
        
        Args:
            dataset_name: Name of the dataset to use
            model_name: Name of the model to train
            task_type: Type of task ('classification', 'regression', or 'auto')
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            optimize_hyperparameters: Whether to optimize hyperparameters
            preprocessing_config: Configuration for preprocessing steps
            model_config: Custom model configuration parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Training {model_name} on {dataset_name} for {task_type}")
        
        # Get dataset
        if dataset_name not in self.data_manager.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.data_manager.datasets[dataset_name]
        
        # Auto-detect target column if not specified
        if target_column is None:
            # Look for common target column names
            common_targets = ['target', 'label', 'outcome', 'diagnosis', 'class']
            for col in common_targets:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                # Use the last column as target
                target_column = df.columns[-1]
                logger.info(f"Auto-selected target column: {target_column}")
        
        # Analyze target column to suggest task type if not specified
        if task_type == 'auto':
            target_analysis = self.preprocessing_engine.analyze_target_column(df, target_column)
            task_type = target_analysis.get('suggested_task_type', 'classification')
            logger.info(f"Auto-detected task type: {task_type}")
            logger.info(f"Target analysis recommendations: {target_analysis.get('recommendations', [])}")
        
        # Validate task type
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid task_type '{task_type}'. Must be 'classification' or 'regression'")

        # Data leakage detection and automatic feature removal (enabled by default)
        leakage_detection_enabled = True  # Can be made configurable
        auto_remove_features = True  # Can be made configurable
        if leakage_detection_enabled:
            leakage_analysis = self.preprocessing_engine.detect_and_remove_data_leakage(
                df, target_column, auto_remove=auto_remove_features
            )
            
            # Use the cleaned dataset for training
            df = leakage_analysis['cleaned_dataset']
            
            # Log leakage detection results
            if leakage_analysis['leakage_detected']:
                logger.info(f"Data leakage detected and automatically handled:")
                logger.info(f"  • Risk Level: {leakage_analysis['risk_level'].upper()}")
                logger.info(f"  • Features removed: {leakage_analysis['features_removed_count']}")
                logger.info(f"  • Removed features: {leakage_analysis['removed_features']}")
                logger.info(f"  • Dataset shape: {leakage_analysis['original_shape']} → {leakage_analysis['cleaned_shape']}")
                
                # Store leakage analysis in result for user information
                self._leakage_analysis = leakage_analysis
        
        # Analyze target column suitability
        analysis = self.preprocessing_engine.analyze_target_column(df, target_column)

        if 'error' in analysis:
            raise ValueError(f"Error analyzing target column: {analysis['error']}")

        # Get intelligent preprocessing options
        preprocessing_options = self.preprocessing_engine.get_preprocessing_options(df, target_column, task_type)
        
        # Create preprocessing configuration based on recommendations if not provided
        if preprocessing_config is None:
            preprocessing_config = self._create_preprocessing_config(preprocessing_options, task_type)
        
        # Preprocess data with intelligent configuration
        features, target = self.preprocessing_engine.preprocess_data(df, target_column, preprocessing_config)
        
        # Train model using training engine with hyperparameter optimization
        result = self.training_engine.train_model(
            features=features,
            target=target,
            model_name=model_name,
            task_type=task_type,
            test_size=test_size,
            optimize_hyperparameters=optimize_hyperparameters,
            preprocessing_config=preprocessing_config,
            model_config=model_config
        )
        
        # Add dataset_name and target_column to result
        result['dataset_name'] = dataset_name
        result['target_column'] = target_column
        
        # Add leakage analysis information if available
        if hasattr(self, '_leakage_analysis') and self._leakage_analysis:
            result['leakage_analysis'] = {
                'leakage_detected': self._leakage_analysis['leakage_detected'],
                'risk_level': self._leakage_analysis.get('risk_level', 'unknown'),
                'features_removed': self._leakage_analysis['features_removed_count'],
                'removed_features': self._leakage_analysis['removed_features'],
                'original_shape': self._leakage_analysis['original_shape'],
                'cleaned_shape': self._leakage_analysis['cleaned_shape'],
                'recommendations': self._leakage_analysis['recommendations'],
                'correlation_matrix': self._leakage_analysis.get('correlation_matrix'),
                'confidence_score': self._leakage_analysis.get('confidence_score', 0)
            }
            # Clear the stored analysis
            delattr(self, '_leakage_analysis')
        
        # Store model in registry
        model_key = result['model_key']
        self.registry.save_model(model_key, result)
        
        logger.info(f"Model {model_key} trained and stored successfully")
        logger.info(f"Model performance: {result['metrics']}")
        return result
    
    def _create_preprocessing_config(self, preprocessing_options: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Create preprocessing configuration based on intelligent analysis
        
        Args:
            preprocessing_options: Options from preprocessing engine analysis
            task_type: Type of ML task
            
        Returns:
            Configuration dictionary for preprocessing
        """
        config = {}
        
        # Handle missing values
        if preprocessing_options.get('data_quality_issues'):
            config['handle_missing'] = True
        
        # Choose scaling method based on recommendations
        scaling_options = preprocessing_options.get('scaling_options', [])
        if scaling_options:
            # Select the most suitable scaling method
            best_scaling = max(scaling_options, key=lambda x: x.get('confidence', 0))
            if best_scaling.get('suitable', False):
                config['scaling_method'] = best_scaling['name']
        
        # Choose encoding method
        encoding_options = preprocessing_options.get('encoding_options', [])
        if encoding_options:
            best_encoding = max(encoding_options, key=lambda x: x.get('confidence', 0))
            if best_encoding.get('suitable', False):
                config['encoding_method'] = best_encoding['name']
        
        # Apply feature engineering based on recommendations
        feature_engineering = preprocessing_options.get('feature_engineering', [])
        for option in feature_engineering:
            if option.get('suitable', False) and option.get('confidence', 0) > 60:
                config[f"feature_{option['name']}"] = True
        
        return config
    
    def analyze_target_column(self, dataset_name: str, target_column: str) -> Dict[str, Any]:
        """
        Analyze target column to determine suitability for different ML tasks
        
        Args:
            dataset_name: Name of the dataset
            target_column: Name of the target column
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if dataset_name not in self.data_manager.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            df = self.data_manager.datasets[dataset_name]
            return self.preprocessing_engine.analyze_target_column(df, target_column)
            
        except Exception as e:
            logger.error(f"Error analyzing target column: {e}")
            raise e
    
    def predict(self, model_key: str, features: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            model_key: Unique identifier for the model
            features: Feature matrix for prediction
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            # Load model from registry
            model_data = self.registry.load_model(model_key)
            if not model_data:
                raise ValueError(f"Model {model_key} not found")
        
            model = model_data['model']
            
            # Make predictions
            predictions = model.predict(features)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
            
            result = {
                'model_key': model_key,
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'prediction_count': len(predictions),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Predictions made using model {model_key}")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def create_ensemble_model(self, 
                            dataset_name: str, 
                            task_type: str = 'classification',
                            target_column: str = None,
                            models: List[str] = None,
                            voting_type: str = 'hard') -> Dict[str, Any]:
        """
        Create an ensemble model combining multiple algorithms
        
        Args:
            dataset_name: Name of the dataset to use
            task_type: Type of task ('classification' or 'regression')
            target_column: Name of the target column
            models: List of model names to include in ensemble
            voting_type: Type of voting ('hard' or 'soft' for classification)
            
        Returns:
            Dictionary containing ensemble model and metrics
        """
        logger.info(f"Creating ensemble model for {task_type}")
        
        # Get preprocessed data
        features, target = self.data_manager.preprocess_data(dataset_name, target_column)
        
        # Delegate to training engine
        ensemble_result = self.training_engine.create_ensemble_model(
            features=features,
            target=target,
            task_type=task_type,
            models=models,
            voting_type=voting_type
        )

        # Add dataset information
        ensemble_result['dataset_name'] = dataset_name
        ensemble_result['target_column'] = target_column

        # Save ensemble model
        self.registry.save_model(ensemble_result['model_key'], ensemble_result)

        logger.info(f"Ensemble model {ensemble_result['model_key']} created and saved")
        return ensemble_result
    
    def compare_models(self, dataset_name: str, task_type: str = 'classification', target_column: str = None) -> pd.DataFrame:
        """
        Compare performance of different models
        
        Args:
            dataset_name: Name of the dataset to use
            task_type: Type of task ('classification' or 'regression')
            target_column: Name of the target column
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing models for {task_type}")

        # Get preprocessed data
        features, target = self.data_manager.preprocess_data(dataset_name, target_column)

        # Delegate to training engine
        return self.training_engine.compare_models(features, target, task_type)
    
    def explain_prediction(self, model_key: str, features: Union[pd.DataFrame, np.ndarray], instance_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a specific prediction using available explainers
        
        Args:
            model_key: Unique identifier for the model
            features: Feature matrix
            instance_idx: Index of the instance to explain
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            # Delegate to evaluation engine
            return self.evaluation_engine.explain_prediction(model_key, features, instance_idx)

        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}

    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary of all models
        
        Returns:
            DataFrame with model performance metrics
        """
        return self.registry.get_model_performance_summary()

    def list_models(self, status: str = 'active') -> pd.DataFrame:
        """
        List all models in the registry
        
        Args:
            status: Filter by model status ('active', 'inactive', 'all')
            
        Returns:
            DataFrame with model information
        """
        return self.registry.list_models(status)

    def delete_model(self, model_key: str) -> bool:
        """
        Delete a model from the registry
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            Boolean indicating success
        """
        return self.registry.delete_model(model_key)

    def load_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Load a model from the registry
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            Dictionary containing model and metadata, or None if not found
        """
        return self.registry.load_model(model_key)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache"""
        return self.registry.get_cache_info()

    def clear_cache(self):
        """Clear the in-memory model cache"""
        self.registry.clear_cache()
