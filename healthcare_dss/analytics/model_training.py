"""
Model Training Engine for Healthcare DSS
=======================================

This module handles all machine learning model training, hyperparameter
optimization, and ensemble model creation capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna

# Configure logging
logger = logging.getLogger(__name__)

# Import intelligent task detection
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently


class ModelTrainingEngine:
    """
    Advanced model training engine for healthcare data analysis
    
    Provides comprehensive model training, hyperparameter optimization,
    and ensemble modeling capabilities.
    """
    
    def __init__(self):
        """Initialize the model training engine"""
        self.model_configs = self._init_model_configs()
        
    def _init_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations for different algorithms"""
        return {
            'random_forest': {
                'classification': {
                    'model': RandomForestClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': 42
                    }
                },
                'regression': {
                    'model': RandomForestRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': 42
                    }
                }
            },
            'svm': {
                'classification': {
                    'model': SVC,
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto'],
                        'random_state': 42
                    }
                },
                'regression': {
                    'model': SVR,
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                }
            },
            'neural_network': {
                'classification': {
                    'model': MLPClassifier,
                    'params': {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive'],
                        'random_state': 42,
                        'max_iter': 2000
                    }
                },
                'regression': {
                    'model': MLPRegressor,
                    'params': {
                        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate': ['constant', 'adaptive'],
                        'random_state': 42,
                        'max_iter': 2000
                    }
                }
            },
            'knn': {
                'classification': {
                    'model': KNeighborsClassifier,
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                },
                'regression': {
                    'model': KNeighborsRegressor,
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                }
            },
            'linear_regression': {
                'regression': {
                    'model': LinearRegression,
                    'params': {
                        'fit_intercept': [True, False]
                    }
                }
            },
            'logistic_regression': {
                'classification': {
                    'model': LogisticRegression,
                    'params': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga'],
                        'random_state': 42,
                        'max_iter': 2000
                    }
                }
            },
            'decision_tree': {
                'classification': {
                    'model': DecisionTreeClassifier,
                    'params': {
                        'max_depth': [None, 5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': 42
                    }
                },
                'regression': {
                    'model': DecisionTreeRegressor,
                    'params': {
                        'max_depth': [None, 5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': 42
                    }
                }
            },
            'xgboost': {
                'classification': {
                    'model': xgb.XGBClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'random_state': 42
                    }
                },
                'regression': {
                    'model': xgb.XGBRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'random_state': 42
                    }
                }
            },
            'lightgbm': {
                'classification': {
                    'model': lgb.LGBMClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'random_state': 42,
                        'verbose': -1
                    }
                },
                'regression': {
                    'model': lgb.LGBMRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'random_state': 42,
                        'verbose': -1
                    }
                }
            }
        }
    
    def train_model(self, 
                   features: pd.DataFrame,
                   target: pd.Series,
                   model_name: str, 
                   task_type: str = 'classification',
                   test_size: float = 0.2,
                   optimize_hyperparameters: bool = False,
                   preprocessing_config: Dict[str, Any] = None,
                   model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a machine learning model on healthcare data
        
        Args:
            features: Feature matrix
            target: Target variable
            model_name: Name of the model to train
            task_type: Type of task ('classification' or 'regression')
            test_size: Proportion of data to use for testing
            optimize_hyperparameters: Whether to optimize hyperparameters
            preprocessing_config: Configuration for preprocessing steps
            model_config: Custom model configuration parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Training {model_name} for {task_type}")

        # Validate task type
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid task_type '{task_type}'. Must be 'classification' or 'regression'")

        # Check if model supports the task type
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model '{model_name}'. Available models: {list(self.model_configs.keys())}")
        
        if task_type not in self.model_configs[model_name]:
            raise ValueError(f"Model '{model_name}' does not support '{task_type}' task")

        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42, stratify=target if task_type == 'classification' else None
            )

            # Get model configuration
            base_model_config = self.model_configs[model_name][task_type]
            model_class = base_model_config['model']
            default_params = base_model_config['params']

            # Merge custom model configuration if provided
            if model_config:
                # Override default parameters with custom ones
                for param, value in model_config.items():
                    if param in default_params:
                        default_params[param] = value
                    else:
                        # Add new parameters
                        default_params[param] = value

            # Filter out training configuration parameters that shouldn't be passed to model
            training_params = {'test_size', 'random_state', 'optimize_hyperparameters', 'cv_folds', 'stratify_cv'}
            model_params = {k: v for k, v in default_params.items() if k not in training_params}

            # Initialize model with filtered parameters
            model = model_class(**{k: v[0] if isinstance(v, list) else v for k, v in model_params.items()})

            # Optimize hyperparameters if requested
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(model, X_train, y_train, task_type, model_params)

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, task_type)

            # Get feature importance if available
            feature_importance = self._get_feature_importance(model, features.columns.tolist())

            # Create model key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_key = f"{model_name}_{task_type}_{timestamp}"

            result = {
                'model_key': model_key,
                'model_name': model_name,
                'task_type': task_type,
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_data': {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred': y_pred
                },
                'preprocessing_config': preprocessing_config,
                'timestamp': timestamp
            }

            logger.info(f"Model {model_key} trained successfully with metrics: {metrics}")
            return result

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def _optimize_hyperparameters(self, model, X_train, y_train, task_type, param_grid):
        """Optimize hyperparameters using GridSearchCV"""
        try:
            # Use a subset of parameters for faster optimization
            limited_params = {}
            for param, values in param_grid.items():
                if isinstance(values, list) and len(values) > 2:
                    # Take first 2 values for faster optimization
                    limited_params[param] = values[:2]
                elif isinstance(values, list):
                    limited_params[param] = values
                else:
                    # Single values need to be wrapped in a list
                    limited_params[param] = [values]

            grid_search = GridSearchCV(
                model, limited_params, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}. Using default parameters.")
            return model

    def _calculate_metrics(self, y_true, y_pred, task_type):
        """Calculate appropriate metrics based on task type"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        from sklearn.preprocessing import LabelEncoder
        
        metrics = {}
        
        # Use intelligent task detection to determine actual task type
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(y_true)
        
        logger.debug(f"Analytics metrics calculation - Task type detection:")
        logger.debug(f"  - Input task_type: {task_type}")
        logger.debug(f"  - Detected task_type: {detected_task_type}")
        logger.debug(f"  - Confidence: {confidence:.3f}")
        
        # Use detected task type if confidence is high AND no explicit task type was provided
        # If user explicitly provides task_type, respect their choice
        if task_type and task_type != 'auto':
            actual_task_type = task_type
            logger.debug(f"  - Using explicit task type: {actual_task_type} (user specified)")
        elif confidence > 0.7:
            actual_task_type = detected_task_type
            logger.debug(f"  - Using detected task type: {actual_task_type} (confidence: {confidence:.3f})")
        else:
            actual_task_type = task_type if task_type else detected_task_type
            logger.debug(f"  - Using input task type: {actual_task_type} (low confidence: {confidence:.3f})")
        
        # Store detection information
        metrics['detected_task_type'] = detected_task_type
        metrics['detection_confidence'] = confidence
        metrics['detection_analysis'] = analysis_details
        metrics['actual_task_type'] = actual_task_type
        
        if actual_task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Additional classification metrics
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
            except:
                metrics['auc_roc'] = None  # Skip if not applicable
            
            # RMSE and MAE for classification (less common but useful for error magnitude analysis)
            # Encode string labels to numeric values for RMSE/MAE calculation
            try:
                # Convert to numeric if they're strings
                if not pd.api.types.is_numeric_dtype(y_true) or not pd.api.types.is_numeric_dtype(y_pred):
                    le = LabelEncoder()
                    # Fit on both y_true and y_pred to ensure consistent encoding
                    all_values = pd.concat([pd.Series(y_true), pd.Series(y_pred)]).unique()
                    le.fit(all_values)
                    y_true_encoded = le.transform(y_true)
                    y_pred_encoded = le.transform(y_pred)
                else:
                    y_true_encoded = y_true
                    y_pred_encoded = y_pred
                
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true_encoded, y_pred_encoded))
                metrics['mae'] = mean_absolute_error(y_true_encoded, y_pred_encoded)
            except Exception as e:
                logger.warning(f"Could not calculate RMSE/MAE for classification: {e}")
                metrics['rmse'] = None
                metrics['mae'] = None
            
            # Confusion matrix - only generate if we're confident this is classification
            if detected_task_type == 'classification' and confidence > 0.6:
                try:
                    # Ensure both y_true and y_pred are discrete for confusion matrix
                    unique_true = np.unique(y_true)
                    
                    if len(unique_true) <= 20:  # Reasonable number of classes
                        # Convert predictions to discrete classes if needed
                        y_pred_discrete = np.zeros_like(y_pred)
                        for i, pred_val in enumerate(y_pred):
                            # Find closest true value
                            closest_idx = np.argmin(np.abs(unique_true - pred_val))
                            y_pred_discrete[i] = unique_true[closest_idx]
                        
                        logger.debug(f"Converted predictions to discrete classes for confusion matrix")
                        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_discrete).tolist()
                    else:
                        logger.warning(f"Cannot generate confusion matrix: Too many unique values ({len(unique_true)})")
                        metrics['confusion_matrix'] = None
                        
                except Exception as e:
                    logger.warning(f"Could not generate confusion matrix: {str(e)}")
                    metrics['confusion_matrix'] = None
            else:
                logger.info(f"Skipping confusion matrix: Detected task type is {detected_task_type} (confidence: {confidence:.1%})")
                metrics['confusion_matrix'] = None
            
        else:  # regression
            # Core regression metrics
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Additional regression metrics
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
            
            # Adjusted R² (accounts for number of features)
            n = len(y_true)
            p = 1  # Simplified - in practice would need feature count
            adj_r2 = 1 - (1 - metrics['r2_score']) * (n - 1) / (n - p - 1)
            metrics['adjusted_r2'] = adj_r2
            
            # Mean Absolute Scaled Error (MASE) - relative to naive forecast
            naive_mae = np.mean(np.abs(np.diff(y_true)))
            metrics['mase'] = metrics['mae'] / naive_mae if naive_mae > 0 else None
        
        return metrics

    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_names, abs(model.coef_)))
                else:
                    # Multi-class case
                    importance_dict = dict(zip(feature_names, abs(model.coef_).mean(axis=0)))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}

    def create_ensemble_model(self, 
                            features: pd.DataFrame,
                            target: pd.Series,
                            task_type: str = 'classification',
                            models: List[str] = None,
                            voting_type: str = 'hard') -> Dict[str, Any]:
        """
        Create an ensemble model combining multiple algorithms
        
        Args:
            features: Feature matrix
            target: Target variable
            task_type: Type of task ('classification' or 'regression')
            models: List of model names to include in ensemble
            voting_type: Type of voting ('hard' or 'soft' for classification)
            
        Returns:
            Dictionary containing ensemble model and metrics
        """
        logger.info(f"Creating ensemble model for {task_type}")

        if models is None:
            # Default models for ensemble
            if task_type == 'classification':
                models = ['random_forest', 'svm', 'neural_network']
            else:
                models = ['random_forest', 'linear_regression', 'xgboost']

        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, 
                stratify=target if task_type == 'classification' else None
            )

            # Train individual models
            trained_models = []
            for model_name in models:
                if model_name in self.model_configs and task_type in self.model_configs[model_name]:
                    model_config = self.model_configs[model_name][task_type]
                    model_class = model_config['model']
                    default_params = model_config['params']
                    
                    # Initialize model with default parameters
                    model = model_class(**{k: v[0] if isinstance(v, list) else v for k, v in default_params.items()})
                    model.fit(X_train, y_train)
                    trained_models.append((model_name, model))

            if not trained_models:
                raise ValueError("No valid models found for ensemble")

            # Create ensemble
            if task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=trained_models,
                    voting=voting_type
                )
            else:
                ensemble = VotingRegressor(estimators=trained_models)

            # Train ensemble
            ensemble.fit(X_train, y_train)

            # Make predictions
            y_pred = ensemble.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, task_type)

            # Create model key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_key = f"ensemble_{task_type}_{timestamp}"

            result = {
                'model_key': model_key,
                'model_name': 'ensemble',
                'task_type': task_type,
                'model': ensemble,
                'metrics': metrics,
                'individual_models': trained_models,
                'training_data': {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred': y_pred
                },
                'timestamp': timestamp
            }

            logger.info(f"Ensemble model {model_key} created successfully with metrics: {metrics}")
            return result

        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            raise

    def compare_models(self, features: pd.DataFrame, target: pd.Series, 
                      task_type: str = 'classification') -> pd.DataFrame:
        """
        Compare performance of different models
        
        Args:
            features: Feature matrix
            target: Target variable
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing models for {task_type}")

        results = []
        
        for model_name in self.model_configs.keys():
            if task_type in self.model_configs[model_name]:
                try:
                    # Train model with default parameters
                    result = self.train_model(
                        features, target, model_name, task_type, 
                        optimize_hyperparameters=False
                    )
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': result['metrics'].get('accuracy', 0),
                        'R² Score': result['metrics'].get('r2_score', 0),
                        'RMSE': result['metrics'].get('rmse', 0),
                        'MAE': result['metrics'].get('mae', 0),
                        'Precision': result['metrics'].get('precision', 0),
                        'Recall': result['metrics'].get('recall', 0),
                        'F1 Score': result['metrics'].get('f1_score', 0)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
                    results.append({
                        'Model': model_name,
                        'Accuracy': 0,
                        'R² Score': 0,
                        'RMSE': float('inf'),
                        'MAE': float('inf'),
                        'Precision': 0,
                        'Recall': 0,
                        'F1 Score': 0
                    })

        return pd.DataFrame(results)
