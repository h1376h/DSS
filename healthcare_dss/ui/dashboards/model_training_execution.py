"""
Model Training Execution Module
===============================

This module handles model training execution and progress tracking:
- Model training with progress bars
- Cross-validation execution
- Training metrics collection
- Error handling and recovery
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from healthcare_dss.ui.utils.common import (
    display_error_message,
    display_success_message,
    display_warning_message
)
from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.utils.intelligent_task_detection import detect_task_type_intelligently, validate_task_type_with_data

# Import split components
from healthcare_dss.ui.dashboards.model_training_data_preprocessing import (
    prepare_data,
    detect_data_leakage,
    estimate_feature_importance_pre_training,
    detect_leakage_by_importance,
    filter_suspicious_features,
    apply_preprocessing_pipeline,
    apply_legacy_preprocessing
)
from healthcare_dss.ui.dashboards.model_training_model_creation import (
    get_model_configurations,
    create_model
)
from healthcare_dss.ui.dashboards.model_training_metrics import (
    calculate_metrics,
    get_feature_importance,
    display_training_summary
)


def train_model_with_progress():
    """Train model with progress tracking"""
    
    # Check if all required data is available
    if 'selected_data' not in st.session_state or st.session_state.selected_data is None:
        st.error("Please select data first.")
        return None
    
    if 'model_config' not in st.session_state or st.session_state.model_config is None:
        st.error("Please configure model first.")
        return None
    
    selected_data = st.session_state.selected_data
    model_config = st.session_state.model_config
    advanced_settings = st.session_state.get('advanced_settings', {})
    
    # Prepare data
    df = selected_data['dataframe']
    target_column = selected_data['target_column']
    model_type = model_config['model_type']
    task_type = model_config['task_type']
    parameters = model_config['parameters']
    
    # Create progress container
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Initial data preparation (minimal preprocessing)
        status_text.text("Preparing data...")
        progress_bar.progress(10)
        
        # Debug: Check if df is valid
        if df is None:
            raise ValueError("DataFrame is None")
        if not hasattr(df, 'columns'):
            raise ValueError("DataFrame does not have columns attribute")
        
        # Separate features and target (no preprocessing yet)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Initialize feature names from DataFrame columns
        feature_names = list(X.columns)
        
        # Debug: Check if feature_names is properly initialized
        if not feature_names:
            raise ValueError("No features found in DataFrame")
        
        # Step 2: Check if binning is needed BEFORE train-test split
        status_text.text("Analyzing task type...")
        progress_bar.progress(20)
        
        # Use intelligent task type detection
        detected_task_type, confidence, analysis_details = detect_task_type_intelligently(
            y, target_column
        )
        
        debug_manager.log_debug(f"Intelligent task type detection results:")
        debug_manager.log_debug(f"  - Detected type: {detected_task_type}")
        debug_manager.log_debug(f"  - Confidence: {confidence:.3f}")
        debug_manager.log_debug(f"  - Analysis details: {analysis_details}")
        
        # Check if binning is needed for classification on continuous data
        binning_applied = False
        binning_info = None
        y_for_split = y  # Use original y for split initially
        
        if task_type == 'classification' and detected_task_type == 'regression':
            debug_manager.log_debug(f"Classification requested on continuous data - applying intelligent binning")
            status_text.text("Applying intelligent binning...")
            
            try:
                from healthcare_dss.utils.intelligent_binning import intelligent_binning
                
                # Get binning configuration from session state
                binning_config = st.session_state.get('binning_config', {})
                
                if binning_config.get('enabled', False):
                    mode = binning_config.get('mode', 'automatic')
                    
                    debug_manager.log_debug(f"Applying binning with mode: {mode}")
                    
                    if mode == 'advanced_override':
                        # Handle advanced override configuration
                        override_config = binning_config.get('override_config', {})
                        
                        y_binned, binning_info = intelligent_binning.apply_user_override_binning(
                            y.values, override_config
                        )
                        
                        debug_manager.log_debug(f"Advanced override binning applied:")
                        debug_manager.log_debug(f"  - Strategy: {binning_info['strategy']}")
                        debug_manager.log_debug(f"  - User configured: {binning_info.get('user_configured', False)}")
                        
                        if binning_info.get('warning'):
                            st.warning(binning_info['warning'])
                        
                        st.success(f"Applied advanced override binning: {binning_info['strategy']}")
                        
                    else:
                        # Handle standard binning (automatic or manual)
                        strategy = binning_config.get('strategy', 'quantile')
                        n_bins = binning_config.get('n_bins', 3)
                        
                        debug_manager.log_debug(f"Applying standard binning: strategy={strategy}, n_bins={n_bins}")
                        
                        # Apply binning to the full dataset
                        y_binned, binning_info = intelligent_binning.apply_binning(
                            y.values, strategy, n_bins
                        )
                        
                        debug_manager.log_debug(f"Standard binning applied successfully:")
                        debug_manager.log_debug(f"  - Strategy: {strategy}")
                        debug_manager.log_debug(f"  - Bins: {binning_info['n_bins']}")
                        debug_manager.log_debug(f"  - Bin counts: {binning_info['bin_counts']}")
                        debug_manager.log_debug(f"  - Min class count: {min(binning_info['bin_counts'])}")
                        
                        st.success(f"Applied {strategy} binning with {n_bins} bins")
                    
                    # Convert back to pandas Series
                    y_for_split = pd.Series(y_binned, index=y.index)
                    binning_applied = True
                    
                else:
                    # Auto-apply binning with optimal settings
                    debug_manager.log_debug(f"No binning config found, auto-applying optimal binning")
                    
                    needs_binning, analysis = intelligent_binning.detect_binning_need(y.values, 'classification')
                    suggestions = intelligent_binning.suggest_optimal_bins(y.values, analysis)
                    
                    strategy = suggestions['recommended_strategy']
                    n_bins = suggestions['optimal_bins']
                    
                    y_binned, binning_info = intelligent_binning.apply_binning(
                        y.values, strategy, n_bins
                    )
                    
                    y_for_split = pd.Series(y_binned, index=y.index)
                    binning_applied = True
                    
                    debug_manager.log_debug(f"Auto-binning applied:")
                    debug_manager.log_debug(f"  - Strategy: {strategy}")
                    debug_manager.log_debug(f"  - Bins: {binning_info['n_bins']}")
                    debug_manager.log_debug(f"  - Bin counts: {binning_info['bin_counts']}")
                    
                    st.success(f"Auto-applied {strategy} binning with {n_bins} bins")
                    
            except Exception as e:
                debug_manager.log_debug(f"Binning failed: {str(e)}", "ERROR")
                st.error(f"Binning failed: {str(e)}")
                st.error("Please use regression instead of classification for continuous data.")
                return None
        
        # Step 3: Train-test split (AFTER binning if applied)
        status_text.text("Splitting data...")
        progress_bar.progress(30)
        
        test_size = advanced_settings.get('test_size', 0.2)
        random_state = advanced_settings.get('random_state', 42)
        
        # Use binned y for stratification if binning was applied
        stratify_y = y_for_split if task_type == 'classification' else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_for_split, test_size=test_size, random_state=random_state, stratify=stratify_y
        )
        
        # Step 4: Filter suspicious features FIRST (before preprocessing)
        status_text.text("Filtering suspicious features...")
        progress_bar.progress(40)
        
        X_train, X_test, feature_names, filtered_features = filter_suspicious_features(
            X_train, X_test, y_train, y_test, feature_names, advanced_settings, target_column
        )
        
        # Step 5: Pre-training feature validation (aggressive detection)
        status_text.text("Validating features for data leakage...")
        progress_bar.progress(42)
        
        # Run aggressive pre-training validation
        pre_training_leakage = detect_leakage_by_importance(X_train, y_train, feature_names)
        if pre_training_leakage:
            debug_manager.log_debug(f"Pre-training validation detected leakage: {pre_training_leakage}")
        
        # Step 6: Apply preprocessing pipeline (after filtering)
        status_text.text("Applying preprocessing...")
        progress_bar.progress(44)
        
        # Apply data leakage prevention if enabled
        if advanced_settings.get('prevent_data_leakage', True):
            X_train, X_test, feature_names = apply_preprocessing_pipeline(
                X_train, X_test, advanced_settings
            )
        else:
            # Legacy preprocessing (with data leakage)
            X_train, X_test, feature_names = apply_legacy_preprocessing(X_train, X_test)
        
        # Step 7: Detect potential data leakage (after filtering, before preprocessing)
        status_text.text("Checking for data leakage...")
        progress_bar.progress(46)
        
        # Only run traditional leakage detection if we still have DataFrames
        if isinstance(X_train, pd.DataFrame):
            leakage_detected = detect_data_leakage(X_train, X_test, y_train, y_test, feature_names, target_column)
            if leakage_detected:
                debug_manager.log_debug(f"Data leakage detected: {leakage_detected}")
                # Log critical issues
                for feature, details in leakage_detected.items():
                    if details['severity'] in ['critical', 'high']:
                        debug_manager.log_debug(f"CRITICAL: {feature} - {details['type']} - {details['severity']}")
        else:
            leakage_detected = {}
        
        # Step 8: Feature scaling
        status_text.text("Scaling features...")
        progress_bar.progress(48)
        
        scaler = None
        if advanced_settings.get('scale_features', True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Step 9: Final task type validation
        status_text.text("Validating task type...")
        progress_bar.progress(50)
        
        # Use detected task type (unless user explicitly chose classification with binning)
        actual_task_type = task_type if binning_applied else detected_task_type
        
        # Log comprehensive task type detection
        debug_manager.log_debug(f"Task type detection:")
        debug_manager.log_debug(f"  - Target dtype: {y_train.dtype}")
        debug_manager.log_debug(f"  - Unique values: {len(np.unique(y_train))}")
        debug_manager.log_debug(f"  - Unique values list: {np.unique(y_train)[:10]}...")  # Show first 10
        debug_manager.log_debug(f"  - Original task type: {task_type}")
        debug_manager.log_debug(f"  - Detected task type: {detected_task_type}")
        debug_manager.log_debug(f"  - Actual task type: {actual_task_type}")
        debug_manager.log_debug(f"  - Detection confidence: {confidence:.3f}")
        debug_manager.log_debug(f"  - Binning applied: {binning_applied}")
        
        # Update task type
        if task_type != actual_task_type and not binning_applied:
            debug_manager.log_debug(f"Correcting model from {task_type} to {actual_task_type} (confidence: {confidence:.3f})")
            task_type = actual_task_type
            
            # Update model config in session state
            if 'model_config' in st.session_state:
                st.session_state.model_config['task_type'] = actual_task_type
        
        # Always create model with detected task type
        debug_manager.log_debug(f"Creating model: {model_type} for {task_type} task")
        model = create_model(model_type, parameters, task_type)
        
        # Step 10: Feature selection
        status_text.text("Selecting features...")
        progress_bar.progress(55)
        
        feature_selector = None
        if advanced_settings.get('feature_selection', False):
            max_features = advanced_settings.get('max_features', 20)
            feature_selector = SelectKBest(
                score_func=f_classif if task_type == 'classification' else f_regression,
                k=min(max_features, X_train.shape[1])
            )
            X_train = feature_selector.fit_transform(X_train, y_train)
            X_test = feature_selector.transform(X_test)
        
        # Step 11: Cross-validation
        status_text.text("Running cross-validation...")
        progress_bar.progress(70)
        
        cv_folds = advanced_settings.get('cv_folds', 5)
        
        try:
            # Try cross-validation with detected task type
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
            debug_manager.log_debug(f"Cross-validation successful with {task_type} scoring")
        except Exception as e:
            debug_manager.log_debug(f"Cross-validation failed with {task_type}: {str(e)}")
            
            # If cross-validation fails, try the opposite task type
            fallback_task_type = 'regression' if task_type == 'classification' else 'classification'
            debug_manager.log_debug(f"Trying fallback task type: {fallback_task_type}")
            
            try:
                # Recreate model with fallback task type
                task_type = fallback_task_type
                model = create_model(model_type, parameters, task_type)
                
                # Try cross-validation again
                scoring = 'accuracy' if task_type == 'classification' else 'r2'
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                debug_manager.log_debug(f"Cross-validation successful with fallback {task_type} scoring")
                
                # Update session state
                if 'model_config' in st.session_state:
                    st.session_state.model_config['task_type'] = task_type
                    
            except Exception as e2:
                debug_manager.log_debug(f"Cross-validation failed with fallback {task_type}: {str(e2)}")
                # Use default scores as fallback
                cv_scores = np.array([0.0] * cv_folds)
        
        # Step 12: Model training
        status_text.text("Training model...")
        progress_bar.progress(80)
        
        # Add a simple data validation test
        debug_manager.log_debug(f"Data validation before training:")
        debug_manager.log_debug(f"  - X_train shape: {X_train.shape}")
        debug_manager.log_debug(f"  - y_train shape: {y_train.shape}")
        debug_manager.log_debug(f"  - X_train has NaN: {np.isnan(X_train).any()}")
        debug_manager.log_debug(f"  - y_train has NaN: {np.isnan(y_train).any()}")
        debug_manager.log_debug(f"  - X_train min/max: {np.min(X_train)}/{np.max(X_train)}")
        debug_manager.log_debug(f"  - y_train unique: {np.unique(y_train)}")
        
        # Test with a simple model first to see if the issue is with the data
        if task_type == 'classification':
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
            dummy_model.fit(X_train, y_train)
            dummy_pred = dummy_model.predict(X_test)
            dummy_accuracy = np.mean(dummy_pred == y_test)
            debug_manager.log_debug(f"Dummy classifier accuracy: {dummy_accuracy:.3f}")
            
            if dummy_accuracy == 0.0:
                debug_manager.log_debug(f"ERROR: Even dummy classifier gets 0 accuracy!")
                debug_manager.log_debug(f"This suggests a fundamental data issue")
        
        # Check if target variable needs encoding for classification
        if task_type == 'classification':
            # Check if target is properly encoded
            unique_targets = np.unique(y_train)
            debug_manager.log_debug(f"Target encoding check:")
            debug_manager.log_debug(f"  - Unique targets: {unique_targets}")
            debug_manager.log_debug(f"  - Target dtype: {y_train.dtype}")
            debug_manager.log_debug(f"  - Target min/max: {np.min(y_train)}/{np.max(y_train)}")
            
            # If target is not numeric, encode it
            if y_train.dtype == 'object' or not np.issubdtype(y_train.dtype, np.number):
                debug_manager.log_debug(f"Target is not numeric, encoding for classification")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
                debug_manager.log_debug(f"Target after encoding: {np.unique(y_train)}")
            elif len(unique_targets) == 2 and not np.allclose(unique_targets, [0, 1]):
                # Binary classification with non-standard encoding (e.g., [-0.044, 0.051])
                debug_manager.log_debug(f"Binary classification with non-standard encoding, converting to [0, 1]")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
                debug_manager.log_debug(f"Target after encoding: {np.unique(y_train)}")
        
        try:
            start_time = time.time()
            debug_manager.log_debug(f"Starting model training with {task_type}")
            debug_manager.log_debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            debug_manager.log_debug(f"y_train unique values: {np.unique(y_train)}")
            debug_manager.log_debug(f"Model parameters: {parameters}")
            
            # Test with a simple model first to isolate the issue
            if task_type == 'classification':
                from sklearn.linear_model import LogisticRegression
                test_model = LogisticRegression(random_state=42, max_iter=1000)
                test_model.fit(X_train, y_train)
                test_pred = test_model.predict(X_test)
                test_accuracy = np.mean(test_pred == y_test)
                debug_manager.log_debug(f"Simple LogisticRegression test accuracy: {test_accuracy:.3f}")
                
                if test_accuracy == 0.0:
                    debug_manager.log_debug(f"ERROR: Even simple LogisticRegression gets 0 accuracy!")
                    debug_manager.log_debug(f"This confirms a fundamental data issue")
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            debug_manager.log_debug(f"Model training successful with {task_type} in {training_time:.2f}s")
            
            # Test prediction on training data to see if model learned anything
            train_pred = model.predict(X_train[:5])  # Test on first 5 samples
            debug_manager.log_debug(f"Training prediction test: {train_pred}")
            
            # Check if model can at least memorize training data
            train_pred_all = model.predict(X_train)
            train_accuracy = np.mean(train_pred_all == y_train) if task_type == 'classification' else None
            debug_manager.log_debug(f"Training accuracy (memorization test): {train_accuracy}")
            
            if task_type == 'classification' and train_accuracy is not None:
                if train_accuracy < 0.5:
                    debug_manager.log_debug(f"WARNING: Model cannot even memorize training data (accuracy: {train_accuracy:.3f})")
                    debug_manager.log_debug(f"This suggests a fundamental problem with the model or data")
            
        except Exception as e:
            debug_manager.log_debug(f"Model training failed with {task_type}: {str(e)}")
            
            # If training fails, try the opposite task type
            fallback_task_type = 'regression' if task_type == 'classification' else 'classification'
            debug_manager.log_debug(f"Trying fallback task type for training: {fallback_task_type}")
            
            try:
                # Recreate model with fallback task type
                task_type = fallback_task_type
                model = create_model(model_type, parameters, task_type)
                
                # Try training again
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                debug_manager.log_debug(f"Model training successful with fallback {task_type}")
                
                # Update session state
                if 'model_config' in st.session_state:
                    st.session_state.model_config['task_type'] = task_type
                    
            except Exception as e2:
                debug_manager.log_debug(f"Model training failed with fallback {task_type}: {str(e2)}")
                raise Exception(f"Model training failed with both task types. Original error: {str(e)}, Fallback error: {str(e2)}")
        
        # Step 13: Model evaluation
        status_text.text("Evaluating model...")
        progress_bar.progress(90)
        
        debug_manager.log_debug(f"Starting model evaluation with {task_type}")
        debug_manager.log_debug(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        debug_manager.log_debug(f"y_test unique values: {np.unique(y_test)}")
        
        y_pred = model.predict(X_test)
        debug_manager.log_debug(f"Prediction completed. y_pred shape: {y_pred.shape}")
        debug_manager.log_debug(f"y_pred unique values: {np.unique(y_pred)}")
        debug_manager.log_debug(f"y_pred sample: {y_pred[:10]}")
        
        # Validate task type with prediction analysis
        validated_task_type, updated_confidence = validate_task_type_with_data(
            y_test, y_pred, task_type, confidence
        )
        
        if validated_task_type != task_type:
            debug_manager.log_debug(f"Task type validation corrected from {task_type} to {validated_task_type}")
            task_type = validated_task_type
            
            # Update session state
            if 'model_config' in st.session_state:
                st.session_state.model_config['task_type'] = validated_task_type
        
        # Check if all predictions are the same
        if len(np.unique(y_pred)) == 1:
            debug_manager.log_debug(f"WARNING: All predictions are the same value: {y_pred[0]}")
            debug_manager.log_debug(f"This will result in 0.000 metrics for classification")
        
        metrics = calculate_metrics(y_test, y_pred, task_type)
        
        # Step 14: Feature importance
        status_text.text("Calculating feature importance...")
        progress_bar.progress(95)
        
        feature_importance = get_feature_importance(model, feature_names, feature_selector)
        
        # Step 15: Post-training validation
        status_text.text("Final validation...")
        progress_bar.progress(97)
        
        # Check if any features still have extremely high importance
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0.9:
                debug_manager.log_debug(f"CRITICAL: Post-training validation found feature with {max_importance:.3f} importance!")
                # Find the problematic feature
                for feature_name, importance in feature_importance.items():
                    if importance > 0.9:
                        debug_manager.log_debug(f"CRITICAL FEATURE: {feature_name} = {importance:.3f}")
                        # Add to leakage detection
                        if not hasattr(st.session_state, 'leakage_detected'):
                            st.session_state.leakage_detected = {}
                        st.session_state.leakage_detected[feature_name] = {
                            'type': 'post_training_extreme_importance',
                            'importance': importance,
                            'severity': 'critical'
                        }
        
        # Training completed
        status_text.text("Training completed!")
        progress_bar.progress(100)
        
        # Store results
        training_results = {
            'model': model,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'training_time': training_time,
            'model_type': model_type,
            'task_type': metrics.get('actual_task_type', task_type),  # Use actual task type from metrics
            'parameters': parameters,
            'data_leakage_prevention': advanced_settings.get('prevent_data_leakage', True),
            'leakage_detected': leakage_detected,
            'features_filtered': filtered_features,
            'feature_names': feature_names,
            'binning_applied': binning_applied,
            'binning_info': binning_info,
            'timestamp': datetime.now().isoformat()
        }
        
        debug_manager.log_debug(f"Storing training results with task_type: {metrics.get('actual_task_type', task_type)}")
        debug_manager.log_debug(f"Original task_type: {task_type}")
        debug_manager.log_debug(f"Actual task_type from metrics: {metrics.get('actual_task_type', 'Not available')}")
        debug_manager.log_debug(f"Metrics keys: {list(metrics.keys())}")
        debug_manager.log_debug(f"Metrics values: {metrics}")
        
        st.session_state.training_results = training_results
        
        with results_container:
            display_training_summary(training_results)
        
        return training_results
        
    except Exception as e:
        status_text.text("❌ Training failed!")
        display_error_message(f"Training failed: {str(e)}")
        debug_manager.log_debug(f"Model training error: {str(e)}")
        return None