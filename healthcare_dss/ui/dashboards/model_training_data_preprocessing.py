"""
Model Training Data Preprocessing Module
======================================

This module handles data preprocessing for model training:
- Data preparation and cleaning
- Data leakage detection and prevention
- Feature preprocessing pipelines
- Categorical encoding and missing value handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import IsolationForest
from scipy import stats

from healthcare_dss.utils.debug_manager import debug_manager


def prepare_data(df: pd.DataFrame, target_column: str, advanced_settings: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for training"""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
    
    # Convert to numpy arrays
    X = X.values
    y = y.values
    
    # Get feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    return X, y, feature_names


def detect_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, feature_names: List[str], target_column: str = None) -> Dict[str, Any]:
    """Detect potential data leakage patterns with enhanced detection"""
    
    leakage_detected = {}
    
    try:
        # Check 1: Perfect correlation between features and target
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Skip if this is the target feature
                if target_column and feature_name == target_column:
                    continue
                    
                # Calculate correlation with target
                train_corr = np.corrcoef(X_train.iloc[:, i], y_train)[0, 1]
                test_corr = np.corrcoef(X_test.iloc[:, i], y_test)[0, 1]
                
                if abs(train_corr) > 0.99 or abs(test_corr) > 0.99:
                    leakage_detected[feature_name] = {
                        'type': 'perfect_correlation',
                        'train_corr': train_corr,
                        'test_corr': test_corr,
                        'severity': 'critical'
                    }
        
        # Check 2: Features that are identical to target (with different names)
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Skip if this is the target feature
                if target_column and feature_name == target_column:
                    continue
                    
                # Check if feature values match target values exactly
                train_identical = np.array_equal(X_train.iloc[:, i], y_train)
                test_identical = np.array_equal(X_test.iloc[:, i], y_test)
                
                if train_identical or test_identical:
                    leakage_detected[feature_name] = {
                        'type': 'identical_to_target',
                        'train_identical': train_identical,
                        'test_identical': test_identical,
                        'severity': 'critical'
                    }
        
        # Check 3: Features with very low variance (might be constants)
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                train_var = np.var(X_train.iloc[:, i])
                test_var = np.var(X_test.iloc[:, i])
                
                if train_var < 1e-10 or test_var < 1e-10:
                    leakage_detected[feature_name] = {
                        'type': 'low_variance',
                        'train_var': train_var,
                        'test_var': test_var,
                        'severity': 'medium'
                    }
        
        # Check 4: Features that are perfect predictors
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Check if feature perfectly separates classes (for classification)
                if len(np.unique(y_train)) > 1:  # Classification task
                    unique_values = np.unique(X_train.iloc[:, i])
                    perfect_separation = True
                    
                    for val in unique_values:
                        mask = X_train.iloc[:, i] == val
                        if len(np.unique(y_train[mask])) > 1:
                            perfect_separation = False
                            break
                    
                    if perfect_separation:
                        leakage_detected[feature_name] = {
                            'type': 'perfect_separation',
                            'severity': 'critical'
                        }
        
        # Check 5: Features that might be derived from target (common data leakage)
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Check if feature is a simple transformation of target
                # This is a heuristic check for common leakage patterns
                
                # Check if feature is target + constant
                train_diff = X_train.iloc[:, i] - y_train
                test_diff = X_test.iloc[:, i] - y_test
                
                if np.std(train_diff) < 1e-10 and np.std(test_diff) < 1e-10:
                    leakage_detected[feature_name] = {
                        'type': 'target_plus_constant',
                        'severity': 'critical'
                    }
                
                # Check if feature is target * constant
                try:
                    train_ratio = X_train.iloc[:, i] / (y_train + 1e-10)  # Avoid division by zero
                    test_ratio = X_test.iloc[:, i] / (y_test + 1e-10)
                    
                    if np.std(train_ratio) < 1e-10 and np.std(test_ratio) < 1e-10:
                        leakage_detected[feature_name] = {
                            'type': 'target_times_constant',
                            'severity': 'critical'
                        }
                except:
                    pass
                
                # Check if feature contains target as substring (for categorical features)
                if isinstance(feature_name, str) and 'target' in feature_name.lower():
                    leakage_detected[feature_name] = {
                        'type': 'suspicious_naming',
                        'severity': 'medium'
                    }
        
        # Check 6: Enhanced detection for common leakage patterns
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Skip if this is the target feature
                if target_column and feature_name == target_column:
                    continue
                    
                # Check for features that are highly predictive (R² > 0.95)
                try:
                    train_r2 = r2_score(y_train, X_train.iloc[:, i])
                    test_r2 = r2_score(y_test, X_test.iloc[:, i])
                    
                    if train_r2 > 0.95 or test_r2 > 0.95:
                        leakage_detected[feature_name] = {
                            'type': 'extremely_predictive',
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'severity': 'critical'
                        }
                except:
                    pass
                
                # Check for features with suspicious naming patterns
                suspicious_patterns = [
                    'progression', 'outcome', 'result', 'target', 'label', 
                    'prediction', 'forecast', 'estimate', 'score', 'rating',
                    'status', 'class', 'category', 'group', 'type'
                ]
                
                if isinstance(feature_name, str):
                    feature_lower = feature_name.lower()
                    for pattern in suspicious_patterns:
                        if pattern in feature_lower:
                            leakage_detected[feature_name] = {
                                'type': 'suspicious_naming_pattern',
                                'pattern': pattern,
                                'severity': 'high'
                            }
                            break
        
        # Check 7: Check for features that are perfect inverses of target
        for i, feature_name in enumerate(feature_names):
            if i < X_train.shape[1]:
                # Skip if this is the target feature
                if target_column and feature_name == target_column:
                    continue
                    
                try:
                    # Check if feature is 1 - target or similar inverse
                    train_inverse_diff = X_train.iloc[:, i] - (1 - y_train)
                    test_inverse_diff = X_test.iloc[:, i] - (1 - y_test)
                    
                    if np.std(train_inverse_diff) < 1e-10 and np.std(test_inverse_diff) < 1e-10:
                        leakage_detected[feature_name] = {
                            'type': 'target_inverse',
                            'severity': 'critical'
                        }
                except:
                    pass
        
        return leakage_detected
        
    except Exception as e:
        debug_manager.log_debug(f"Error in data leakage detection: {str(e)}")
        return {}


def estimate_feature_importance_pre_training(X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str]) -> Dict[str, float]:
    """Estimate feature importance before training to detect potential leakage"""
    
    try:
        # Determine if classification or regression
        is_classification = len(np.unique(y_train)) <= 20 and len(np.unique(y_train)) > 1
        
        # Use a simple model to estimate importance
        if is_classification:
            estimator = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        else:
            estimator = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
        
        # Fit the estimator
        estimator.fit(X_train, y_train)
        
        # Get feature importances
        importances = estimator.feature_importances_
        
        # Create importance dictionary
        importance_dict = {}
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(importance)
        
        return importance_dict
        
    except Exception as e:
        debug_manager.log_debug(f"Error estimating pre-training feature importance: {str(e)}")
        return {}


def detect_leakage_by_importance(X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    """Detect data leakage by analyzing feature importance patterns"""
    
    leakage_detected = {}
    
    try:
        # Estimate feature importance
        importance_dict = estimate_feature_importance_pre_training(X_train, y_train, feature_names)
        
        if not importance_dict:
            return leakage_detected
        
        # Check for extremely high importance
        max_importance = max(importance_dict.values()) if importance_dict else 0
        
        if max_importance > 0.8:  # Lower threshold for pre-training detection
            # Find features with suspiciously high importance
            for feature_name, importance in importance_dict.items():
                if importance > 0.8:
                    leakage_detected[feature_name] = {
                        'type': 'extremely_high_importance',
                        'importance': importance,
                        'severity': 'critical'
                    }
                    debug_manager.log_debug(f"Pre-training leakage detected: {feature_name} has importance {importance:.3f}")
        
        # Check for features that are too predictive
        for feature_name, importance in importance_dict.items():
            if importance > 0.6:  # High but not extreme threshold
                # Additional validation: check if this feature alone can predict target very well
                try:
                    # Use simple linear model to test predictive power
                    if len(np.unique(y_train)) <= 20:  # Classification
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        # Reshape for single feature
                        X_single = X_train.iloc[:, feature_names.index(feature_name)].values.reshape(-1, 1)
                        model.fit(X_single, y_train)
                        y_pred = model.predict(X_single)
                        score = accuracy_score(y_train, y_pred)
                    else:  # Regression
                        model = LinearRegression()
                        X_single = X_train.iloc[:, feature_names.index(feature_name)].values.reshape(-1, 1)
                        model.fit(X_single, y_train)
                        y_pred = model.predict(X_single)
                        score = r2_score(y_train, y_pred)
                    
                    # If single feature can predict target very well, it's suspicious
                    if score > 0.95:
                        leakage_detected[feature_name] = {
                            'type': 'single_feature_overprediction',
                            'importance': importance,
                            'single_feature_score': score,
                            'severity': 'critical'
                        }
                        debug_manager.log_debug(f"Single feature overprediction: {feature_name} can predict target with {score:.3f} accuracy/R²")
                        
                except Exception as e:
                    debug_manager.log_debug(f"Error in single feature validation for {feature_name}: {str(e)}")
        
        return leakage_detected
        
    except Exception as e:
        debug_manager.log_debug(f"Error in importance-based leakage detection: {str(e)}")
        return {}


def filter_suspicious_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, feature_names: List[str], advanced_settings: Dict, target_column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Filter out suspicious features that might cause data leakage"""
    
    if not advanced_settings.get('prevent_data_leakage', True):
        return X_train, X_test, feature_names, []
    
    # Combine multiple detection methods
    leakage_detected = {}
    
    # Method 1: Traditional correlation/pattern detection
    traditional_leakage = detect_data_leakage(X_train, X_test, y_train, y_test, feature_names, target_column)
    leakage_detected.update(traditional_leakage)
    
    # Method 2: Importance-based detection (NEW)
    importance_leakage = detect_leakage_by_importance(X_train, y_train, feature_names)
    leakage_detected.update(importance_leakage)
    
    if not leakage_detected:
        return X_train, X_test, feature_names, []
    
    # Filter out critical and high severity features
    features_to_remove = []
    for feature_name, details in leakage_detected.items():
        if details['severity'] in ['critical', 'high']:
            features_to_remove.append(feature_name)
            debug_manager.log_debug(f"Removing suspicious feature: {feature_name} - {details['type']} (severity: {details['severity']})")
    
    if features_to_remove:
        # Find indices of features to remove
        indices_to_remove = []
        for feature_name in features_to_remove:
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                indices_to_remove.append(idx)
        
        # Remove features from dataframes
        if indices_to_remove:
            # Sort indices in descending order to avoid index shifting issues
            indices_to_remove.sort(reverse=True)
            
            # Remove from feature names
            for idx in indices_to_remove:
                feature_names.pop(idx)
            
            # Remove from dataframes
            X_train_filtered = X_train.drop(X_train.columns[indices_to_remove], axis=1)
            X_test_filtered = X_test.drop(X_test.columns[indices_to_remove], axis=1)
            
            debug_manager.log_debug(f"Filtered out {len(features_to_remove)} suspicious features: {features_to_remove}")
            
            return X_train_filtered, X_test_filtered, feature_names, features_to_remove
    
    return X_train, X_test, feature_names, []


def apply_preprocessing_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, advanced_settings: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply preprocessing pipeline with data leakage prevention"""
    
    debug_manager.log_debug(f"Starting preprocessing pipeline")
    debug_manager.log_debug(f"X_train shape before preprocessing: {X_train.shape}")
    debug_manager.log_debug(f"X_test shape before preprocessing: {X_test.shape}")
    debug_manager.log_debug(f"X_train columns: {list(X_train.columns)}")
    
    # Handle categorical variables
    categorical_strategy = advanced_settings.get('categorical_strategy', 'one_hot')
    
    if categorical_strategy == 'one_hot':
        # One-hot encoding - fit on training data only using sklearn
        # Identify categorical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols:
            # Initialize OneHotEncoder
            ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            
            # Fit on training data only
            X_train_cat_encoded = ohe.fit_transform(X_train[categorical_cols])
            X_test_cat_encoded = ohe.transform(X_test[categorical_cols])
            
            # Get feature names
            cat_feature_names = ohe.get_feature_names_out(categorical_cols)
            
            # Combine with numeric features
            if numeric_cols:
                X_train_encoded = np.column_stack([
                    X_train[numeric_cols].values,
                    X_train_cat_encoded
                ])
                X_test_encoded = np.column_stack([
                    X_test[numeric_cols].values,
                    X_test_cat_encoded
                ])
                feature_names = list(numeric_cols) + list(cat_feature_names)
            else:
                X_train_encoded = X_train_cat_encoded
                X_test_encoded = X_test_cat_encoded
                feature_names = list(cat_feature_names)
        else:
            # No categorical columns, just numeric
            X_train_encoded = X_train.values
            X_test_encoded = X_test.values
            feature_names = list(X_train.columns)
        
        # Convert to DataFrame for easier handling
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)
        
    elif categorical_strategy == 'label':
        # Label encoding - fit on training data only
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        label_encoders = {}
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            
            # Handle unseen categories in test set
            test_values = X_test[col].astype(str)
            unseen_mask = ~test_values.isin(le.classes_)
            if unseen_mask.any():
                # Replace unseen categories with most frequent class
                most_frequent = le.classes_[0]
                test_values[unseen_mask] = most_frequent
            
            X_test_encoded[col] = le.transform(test_values)
            label_encoders[col] = le
    
    else:  # target encoding or other strategies
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
    
    # Handle missing values using sklearn for consistency
    missing_value_strategy = advanced_settings.get('missing_value_strategy', 'mean')
    
    if missing_value_strategy != 'drop':
        # Determine strategy for SimpleImputer
        if missing_value_strategy == 'mean':
            strategy = 'mean'
        elif missing_value_strategy == 'median':
            strategy = 'median'
        elif missing_value_strategy == 'mode':
            strategy = 'most_frequent'
        else:
            strategy = 'mean'  # fallback
        
        # Initialize imputer
        imputer = SimpleImputer(strategy=strategy)
        
        # Fit on training data only
        X_train_encoded = pd.DataFrame(
            imputer.fit_transform(X_train_encoded),
            columns=X_train_encoded.columns,
            index=X_train_encoded.index
        )
        
        # Transform test data using training statistics
        X_test_encoded = pd.DataFrame(
            imputer.transform(X_test_encoded),
            columns=X_test_encoded.columns,
            index=X_test_encoded.index
        )
    
    elif missing_value_strategy == 'drop':
        # Drop rows with missing values
        X_train_encoded = X_train_encoded.dropna()
        X_test_encoded = X_test_encoded.dropna()
    
    # Handle outliers if enabled
    if advanced_settings.get('handle_outliers', False):
        outlier_method = advanced_settings.get('outlier_method', 'iqr')
        
        if outlier_method == 'iqr':
            # IQR-based outlier detection on training data only
            numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = X_train_encoded[col].quantile(0.25)
                Q3 = X_train_encoded[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers in both training and test sets
                X_train_encoded[col] = X_train_encoded[col].clip(lower_bound, upper_bound)
                X_test_encoded[col] = X_test_encoded[col].clip(lower_bound, upper_bound)
        
        elif outlier_method == 'zscore':
            # Z-score based outlier detection on training data only
            numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Calculate mean and std on training data only
                train_mean = X_train_encoded[col].mean()
                train_std = X_train_encoded[col].std()
                
                # Cap outliers at 3 standard deviations
                lower_bound = train_mean - 3 * train_std
                upper_bound = train_mean + 3 * train_std
                
                X_train_encoded[col] = X_train_encoded[col].clip(lower_bound, upper_bound)
                X_test_encoded[col] = X_test_encoded[col].clip(lower_bound, upper_bound)
        
        elif outlier_method == 'isolation_forest':
            # Isolation Forest outlier detection (more advanced)
            numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Fit isolation forest on training data only
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_mask_train = iso_forest.fit_predict(X_train_encoded[numeric_cols]) == -1
                outlier_mask_test = iso_forest.predict(X_test_encoded[numeric_cols]) == -1
                
                # Cap outliers using IQR method as fallback
                for col in numeric_cols:
                    Q1 = X_train_encoded[col].quantile(0.25)
                    Q3 = X_train_encoded[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    X_train_encoded[col] = X_train_encoded[col].clip(lower_bound, upper_bound)
                    X_test_encoded[col] = X_test_encoded[col].clip(lower_bound, upper_bound)
    
    # Convert to numpy arrays
    X_train_array = X_train_encoded.values
    X_test_array = X_test_encoded.values
    
    # Get feature names
    feature_names = list(X_train_encoded.columns)
    
    debug_manager.log_debug(f"Preprocessing completed")
    debug_manager.log_debug(f"X_train_array shape after preprocessing: {X_train_array.shape}")
    debug_manager.log_debug(f"X_test_array shape after preprocessing: {X_test_array.shape}")
    debug_manager.log_debug(f"Feature names count: {len(feature_names)}")
    debug_manager.log_debug(f"Feature names: {feature_names[:10]}...")  # Show first 10
    
    # Check if we have any features left
    if X_train_array.shape[1] == 0:
        debug_manager.log_debug(f"ERROR: No features left after preprocessing!")
        raise ValueError("No features remaining after preprocessing. Check data leakage prevention settings.")
    
    return X_train_array, X_test_array, feature_names


def apply_legacy_preprocessing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Apply legacy preprocessing (with data leakage) for backward compatibility"""
    
    # Combine train and test for preprocessing (DATA LEAKAGE!)
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    
    # Handle categorical variables
    X_combined = pd.get_dummies(X_combined, drop_first=True)
    
    # Handle missing values
    X_combined = X_combined.fillna(X_combined.mean())
    
    # Split back
    train_size = len(X_train)
    X_train_processed = X_combined.iloc[:train_size]
    X_test_processed = X_combined.iloc[train_size:]
    
    # Convert to numpy arrays
    X_train_array = X_train_processed.values
    X_test_array = X_test_processed.values
    
    # Get feature names
    feature_names = list(X_train_processed.columns)
    
    return X_train_array, X_test_array, feature_names
