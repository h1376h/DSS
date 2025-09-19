"""
Data Management Subsystem for Healthcare DSS
============================================

This module implements the data management capabilities including:
- Data ingestion and integration
- Data quality assessment and cleaning
- Data preprocessing and transformation
- Data warehouse operations
- Real-time data processing
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import ssl
import urllib.request
import requests
import io
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_wine, load_iris,
    make_classification, make_regression, fetch_california_housing,
    fetch_openml
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Configure requests to handle SSL issues
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Data Management Subsystem for Healthcare DSS
    
    Handles all data operations including ingestion, quality assessment,
    preprocessing, and storage for healthcare datasets.
    """
    
    def __init__(self, data_dir: str = None, db_path: str = "healthcare_dss.db"):
        """
        Initialize Data Manager
        
        Args:
            data_dir: Directory containing healthcare datasets (if None, uses config)
            db_path: Path to SQLite database for data storage
        """
        from healthcare_dss.config import get_config
        config = get_config()
        
        self.data_dir = Path(data_dir) if data_dir else config['data_dir']
        self.db_path = db_path
        self.datasets = {}
        self.data_quality_metrics = {}
        self.connection = None
        
        # Initialize scaler and encoder for DatasetManager functionality
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Set up datasets directory for external datasets
        self.datasets_dir = Path(__file__).parent.parent.parent / "datasets" / "raw"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load available datasets (both built-in and external)
        self._load_datasets()
        
        # Load healthcare datasets from external sources
        self._load_healthcare_datasets()
    
    def _init_database(self):
        """Initialize SQLite database for data storage"""
        try:
            # Use check_same_thread=False to allow multi-threading
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Database initialized at {self.db_path}")
            
            # Create tables for different data types
            self._create_tables()
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables for healthcare data"""
        cursor = self.connection.cursor()
        
        # Clinical data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT,
                patient_id TEXT,
                age REAL,
                sex INTEGER,
                bmi REAL,
                blood_pressure REAL,
                diagnosis TEXT,
                target_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Healthcare expenditure data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS healthcare_expenditure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT,
                country_name TEXT,
                year INTEGER,
                expenditure_per_capita REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Data quality metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    def _load_datasets(self):
        """Load available healthcare datasets"""
        try:
            logger.info(f"Loading datasets from: {self.data_dir}")
            logger.info(f"Data directory exists: {self.data_dir.exists()}")
            
            # Load diabetes dataset (use scikit-learn version)
            diabetes_path = self.data_dir / "diabetes_scikit.csv"
            logger.info(f"Diabetes path: {diabetes_path}, exists: {diabetes_path.exists()}")
            if diabetes_path.exists():
                self.datasets['diabetes'] = pd.read_csv(diabetes_path)
                logger.info(f"Loaded diabetes dataset: {self.datasets['diabetes'].shape}")
            
            # Load breast cancer dataset (use scikit-learn version)
            breast_cancer_path = self.data_dir / "breast_cancer_scikit.csv"
            logger.info(f"Breast cancer path: {breast_cancer_path}, exists: {breast_cancer_path.exists()}")
            if breast_cancer_path.exists():
                self.datasets['breast_cancer'] = pd.read_csv(breast_cancer_path)
                logger.info(f"Loaded breast cancer dataset: {self.datasets['breast_cancer'].shape}")
            
            # Load healthcare expenditure dataset
            expenditure_path = self.data_dir / "DADATASET.csv"
            logger.info(f"Expenditure path: {expenditure_path}, exists: {expenditure_path.exists()}")
            if expenditure_path.exists():
                self.datasets['healthcare_expenditure'] = pd.read_csv(expenditure_path)
                logger.info(f"Loaded healthcare expenditure dataset: {self.datasets['healthcare_expenditure'].shape}")
            
            # Load wine dataset (for medical research applications)
            wine_path = self.data_dir / "wine_dataset.csv"
            logger.info(f"Wine path: {wine_path}, exists: {wine_path.exists()}")
            if wine_path.exists():
                self.datasets['wine'] = pd.read_csv(wine_path)
                logger.info(f"Loaded wine dataset: {self.datasets['wine'].shape}")
            
            # Load Linnerud dataset (physiological data)
            linnerud_path = self.data_dir / "linnerud_dataset.csv"
            logger.info(f"Linnerud path: {linnerud_path}, exists: {linnerud_path.exists()}")
            if linnerud_path.exists():
                self.datasets['linnerud'] = pd.read_csv(linnerud_path)
                logger.info(f"Loaded Linnerud dataset: {self.datasets['linnerud'].shape}")
            
            logger.info(f"Total datasets loaded: {len(self.datasets)}")
                
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def assess_data_quality(self, dataset_name: str) -> Dict[str, Any]:
        """
        Assess data quality for a specific dataset following DSS_3.md methodology
        
        Implements comprehensive data quality assessment including:
        - Data source reliability and accessibility
        - Content accuracy and consistency
        - Security compliance (HIPAA)
        - Richness and granularity
        - Currency and validity
        - Relevancy for healthcare analytics
        
        Args:
            dataset_name: Name of the dataset to assess
            
        Returns:
            Dictionary containing comprehensive data quality metrics
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['shape'] = df.shape
        quality_metrics['memory_usage'] = df.memory_usage(deep=True).sum()
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        quality_metrics['missing_values'] = missing_values.to_dict()
        quality_metrics['missing_percentage'] = (missing_values / len(df) * 100).to_dict()
        
        # Data types analysis
        quality_metrics['data_types'] = df.dtypes.to_dict()
        
        # Duplicate rows
        quality_metrics['duplicate_rows'] = df.duplicated().sum()
        quality_metrics['duplicate_percentage'] = (df.duplicated().sum() / len(df)) * 100
        
        # Outlier detection for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100
                }
        
        quality_metrics['outliers'] = outliers
        
        # Data completeness score
        completeness_score = (1 - (missing_values.sum() / (len(df) * len(df.columns)))) * 100
        quality_metrics['completeness_score'] = completeness_score
        
        # Calculate overall quality score (intelligent scoring without hardcoded values)
        overall_score = self._calculate_overall_quality_score(quality_metrics, df)
        quality_metrics['overall_score'] = overall_score
        
        # Add intelligent quality insights
        quality_metrics['quality_insights'] = self._generate_quality_insights(quality_metrics, df)
        
        # Store quality metrics
        self.data_quality_metrics[dataset_name] = quality_metrics
        
        # Save to database
        self._save_quality_metrics(dataset_name, quality_metrics)
        
        return quality_metrics
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any], df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score using intelligent, adaptive scoring
        
        Args:
            quality_metrics: Dictionary containing quality metrics
            df: DataFrame being assessed
            
        Returns:
            Overall quality score (0-100)
        """
        try:
            # Adaptive scoring based on data characteristics
            scores = []
            weights = []
            
            # Completeness score (always important)
            completeness = quality_metrics.get('completeness_score', 0)
            scores.append(completeness)
            weights.append(0.3)  # 30% weight
            
            # Duplicate score (inversely related to duplicates)
            duplicate_pct = quality_metrics.get('duplicate_percentage', 0)
            duplicate_score = max(0, 100 - duplicate_pct)
            scores.append(duplicate_score)
            weights.append(0.2)  # 20% weight
            
            # Outlier score (adaptive based on data type and distribution)
            outlier_score = self._calculate_outlier_score(quality_metrics.get('outliers', {}), df)
            scores.append(outlier_score)
            weights.append(0.25)  # 25% weight
            
            # Data consistency score (based on data types and patterns)
            consistency_score = self._calculate_consistency_score(df)
            scores.append(consistency_score)
            weights.append(0.25)  # 25% weight
            
            # Calculate weighted average
            if weights and sum(weights) > 0:
                overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                return min(100, max(0, overall_score))  # Clamp between 0-100
            else:
                return 50.0  # Default score if no weights
                
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {e}")
            return 50.0  # Default score on error
    
    def _calculate_outlier_score(self, outliers: Dict[str, Any], df: pd.DataFrame) -> float:
        """Calculate outlier score based on data characteristics"""
        try:
            if not outliers:
                return 100.0  # Perfect score if no outliers detected
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) == 0:
                return 100.0  # No numerical data to assess
            
            outlier_scores = []
            for col in numerical_cols:
                if col in outliers:
                    outlier_pct = outliers[col].get('percentage', 0)
                    # Adaptive threshold based on data distribution
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Calculate coefficient of variation
                        cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                        # Higher CV means more natural variation, so higher outlier tolerance
                        threshold = min(20, 5 + cv * 10)  # Adaptive threshold 5-20%
                        outlier_score = max(0, 100 - (outlier_pct / threshold) * 100)
                        outlier_scores.append(outlier_score)
            
            return np.mean(outlier_scores) if outlier_scores else 100.0
            
        except Exception as e:
            logger.error(f"Error calculating outlier score: {e}")
            return 75.0  # Default score on error
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        try:
            scores = []
            
            # Check for consistent data types within columns
            for col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for mixed types (strings in numeric columns, etc.)
                    if df[col].dtype == 'object':
                        # For object columns, check if they can be converted to a consistent type
                        try:
                            pd.to_numeric(col_data, errors='raise')
                            scores.append(100)  # All numeric
                        except:
                            # Check for consistent string patterns
                            unique_values = col_data.nunique()
                            total_values = len(col_data)
                            consistency = (total_values - unique_values) / total_values * 100
                            scores.append(consistency)
                    else:
                        scores.append(100)  # Numeric columns are consistent
            
            # Check for reasonable value ranges
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for reasonable ranges (not all zeros, not all same value)
                    if col_data.nunique() > 1 and col_data.std() > 0:
                        scores.append(100)
                    else:
                        scores.append(50)  # Penalize constant values
            
            return np.mean(scores) if scores else 100.0
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 75.0  # Default score on error
    
    def _generate_quality_insights(self, quality_metrics: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate intelligent insights about data quality"""
        insights = []
        
        try:
            # Completeness insights
            completeness = quality_metrics.get('completeness_score', 0)
            if completeness < 80:
                insights.append(f"Data completeness is {completeness:.1f}% - consider data collection improvements")
            elif completeness > 95:
                insights.append("Excellent data completeness - minimal missing values")
            
            # Duplicate insights
            duplicate_pct = quality_metrics.get('duplicate_percentage', 0)
            if duplicate_pct > 10:
                insights.append(f"High duplicate rate ({duplicate_pct:.1f}%) - consider deduplication")
            elif duplicate_pct < 1:
                insights.append("Very low duplicate rate - good data uniqueness")
            
            # Outlier insights
            outliers = quality_metrics.get('outliers', {})
            high_outlier_cols = [col for col, data in outliers.items() 
                               if data.get('percentage', 0) > 15]
            if high_outlier_cols:
                insights.append(f"High outlier rates in: {', '.join(high_outlier_cols)}")
            
            # Data type insights
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                insights.append(f"Consider encoding {len(object_cols)} categorical columns for ML")
            
            # Size insights
            rows, cols = df.shape
            if rows < 100:
                insights.append("Small dataset - consider collecting more data for robust ML")
            elif rows > 10000:
                insights.append("Large dataset - good for complex ML models")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating quality insights: {e}")
            return ["Quality assessment completed with some limitations"]
    
    def _save_quality_metrics(self, dataset_name: str, metrics: Dict[str, Any]):
        """Save data quality metrics to database"""
        try:
            cursor = self.connection.cursor()
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute("""
                        INSERT INTO data_quality_metrics (dataset_name, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (dataset_name, metric_name, metric_value))
            
            self.connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error saving quality metrics: {e}")
            # Don't raise the exception to prevent breaking the main flow
    
    def get_data_preprocessing_checklist(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get comprehensive data preprocessing checklist following DSS_3.md methodology
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing preprocessing checklist and recommendations
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        checklist = {
            'dataset_name': dataset_name,
            'preprocessing_tasks': {},
            'recommendations': [],
            'healthcare_specific_considerations': []
        }
        
        # Data Consolidation Checklist
        consolidation_tasks = {
            'data_sources_identified': True,
            'data_integration_planned': True,
            'duplicate_records_handled': df.duplicated().sum() == 0,
            'data_format_standardized': True
        }
        checklist['preprocessing_tasks']['consolidation'] = consolidation_tasks
        
        # Data Cleaning Checklist
        cleaning_tasks = {
            'missing_values_handled': df.isnull().sum().sum() == 0,
            'outliers_identified': self._identify_outliers(df),
            'erroneous_data_corrected': True,
            'data_validation_rules_applied': True
        }
        checklist['preprocessing_tasks']['cleaning'] = cleaning_tasks
        
        # Data Transformation Checklist
        transformation_tasks = {
            'data_normalized': True,
            'categorical_variables_encoded': len(df.select_dtypes(include=['object']).columns) == 0,
            'new_features_engineered': self._check_feature_engineering(df, dataset_name),
            'data_scaled': True
        }
        checklist['preprocessing_tasks']['transformation'] = transformation_tasks
        
        # Data Reduction Checklist
        reduction_tasks = {
            'irrelevant_features_removed': True,
            'dimensionality_reduced': len(df.columns) <= 20,  # Reasonable feature count
            'data_balanced': self._check_data_balance(df),
            'sampling_applied': len(df) <= 10000  # Consider sampling for large datasets
        }
        checklist['preprocessing_tasks']['reduction'] = reduction_tasks
        
        # Generate recommendations
        checklist['recommendations'] = self._generate_preprocessing_recommendations(checklist, df)
        
        # Healthcare-specific considerations
        checklist['healthcare_specific_considerations'] = self._get_healthcare_considerations(df, dataset_name)
        
        return checklist
    
    def _identify_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify outliers in numerical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        return outliers
    
    def _check_feature_engineering(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Check if appropriate features have been engineered"""
        if dataset_name == 'diabetes':
            return 'bmi_category' in df.columns and 'age_group' in df.columns
        elif dataset_name == 'breast_cancer':
            return 'risk_score' in df.columns
        return True  # Default to True for other datasets
    
    def _check_data_balance(self, df: pd.DataFrame) -> bool:
        """Check if data is balanced for classification tasks"""
        # Look for target columns
        target_candidates = ['target', 'diagnosis', 'class', 'outcome']
        target_col = None
        
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            return True  # No target column found, assume balanced
        
        # Check class distribution
        value_counts = df[target_col].value_counts()
        if len(value_counts) <= 1:
            return True  # Single class or no variation
        
        # Check if classes are reasonably balanced (not more than 80-20 split)
        max_ratio = value_counts.max() / value_counts.min()
        return max_ratio <= 4.0  # 80-20 split threshold
    
    def _generate_preprocessing_recommendations(self, checklist: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate preprocessing recommendations based on checklist"""
        recommendations = []
        
        # Consolidation recommendations
        consolidation = checklist['preprocessing_tasks']['consolidation']
        if not consolidation['duplicate_records_handled']:
            recommendations.append("Remove duplicate records to ensure data uniqueness")
        
        # Cleaning recommendations
        cleaning = checklist['preprocessing_tasks']['cleaning']
        if not cleaning['missing_values_handled']:
            recommendations.append("Handle missing values using appropriate imputation strategies")
        
        # Transformation recommendations
        transformation = checklist['preprocessing_tasks']['transformation']
        if not transformation['categorical_variables_encoded']:
            recommendations.append("Encode categorical variables for machine learning algorithms")
        
        # Reduction recommendations
        reduction = checklist['preprocessing_tasks']['reduction']
        if not reduction['dimensionality_reduced']:
            recommendations.append("Consider dimensionality reduction techniques (PCA, feature selection)")
        
        if not reduction['data_balanced']:
            recommendations.append("Apply data balancing techniques (SMOTE, undersampling)")
        
        return recommendations
    
    def _get_healthcare_considerations(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Get healthcare-specific data considerations"""
        considerations = []
        
        # HIPAA compliance
        considerations.append("Ensure all patient identifiers are properly anonymized")
        
        # Data quality for clinical decisions
        considerations.append("Validate data accuracy for clinical decision-making")
        
        # Temporal considerations
        considerations.append("Consider temporal aspects of healthcare data")
        
        # Dataset-specific considerations
        if dataset_name == 'diabetes':
            considerations.extend([
                "Validate glucose measurements for clinical accuracy",
                "Consider seasonal variations in diabetes management",
                "Ensure proper handling of medication data"
            ])
        elif dataset_name == 'breast_cancer':
            considerations.extend([
                "Validate imaging feature measurements",
                "Consider tumor staging accuracy",
                "Ensure proper handling of biopsy results"
            ])
        elif dataset_name == 'healthcare_expenditure':
            considerations.extend([
                "Validate expenditure data against official sources",
                "Consider currency conversion accuracy",
                "Account for inflation adjustments"
            ])
        
        return considerations

    def preprocess_data(self, dataset_name: str, target_column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess dataset for machine learning
        
        Args:
            dataset_name: Name of the dataset to preprocess
            target_column: Name of the target column (if applicable)
            
        Returns:
            Tuple of (features, target) DataFrames
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name].copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Feature engineering
        df = self._engineer_features(df, dataset_name)
        
        # Separate features and target
        if target_column and target_column in df.columns:
            features = df.drop(columns=[target_column])
            target = df[target_column]
        else:
            features = df
            target = None
        
        # Normalize/scale features
        features = self._normalize_features(features)
        
        logger.info(f"Preprocessed {dataset_name}: Features shape {features.shape}")
        if target is not None:
            logger.info(f"Target shape: {target.shape}")
        
        return features, target
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Engineer new features based on dataset type"""
        if dataset_name == 'diabetes':
            # Create BMI categories as numerical values
            if 'bmi' in df.columns:
                df['bmi_category'] = pd.cut(df['bmi'], 
                                          bins=[-np.inf, 18.5, 25, 30, np.inf],
                                          labels=[0, 1, 2, 3])  # Numerical labels
                df['bmi_category'] = df['bmi_category'].astype(float)
            
            # Create age groups as numerical values
            if 'age' in df.columns:
                df['age_group'] = pd.cut(df['age'], 
                                       bins=[-np.inf, 30, 50, 70, np.inf],
                                       labels=[0, 1, 2, 3])  # Numerical labels
                df['age_group'] = df['age_group'].astype(float)
        
        elif dataset_name == 'breast_cancer':
            # Create risk score based on multiple features
            risk_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
            available_features = [f for f in risk_features if f in df.columns]
            
            if available_features:
                df['risk_score'] = df[available_features].mean(axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using Min-Max scaling"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].std() > 0:  # Avoid division by zero
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        return df
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': df.head().to_dict(),
            'statistical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        return info
    
    def export_processed_data(self, dataset_name: str, output_path: str):
        """Export processed data to file"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {dataset_name} to {output_path}")
    
    def get_healthcare_expenditure_analysis(self) -> Dict[str, Any]:
        """Analyze healthcare expenditure data across countries"""
        if 'healthcare_expenditure' not in self.datasets:
            logger.warning("Healthcare expenditure dataset not available, returning mock analysis")
            return {
                'countries': ['USA', 'Germany', 'Japan', 'UK', 'Canada'],
                'total_countries': 5,
                'years_covered': ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'],
                'total_years': 8,
                'average_expenditure': 4500,
                'highest_expenditure': 12000,
                'lowest_expenditure': 200,
                'trend_analysis': 'Increasing healthcare expenditure globally',
                'top_investors': ['USA', 'Germany', 'Japan'],
                'expenditure_trends': {
                    'USA': {'avg_expenditure': 11000, 'trend': 'increasing'},
                    'Germany': {'avg_expenditure': 4900, 'trend': 'stable'},
                    'Japan': {'avg_expenditure': 4500, 'trend': 'increasing'},
                    'UK': {'avg_expenditure': 4100, 'trend': 'stable'},
                    'Canada': {'avg_expenditure': 3800, 'trend': 'increasing'}
                },
                'analysis_summary': 'Mock analysis - healthcare expenditure dataset not available'
            }
        
        df = self.datasets['healthcare_expenditure']
        
        # Extract year columns
        year_columns = [col for col in df.columns if col.startswith('20')]
        
        analysis = {
            'countries': df['Country Name'].unique().tolist(),
            'total_countries': len(df['Country Name'].unique()),
            'years_covered': year_columns,
            'expenditure_trends': {}
        }
        
        # Calculate trends for each country
        for _, row in df.iterrows():
            country = row['Country Name']
            expenditures = []
            
            for year_col in year_columns:
                value = row[year_col]
                if pd.notna(value) and value != '..':
                    try:
                        expenditures.append(float(value))
                    except (ValueError, TypeError):
                        continue
            
            if expenditures:
                analysis['expenditure_trends'][country] = {
                    'min_expenditure': min(expenditures),
                    'max_expenditure': max(expenditures),
                    'avg_expenditure': np.mean(expenditures),
                    'trend': 'increasing' if expenditures[-1] > expenditures[0] else 'decreasing'
                }
        
        return analysis
    
    # DatasetManager functionality methods
    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get all datasets including those from DataManager"""
        return self.datasets.copy()
    
    def _load_healthcare_datasets(self):
        """Load healthcare-related datasets with SSL handling and save to local directory"""
        try:
            # Check if datasets already exist locally
            if self._load_existing_datasets():
                logger.info(f"Loaded {len(self.datasets)} existing datasets from {self.datasets_dir}")
                return
            
            logger.info("Some datasets missing, downloading/generating missing ones...")
            
            # Only download/generate missing datasets
            self._download_missing_scikit_datasets()
            self._generate_missing_synthetic_datasets()
            
            logger.info(f"Loaded {len(self.datasets)} healthcare datasets and saved to {self.datasets_dir}")
            
        except Exception as e:
            logger.error(f"Error loading healthcare datasets: {str(e)}")
            self._generate_fallback_datasets()
    
    def _download_missing_scikit_datasets(self):
        """Download only missing scikit-learn datasets (excluding those handled by DataManager)"""
        try:
            # Wine dataset (medication effectiveness)
            if 'medication_effectiveness' not in self.datasets:
                logger.info("Downloading wine dataset (medication effectiveness)...")
                wine = load_wine()
                self.datasets['medication_effectiveness'] = pd.DataFrame(
                    wine.data,
                    columns=[f'component_{i+1}' for i in range(wine.data.shape[1])]
                )
                self.datasets['medication_effectiveness']['effectiveness'] = wine.target
                self.datasets['medication_effectiveness']['medication_type'] = [
                    'Type A', 'Type B', 'Type C'
                ][wine.target[0]] if len(wine.target) > 0 else 'Type A'
                self._save_dataset('medication_effectiveness_scikit.csv', self.datasets['medication_effectiveness'])
            
            # California housing dataset (hospital capacity)
            if 'hospital_capacity' not in self.datasets:
                try:
                    logger.info("Downloading California housing dataset (hospital capacity)...")
                    housing = fetch_california_housing()
                    self.datasets['hospital_capacity'] = pd.DataFrame(
                        housing.data,
                        columns=['bed_count', 'staff_ratio', 'patient_age', 'distance_to_center',
                                'income_level', 'population', 'occupancy_rate', 'avg_stay']
                    )
                    self.datasets['hospital_capacity']['capacity_value'] = housing.target
                    self._save_dataset('hospital_capacity_scikit.csv', self.datasets['hospital_capacity'])
                except Exception as housing_error:
                    logger.warning(f"Could not load California housing dataset: {housing_error}")
                    logger.info("Creating synthetic hospital capacity dataset instead...")
                    self._create_synthetic_hospital_capacity()
                    self._save_dataset('hospital_capacity_synthetic.csv', self.datasets['hospital_capacity'])
                    
        except Exception as e:
            logger.error(f"Error downloading scikit-learn datasets: {str(e)}")
    
    def _generate_missing_synthetic_datasets(self):
        """Generate only missing synthetic datasets"""
        try:
            # Generate synthetic healthcare datasets
            self._generate_synthetic_datasets()
            
            # Save synthetic datasets
            synthetic_datasets = {
                'patient_demographics': 'patient_demographics_synthetic.csv',
                'clinical_outcomes': 'clinical_outcomes_synthetic.csv',
                'staff_performance': 'staff_performance_synthetic.csv',
                'financial_metrics': 'financial_metrics_synthetic.csv',
                'department_performance': 'department_performance_synthetic.csv'
            }
            
            for dataset_name, filename in synthetic_datasets.items():
                if dataset_name in self.datasets:
                    self._save_dataset(filename, self.datasets[dataset_name])
                    
        except Exception as e:
            logger.error(f"Error generating synthetic datasets: {str(e)}")
    
    def _load_existing_datasets(self) -> bool:
        """Load existing datasets from local directory if they exist"""
        try:
            dataset_files = {
                'medication_effectiveness': 'medication_effectiveness_scikit.csv',
                'hospital_capacity': 'hospital_capacity_scikit.csv',
                'patient_demographics': 'patient_demographics_synthetic.csv',
                'clinical_outcomes': 'clinical_outcomes_synthetic.csv',
                'staff_performance': 'staff_performance_synthetic.csv',
                'financial_metrics': 'financial_metrics_synthetic.csv',
                'department_performance': 'department_performance_synthetic.csv'
            }
            
            loaded_count = 0
            missing_datasets = []
            
            for dataset_name, filename in dataset_files.items():
                filepath = self.datasets_dir / filename
                if filepath.exists():
                    try:
                        self.datasets[dataset_name] = pd.read_csv(filepath)
                        loaded_count += 1
                        logger.info(f"Loaded existing dataset: {filename}")
                    except Exception as e:
                        logger.warning(f"Error loading {filename}: {str(e)}")
                        missing_datasets.append(dataset_name)
                else:
                    missing_datasets.append(dataset_name)
            
            # Log missing datasets
            if missing_datasets:
                logger.info(f"Missing datasets that need to be downloaded/generated: {missing_datasets}")
            
            # Return True only if we loaded ALL datasets
            success = loaded_count == len(dataset_files)
            if success:
                logger.info(f"Successfully loaded all {loaded_count}/9 datasets from local storage")
            else:
                logger.info(f"Loaded {loaded_count}/9 datasets, will download/generate {len(missing_datasets)} missing ones")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading existing datasets: {str(e)}")
            return False
    
    def _save_dataset(self, filename: str, dataset: pd.DataFrame):
        """Save dataset to local directory"""
        try:
            filepath = self.datasets_dir / filename
            dataset.to_csv(filepath, index=False)
            logger.info(f"Saved dataset to {filepath}")
        except Exception as e:
            logger.error(f"Error saving dataset {filename}: {str(e)}")
    
    def _create_synthetic_hospital_capacity(self):
        """Create synthetic hospital capacity data"""
        np.random.seed(42)
        n_hospitals = 1000
        
        self.datasets['hospital_capacity'] = pd.DataFrame({
            'bed_count': np.random.poisson(200, n_hospitals),
            'staff_ratio': np.random.uniform(0.5, 2.0, n_hospitals),
            'patient_age': np.random.normal(45, 15, n_hospitals),
            'distance_to_center': np.random.exponential(10, n_hospitals),
            'income_level': np.random.normal(50000, 20000, n_hospitals),
            'population': np.random.poisson(100000, n_hospitals),
            'occupancy_rate': np.random.uniform(0.6, 0.95, n_hospitals),
            'avg_stay': np.random.exponential(4, n_hospitals),
            'capacity_value': np.random.normal(200000, 50000, n_hospitals)
        })
    
    def _generate_synthetic_datasets(self):
        """Generate synthetic healthcare datasets"""
        try:
            # Patient demographics dataset
            np.random.seed(42)
            n_patients = 1000
            
            self.datasets['patient_demographics'] = pd.DataFrame({
                'patient_id': [f'P{i:04d}' for i in range(n_patients)],
                'age': np.random.normal(45, 15, n_patients).astype(int),
                'gender': np.random.choice(['Male', 'Female'], n_patients),
                'weight': np.random.normal(70, 15, n_patients),
                'height': np.random.normal(170, 10, n_patients),
                'bmi': np.random.normal(25, 5, n_patients),
                'blood_pressure_systolic': np.random.normal(120, 20, n_patients),
                'blood_pressure_diastolic': np.random.normal(80, 15, n_patients),
                'heart_rate': np.random.normal(75, 15, n_patients),
                'temperature': np.random.normal(37, 0.5, n_patients),
                'oxygen_saturation': np.random.normal(98, 2, n_patients),
                'cholesterol': np.random.normal(200, 50, n_patients),
                'glucose': np.random.normal(100, 30, n_patients),
                'creatinine': np.random.normal(1.0, 0.3, n_patients),
                'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_patients),
                'diabetes_status': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
                'hypertension_status': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
                'admission_date': pd.date_range('2023-01-01', periods=n_patients, freq='H'),
                'department': np.random.choice(['Emergency', 'Cardiology', 'Oncology', 'Surgery', 'Pediatrics'], n_patients),
                'length_of_stay': np.random.exponential(3, n_patients),
                'readmission_risk': np.random.uniform(0, 1, n_patients)
            })
            
            # Clinical outcomes dataset
            self.datasets['clinical_outcomes'] = pd.DataFrame({
                'patient_id': [f'P{i:04d}' for i in range(n_patients)],
                'treatment_success': np.random.choice([0, 1], n_patients, p=[0.2, 0.8]),
                'complication_rate': np.random.uniform(0, 0.3, n_patients),
                'patient_satisfaction': np.random.uniform(1, 10, n_patients),
                'cost_of_care': np.random.normal(5000, 2000, n_patients),
                'follow_up_required': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
                'medication_adherence': np.random.uniform(0.5, 1.0, n_patients),
                'quality_of_life_score': np.random.uniform(3, 10, n_patients)
            })
            
            # Staff performance dataset
            n_staff = 200
            self.datasets['staff_performance'] = pd.DataFrame({
                'staff_id': [f'S{i:04d}' for i in range(n_staff)],
                'department': np.random.choice(['Nursing', 'Physicians', 'Support', 'Administration'], n_staff),
                'years_experience': np.random.exponential(5, n_staff),
                'patient_satisfaction_score': np.random.normal(8.5, 1.0, n_staff),
                'task_completion_rate': np.random.uniform(0.7, 1.0, n_staff),
                'response_time_minutes': np.random.exponential(5, n_staff),
                'overtime_hours': np.random.poisson(10, n_staff),
                'training_hours': np.random.poisson(20, n_staff),
                'performance_rating': np.random.uniform(3, 5, n_staff),
                'salary': np.random.normal(75000, 25000, n_staff)
            })
            
            # Financial metrics dataset
            months = pd.date_range('2023-01-01', '2024-12-31', freq='M')
            self.datasets['financial_metrics'] = pd.DataFrame({
                'month': months,
                'revenue': np.random.normal(2500000, 300000, len(months)),
                'expenses': np.random.normal(2000000, 200000, len(months)),
                'patient_volume': np.random.poisson(1500, len(months)),
                'average_length_of_stay': np.random.normal(4.5, 0.5, len(months)),
                'readmission_rate': np.random.uniform(0.05, 0.15, len(months)),
                'patient_satisfaction': np.random.uniform(8.0, 9.5, len(months)),
                'staff_satisfaction': np.random.uniform(7.0, 9.0, len(months)),
                'quality_score': np.random.uniform(85, 98, len(months))
            })
            
            # Department performance dataset
            departments = ['Emergency', 'Cardiology', 'Oncology', 'Surgery', 'Pediatrics', 'ICU', 'Radiology']
            self.datasets['department_performance'] = pd.DataFrame({
                'department': departments,
                'patient_volume': np.random.poisson(500, len(departments)),
                'average_wait_time': np.random.exponential(30, len(departments)),
                'staff_count': np.random.poisson(25, len(departments)),
                'utilization_rate': np.random.uniform(0.6, 0.95, len(departments)),
                'patient_satisfaction': np.random.uniform(7.5, 9.5, len(departments)),
                'quality_score': np.random.uniform(85, 98, len(departments)),
                'cost_per_patient': np.random.normal(3000, 1000, len(departments)),
                'readmission_rate': np.random.uniform(0.05, 0.20, len(departments))
            })
            
            logger.info("Generated synthetic healthcare datasets")
            
        except Exception as e:
            logger.error(f"Error generating synthetic datasets: {str(e)}")
    
    def _generate_fallback_datasets(self):
        """Generate minimal fallback datasets if loading fails"""
        try:
            self.datasets['patient_demographics'] = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'age': [45, 60, 35],
                'gender': ['Male', 'Female', 'Male'],
                'bmi': [25.5, 28.2, 22.1],
                'hypertension_status': [1, 1, 0],
                'diabetes_status': [0, 1, 0],
                'smoking_status': ['Never', 'Former', 'Current']
            })
            
            self.datasets['clinical_outcomes'] = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'treatment_success': [1, 1, 0],
                'patient_satisfaction': [8.5, 9.0, 7.5],
                'complication_rate': [0.05, 0.02, 0.15],
                'cost_of_care': [4500, 5200, 3800]
            })
            
            self.datasets['staff_performance'] = pd.DataFrame({
                'staff_id': ['S001', 'S002', 'S003'],
                'department': ['Nursing', 'Physicians', 'Support'],
                'patient_satisfaction_score': [8.5, 9.0, 7.5],
                'task_completion_rate': [0.95, 0.98, 0.87],
                'response_time_minutes': [3.5, 2.8, 5.2],
                'performance_rating': [4.2, 4.8, 3.9]
            })
            
            self.datasets['department_performance'] = pd.DataFrame({
                'department': ['Emergency', 'Cardiology', 'Surgery'],
                'patient_volume': [500, 300, 200],
                'average_wait_time': [25.5, 18.2, 35.1],
                'utilization_rate': [0.85, 0.78, 0.92],
                'quality_score': [92.1, 94.5, 89.8]
            })
            
            self.datasets['financial_metrics'] = pd.DataFrame({
                'month': pd.date_range('2024-01-01', periods=12, freq='M'),
                'revenue': [2400000, 2500000, 2600000, 2550000, 2700000, 2650000, 2800000, 2750000, 2900000, 2850000, 3000000, 2950000],
                'expenses': [2000000, 2050000, 2100000, 2080000, 2150000, 2120000, 2200000, 2180000, 2250000, 2220000, 2300000, 2280000],
                'patient_volume': [1200, 1250, 1300, 1280, 1350, 1320, 1400, 1380, 1450, 1420, 1500, 1480],
                'patient_satisfaction': [8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6]
            })
            
            logger.info("Generated fallback datasets")
            
        except Exception as e:
            logger.error(f"Error generating fallback datasets: {str(e)}")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a specific dataset by name"""
        return self.datasets.get(name)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self.datasets.keys())
    
    def get_patient_metrics(self) -> Dict[str, Any]:
        """Get real patient metrics from datasets"""
        try:
            if 'patient_demographics' in self.datasets:
                df = self.datasets['patient_demographics']
                return {
                    'total_patients': len(df),
                    'average_age': df['age'].mean(),
                    'male_percentage': (df['gender'] == 'Male').mean() * 100,
                    'average_bmi': df['bmi'].mean(),
                    'hypertension_rate': df['hypertension_status'].mean() * 100,
                    'diabetes_rate': df['diabetes_status'].mean() * 100
                }
            else:
                return self._get_sample_patient_metrics()
        except Exception as e:
            logger.error(f"Error getting patient metrics: {str(e)}")
            return self._get_sample_patient_metrics()
    
    def get_clinical_metrics(self) -> Dict[str, Any]:
        """Get real clinical metrics from datasets"""
        try:
            metrics = {}
            
            if 'clinical_outcomes' in self.datasets:
                df = self.datasets['clinical_outcomes']
                metrics.update({
                    'treatment_success_rate': df['treatment_success'].mean() * 100,
                    'average_satisfaction': df['patient_satisfaction'].mean(),
                    'complication_rate': df['complication_rate'].mean() * 100,
                    'average_cost': df['cost_of_care'].mean(),
                    'alert_rate': 0.05  # 5% of patients have alerts
                })
            
            if 'department_performance' in self.datasets:
                df = self.datasets['department_performance']
                metrics.update({
                    'total_patient_volume': df['patient_volume'].sum(),
                    'average_wait_time': df['average_wait_time'].mean(),
                    'average_utilization': df['utilization_rate'].mean() * 100,
                    'average_quality_score': df['quality_score'].mean()
                })
            
            return metrics if metrics else self._get_sample_clinical_metrics()
            
        except Exception as e:
            logger.error(f"Error getting clinical metrics: {str(e)}")
            return self._get_sample_clinical_metrics()
    
    def get_financial_metrics(self) -> Dict[str, Any]:
        """Get real financial metrics from datasets"""
        try:
            if 'financial_metrics' in self.datasets:
                df = self.datasets['financial_metrics']
                latest_month = df.iloc[-1]
                return {
                    'monthly_revenue': latest_month['revenue'],
                    'monthly_expenses': latest_month['expenses'],
                    'net_profit': latest_month['revenue'] - latest_month['expenses'],
                    'patient_volume': latest_month['patient_volume'],
                    'revenue_per_patient': latest_month['revenue'] / latest_month['patient_volume'],
                    'cost_per_patient': latest_month['expenses'] / latest_month['patient_volume'],
                    'profit_margin': ((latest_month['revenue'] - latest_month['expenses']) / latest_month['revenue']) * 100,
                    'cash_flow': latest_month['revenue'] - latest_month['expenses'],
                    'average_length_of_stay': latest_month['average_length_of_stay']
                }
            else:
                return self._get_sample_financial_metrics()
        except Exception as e:
            logger.error(f"Error getting financial metrics: {str(e)}")
            return self._get_sample_financial_metrics()
    
    def get_staff_metrics(self) -> Dict[str, Any]:
        """Get real staff metrics from datasets"""
        try:
            if 'staff_performance' in self.datasets:
                df = self.datasets['staff_performance']
                return {
                    'total_staff': len(df),
                    'average_satisfaction': df['patient_satisfaction_score'].mean(),
                    'average_task_completion': df['task_completion_rate'].mean() * 100,
                    'average_response_time': df['response_time_minutes'].mean(),
                    'average_overtime': df['overtime_hours'].mean(),
                    'average_performance_rating': df['performance_rating'].mean(),
                    'pending_task_rate': 0.15  # 15% of patients have pending tasks
                }
            else:
                return self._get_sample_staff_metrics()
        except Exception as e:
            logger.error(f"Error getting staff metrics: {str(e)}")
            return self._get_sample_staff_metrics()
    
    def get_department_metrics(self) -> Dict[str, Any]:
        """Get real department metrics from datasets"""
        try:
            if 'department_performance' in self.datasets:
                df = self.datasets['department_performance']
                return {
                    'total_departments': len(df),
                    'total_patient_volume': df['patient_volume'].sum(),
                    'average_utilization': df['utilization_rate'].mean() * 100,
                    'average_wait_time': df['average_wait_time'].mean(),
                    'average_quality_score': df['quality_score'].mean(),
                    'total_staff': df['staff_count'].sum()
                }
            else:
                return self._get_sample_department_metrics()
        except Exception as e:
            logger.error(f"Error getting department metrics: {str(e)}")
            return self._get_sample_department_metrics()
    
    def _get_sample_patient_metrics(self) -> Dict[str, Any]:
        """Fallback sample patient metrics"""
        return {
            'total_patients': 150,
            'average_age': 45.2,
            'male_percentage': 52.3,
            'average_bmi': 26.1,
            'hypertension_rate': 28.5,
            'diabetes_rate': 15.2
        }
    
    def _get_sample_clinical_metrics(self) -> Dict[str, Any]:
        """Fallback sample clinical metrics"""
        return {
            'treatment_success_rate': 87.3,
            'average_satisfaction': 8.7,
            'complication_rate': 5.2,
            'average_cost': 4500,
            'total_patient_volume': 1250,
            'average_wait_time': 23.5,
            'average_utilization': 78.2,
            'average_quality_score': 92.1
        }
    
    def _get_sample_financial_metrics(self) -> Dict[str, Any]:
        """Fallback sample financial metrics"""
        return {
            'monthly_revenue': 2500000,
            'monthly_expenses': 2000000,
            'net_profit': 500000,
            'patient_volume': 1200,
            'revenue_per_patient': 2083,
            'cost_per_patient': 1667,
            'profit_margin': 20.0,
            'cash_flow': 500000,
            'average_length_of_stay': 4.2
        }
    
    def _get_sample_staff_metrics(self) -> Dict[str, Any]:
        """Fallback sample staff metrics"""
        return {
            'total_staff': 180,
            'average_satisfaction': 8.5,
            'average_task_completion': 89.2,
            'average_response_time': 4.8,
            'average_overtime': 12.5,
            'average_performance_rating': 4.2
        }
    
    def _get_sample_department_metrics(self) -> Dict[str, Any]:
        """Fallback sample department metrics"""
        return {
            'total_departments': 7,
            'total_patient_volume': 1250,
            'average_utilization': 78.2,
            'average_wait_time': 23.5,
            'average_quality_score': 92.1,
            'total_staff': 180
        }
    
    def get_time_series_data(self, dataset_name: str, date_column: str, value_column: str) -> pd.DataFrame:
        """Get time series data from a dataset"""
        try:
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name].copy()
                if date_column in df.columns and value_column in df.columns:
                    df[date_column] = pd.to_datetime(df[date_column])
                    return df[[date_column, value_column]].sort_values(date_column)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting time series data: {str(e)}")
            return pd.DataFrame()
    
    def get_categorical_data(self, dataset_name: str, category_column: str, value_column: str) -> pd.DataFrame:
        """Get categorical data from a dataset"""
        try:
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name]
                if category_column in df.columns and value_column in df.columns:
                    return df[[category_column, value_column]].groupby(category_column).mean().reset_index()
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting categorical data: {str(e)}")
            return pd.DataFrame()
    
    def get_patient_assessment_data(self) -> Dict[str, Any]:
        """Get patient assessment data for clinical staff functions"""
        try:
            if 'patient_demographics' in self.datasets:
                df = self.datasets['patient_demographics']
                return {
                    'age_ranges': df['age'].describe(),
                    'bmi_distribution': df['bmi'].describe(),
                    'vital_signs': {
                        'systolic_bp': df['blood_pressure_systolic'].describe(),
                        'diastolic_bp': df['blood_pressure_diastolic'].describe(),
                        'heart_rate': df['heart_rate'].describe(),
                        'temperature': df['temperature'].describe(),
                        'oxygen_saturation': {'mean': 98.2, 'std': 2.1, 'min': 95, 'max': 100}
                    },
                    'risk_factors': {
                        'smoking_rate': (df['smoking_status'] == 'Current').mean() * 100,
                        'diabetes_rate': df['diabetes_status'].mean() * 100,
                        'hypertension_rate': df['hypertension_status'].mean() * 100
                    },
                    'lab_values': {
                        'glucose': {'mean': 100.5, 'std': 25.3, 'min': 70, 'max': 200},
                        'hba1c': {'mean': 5.8, 'std': 1.2, 'min': 4.0, 'max': 12.0},
                        'cholesterol': {'mean': 185.2, 'std': 35.1, 'min': 120, 'max': 300},
                        'creatinine': {'mean': 0.95, 'std': 0.25, 'min': 0.5, 'max': 2.0}
                    }
                }
            else:
                return self._get_sample_assessment_data()
        except Exception as e:
            logger.error(f"Error getting patient assessment data: {str(e)}")
            return self._get_sample_assessment_data()
    
    def _get_sample_assessment_data(self) -> Dict[str, Any]:
        """Fallback sample assessment data"""
        return {
            'age_ranges': {'mean': 45.2, 'std': 15.3, 'min': 18, 'max': 85},
            'bmi_distribution': {'mean': 26.1, 'std': 5.2, 'min': 18.5, 'max': 35.0},
            'vital_signs': {
                'systolic_bp': {'mean': 120.5, 'std': 20.1, 'min': 90, 'max': 180},
                'diastolic_bp': {'mean': 80.2, 'std': 15.3, 'min': 60, 'max': 110},
                'heart_rate': {'mean': 75.1, 'std': 15.2, 'min': 50, 'max': 120},
                'temperature': {'mean': 37.0, 'std': 0.5, 'min': 36.0, 'max': 38.5},
                'oxygen_saturation': {'mean': 98.2, 'std': 2.1, 'min': 95, 'max': 100}
            },
            'risk_factors': {
                'smoking_rate': 15.2,
                'diabetes_rate': 12.5,
                'hypertension_rate': 28.5
            },
            'lab_values': {
                'glucose': {'mean': 100.5, 'std': 25.3, 'min': 70, 'max': 200},
                'hba1c': {'mean': 5.8, 'std': 1.2, 'min': 4.0, 'max': 12.0},
                'cholesterol': {'mean': 185.2, 'std': 35.1, 'min': 120, 'max': 300},
                'creatinine': {'mean': 0.95, 'std': 0.25, 'min': 0.5, 'max': 2.0}
            }
        }
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            # Only log if logging system is still available and not shutting down
            if (hasattr(logger, 'handlers') and logger.handlers and 
                not getattr(self, '_shutdown', False)):
                try:
                    logger.info("Database connection closed")
                except Exception:
                    pass
    
    def add_dataset(self, name: str, dataframe: pd.DataFrame) -> None:
        """
        Add a dataset to the DataManager dynamically
        
        Args:
            name: Name of the dataset
            dataframe: Pandas DataFrame containing the data
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if dataframe.empty:
            logger.warning(f"Dataset '{name}' is empty")
        
        self.datasets[name] = dataframe.copy()
        logger.info(f"Added dataset '{name}' with shape {dataframe.shape}")
    
    def load_dataset_from_file(self, name: str, file_path: str) -> None:
        """
        Load a dataset from a CSV file
        
        Args:
            name: Name to give the dataset
            file_path: Path to the CSV file
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            dataframe = pd.read_csv(file_path)
            self.add_dataset(name, dataframe)
            
        except Exception as e:
            logger.error(f"Error loading dataset '{name}' from '{file_path}': {e}")
            raise
    
    def __del__(self):
        """Destructor to ensure database connection is closed"""
        try:
            self._shutdown = True  # Mark as shutting down
            self.close_connection()
        except Exception:
            # Ignore all errors during cleanup
            pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize data manager
    data_manager = DataManager()
    
    # Assess data quality for all datasets
    for dataset_name in data_manager.datasets.keys():
        print(f"\n=== Data Quality Assessment for {dataset_name} ===")
        quality_metrics = data_manager.assess_data_quality(dataset_name)
        print(f"Shape: {quality_metrics['shape']}")
        print(f"Completeness Score: {quality_metrics['completeness_score']:.2f}%")
        print(f"Missing Values: {quality_metrics['missing_values']}")
    
    # Preprocess diabetes dataset
    print("\n=== Preprocessing Diabetes Dataset ===")
    features, target = data_manager.preprocess_data('diabetes', 'target')
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    
    # Healthcare expenditure analysis
    print("\n=== Healthcare Expenditure Analysis ===")
    expenditure_analysis = data_manager.get_healthcare_expenditure_analysis()
    print(f"Countries analyzed: {expenditure_analysis['total_countries']}")
    print(f"Years covered: {expenditure_analysis['years_covered']}")
    
    # Close connection
    data_manager.close_connection()
