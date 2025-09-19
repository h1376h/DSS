"""
Base test class for Healthcare DSS tests
========================================

This module provides a base test class with common setup and utilities
for all Healthcare DSS tests, ensuring access to all datasets and
providing helper methods for testing.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the main components
from healthcare_dss.core.data_management import DataManager
from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.knowledge_management import KnowledgeManager


class HealthcareDSSTestCase(unittest.TestCase):
    """
    Base test case class for Healthcare DSS tests
    
    Provides:
    - Access to all 12 datasets
    - Common setup and teardown
    - Helper methods for testing
    - Dynamic dataset selection (no hardcoded values)
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with all datasets"""
        # Use the real datasets directory instead of temporary directory
        cls.data_manager = DataManager(data_dir="datasets/raw")
        
        # Verify we have all datasets
        cls.available_datasets = list(cls.data_manager.datasets.keys())
        cls.expected_datasets = [
            'diabetes', 'breast_cancer', 'healthcare_expenditure', 'wine', 'linnerud',
            'medication_effectiveness', 'hospital_capacity', 'patient_demographics',
            'clinical_outcomes', 'staff_performance', 'financial_metrics', 'department_performance'
        ]
        
        # Verify all expected datasets are available
        missing_datasets = set(cls.expected_datasets) - set(cls.available_datasets)
        if missing_datasets:
            raise unittest.SkipTest(f"Missing datasets: {missing_datasets}")
        
        # Initialize other managers
        cls.model_manager = ModelManager(cls.data_manager)
        cls.knowledge_manager = KnowledgeManager(cls.data_manager, cls.model_manager)
    
    def setUp(self):
        """Set up each test method"""
        # Create temporary directory for test-specific files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, 'test.db')
    
    def tearDown(self):
        """Clean up after each test method"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def get_available_dataset(self, dataset_type=None):
        """
        Get an available dataset dynamically
        
        Args:
            dataset_type (str): Optional type hint ('classification', 'regression', 'clustering')
        
        Returns:
            tuple: (dataset_name, dataset_df)
        """
        if dataset_type == 'classification':
            # Prefer datasets known to be good for classification
            preferred = ['diabetes', 'breast_cancer', 'wine']
            for name in preferred:
                if name in self.available_datasets:
                    return name, self.data_manager.datasets[name]
        
        elif dataset_type == 'regression':
            # Prefer datasets known to be good for regression
            preferred = ['linnerud', 'healthcare_expenditure']
            for name in preferred:
                if name in self.available_datasets:
                    return name, self.data_manager.datasets[name]
        
        elif dataset_type == 'clustering':
            # Prefer datasets known to be good for clustering
            preferred = ['wine', 'breast_cancer']
            for name in preferred:
                if name in self.available_datasets:
                    return name, self.data_manager.datasets[name]
        
        # Return first available dataset
        if self.available_datasets:
            name = self.available_datasets[0]
            return name, self.data_manager.datasets[name]
        
        raise unittest.SkipTest("No datasets available")
    
    def get_dataset_with_target(self, task_type='classification'):
        """
        Get a dataset that has a clear target column for the specified task type
        
        Args:
            task_type: 'classification', 'regression', or 'clustering'
        
        Returns:
            tuple: (dataset_name, dataset_df, target_column)
        """
        # Define datasets by their actual task type based on data characteristics
        classification_datasets = [
            ('breast_cancer', 'target'),  # breast_cancer has numeric target (0/1)
            ('wine', 'target'),  # wine has numeric target (0/1/2)
            ('medication_effectiveness', 'effectiveness'),  # numeric effectiveness score
            ('clinical_outcomes', 'treatment_success')  # binary success (0/1)
        ]
        
        regression_datasets = [
            ('diabetes', 'target'),  # diabetes has continuous target
            ('linnerud', 'Weight'),  # linnerud has continuous targets
            ('healthcare_expenditure', '2015 [YR2015]'),  # continuous expenditure values
            ('financial_metrics', 'revenue')  # continuous financial values
        ]
        
        clustering_datasets = [
            ('wine', 'target'),  # wine can be used for clustering
            ('breast_cancer', 'target'),  # breast_cancer can be used for clustering
            ('patient_demographics', 'age'),  # demographic clustering
            ('staff_performance', 'performance_rating')  # performance clustering
        ]
        
        # Select appropriate datasets based on task type
        if task_type == 'classification':
            target_datasets = classification_datasets
        elif task_type == 'regression':
            target_datasets = regression_datasets
        elif task_type == 'clustering':
            target_datasets = clustering_datasets
        else:
            # Default to classification
            target_datasets = classification_datasets
        
        # Try to find a suitable dataset
        for name, target_col in target_datasets:
            if name in self.available_datasets:
                df = self.data_manager.datasets[name]
                if target_col in df.columns:
                    return name, df, target_col
        
        # Fallback: try to find any dataset with a target column
        for name in self.available_datasets:
            df = self.data_manager.datasets[name]
            # Look for common target column names
            for target_col in ['target', 'label', 'class', 'outcome', 'y']:
                if target_col in df.columns:
                    return name, df, target_col
        
        raise unittest.SkipTest(f"No dataset with target column found for {task_type}")
    
    def create_sample_data(self, n_samples=100, n_features=5, task_type='classification'):
        """
        Create sample data for testing
        
        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            task_type (str): Type of task ('classification', 'regression')
        
        Returns:
            pd.DataFrame: Sample data
        """
        np.random.seed(42)  # For reproducible tests
        
        # Create feature columns
        data = {}
        for i in range(n_features):
            data[f'feature_{i+1}'] = np.random.normal(0, 1, n_samples)
        
        # Create target column
        if task_type == 'classification':
            data['target'] = np.random.choice([0, 1], n_samples)
        else:  # regression
            data['target'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    def assert_dataset_valid(self, dataset_name, dataset_df):
        """
        Assert that a dataset is valid for testing
        
        Args:
            dataset_name (str): Name of the dataset
            dataset_df (pd.DataFrame): Dataset DataFrame
        """
        self.assertIsInstance(dataset_df, pd.DataFrame)
        self.assertGreater(len(dataset_df), 0, f"Dataset {dataset_name} is empty")
        self.assertGreater(len(dataset_df.columns), 0, f"Dataset {dataset_name} has no columns")
    
    def assert_model_result_valid(self, result):
        """
        Assert that a model training result is valid
        
        Args:
            result (dict): Model training result
        """
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIsInstance(result['metrics'], dict)
    
    def skip_if_no_dataset(self, dataset_name):
        """
        Skip test if dataset is not available
        
        Args:
            dataset_name (str): Name of required dataset
        """
        if dataset_name not in self.available_datasets:
            self.skipTest(f"Dataset {dataset_name} not available")
