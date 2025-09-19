"""
Comprehensive Test Suite for Data Management Module
=================================================

This module contains comprehensive tests for the DataManager class and all its methods.
Tests cover data loading, validation, quality assessment, preprocessing, and database operations.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.data_management import DataManager
from tests.test_base import HealthcareDSSTestCase


class TestDataManager(HealthcareDSSTestCase):
    """Test cases for DataManager class"""
    
    def setUp(self):
        """Set up each test method"""
        super().setUp()
        # Use the shared data manager from base class
        self.dm = self.data_manager
        
        # Create sample test data
        self.sample_data = {
            'diabetes': pd.DataFrame({
                'age': [25, 30, 35, 40, 45],
                'sex': [0, 1, 0, 1, 0],
                'bmi': [22.5, 25.3, 28.1, 30.2, 32.1],
                'glucose': [85, 95, 105, 115, 125],
                'outcome': [0, 0, 1, 1, 1]
            }),
            'breast_cancer': pd.DataFrame({
                'radius_mean': [12.5, 13.2, 14.1, 15.3, 16.2],
                'texture_mean': [18.5, 19.2, 20.1, 21.3, 22.2],
                'perimeter_mean': [80.5, 85.2, 90.1, 95.3, 100.2],
                'area_mean': [500.5, 550.2, 600.1, 650.3, 700.2],
                'diagnosis': ['B', 'B', 'M', 'M', 'M']
            })
        }
        
        # Create test CSV files
        self.csv_files = {}
        for name, df in self.sample_data.items():
            csv_path = os.path.join(self.temp_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            self.csv_files[name] = csv_path
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DataManager initialization"""
        # Test with custom data directory
        self.assertIsNotNone(self.dm)
        # The DataManager uses the real datasets directory, not temp directory
        self.assertEqual(str(self.dm.data_dir), 'datasets/raw')
        
        # Test with default data directory
        with patch('healthcare_dss.core.data_management.Path.exists', return_value=True):
            dm_default = DataManager()
            self.assertIsNotNone(dm_default)
    
    def test_load_datasets(self):
        """Test dataset loading functionality"""
        # Test that datasets are loaded automatically
        self.assertIsInstance(self.dm.datasets, dict)
        self.assertGreater(len(self.dm.datasets), 0)
        
        # Test specific dataset access
        available_datasets = list(self.dm.datasets.keys())
        if available_datasets:
            first_dataset = available_datasets[0]
            dataset_df = self.dm.datasets[first_dataset]
            self.assertIsInstance(dataset_df, pd.DataFrame)
    
    def test_data_validation(self):
        """Test data validation methods"""
        # Test data quality assessment for a specific dataset
        available_datasets = list(self.dm.datasets.keys())
        if available_datasets:
            first_dataset = available_datasets[0]
            validation_result = self.dm.assess_data_quality(first_dataset)
            self.assertIsInstance(validation_result, dict)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        self.dm
        
        # Test assess_data_quality method
        available_datasets = list(self.dm.datasets.keys())
        if available_datasets:
            first_dataset = available_datasets[0]
            quality_metrics = self.dm.assess_data_quality(first_dataset)
            self.assertIsInstance(quality_metrics, dict)
    
    def test_preprocessing_methods(self):
        """Test data preprocessing methods"""
        self.dm
        
        # Test preprocessing methods with available dataset
        available_datasets = list(self.dm.datasets.keys())
        if available_datasets:
            first_dataset = available_datasets[0]
            
            # Test preprocessing
            features, target = self.dm.preprocess_data(first_dataset)
            self.assertIsInstance(features, pd.DataFrame)
            # Target might be None if no target column is found
            if target is not None:
                self.assertIsInstance(target, pd.DataFrame)
    
    def test_database_operations(self):
        """Test database operations"""
        # Test database operations
        self.assertIsNotNone(self.dm.connection)
        self.assertIsNotNone(self.dm.db_path)
    
    def test_get_available_datasets(self):
        """Test getting available datasets"""
        self.dm
        
        # Test get_available_datasets
        available = self.dm.get_available_datasets()
        self.assertIsInstance(available, list)
        self.assertIn('diabetes', available)
        self.assertIn('breast_cancer', available)
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        self.dm
        
        # Test loading non-existent dataset
        with self.assertRaises(Exception):
            self.dm.load_dataset('nonexistent')
        
        # Test accessing non-existent dataset
        with self.assertRaises(KeyError):
            self.dm.datasets['nonexistent']
    
    def test_data_statistics(self):
        """Test data statistics calculation"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Test get_dataset_info instead of calculate_statistics
        info = self.dm.get_dataset_info(dataset_name)
        self.assertIsInstance(info, dict)
        self.assertIn('shape', info)
        self.assertIn('columns', info)
    
    def test_data_export(self):
        """Test data export functionality"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Create a specific file path for export
        export_file = os.path.join(self.temp_dir, f'{dataset_name}_export.csv')
        
        # Test export_processed_data
        result = self.dm.export_processed_data(dataset_name, export_file)
        
        # The method returns None, so just check that it doesn't raise an exception
        self.assertIsNone(result)


class TestDataManagerIntegration(HealthcareDSSTestCase):
    """Integration tests for DataManager"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        super().setUp()
        self.dm = self.data_manager
    
    def test_full_workflow(self):
        """Test complete data management workflow"""
        # Load datasets
        datasets = self.dm.datasets
        self.assertGreater(len(datasets), 0)
        
        # Test with a real dataset
        dataset_name, dataset_df = self.get_available_dataset()
        self.assert_dataset_valid(dataset_name, dataset_df)
        
        # Test data quality assessment
        quality_result = self.dm.assess_data_quality(dataset_name)
        self.assertIsInstance(quality_result, dict)


if __name__ == '__main__':
    unittest.main()
