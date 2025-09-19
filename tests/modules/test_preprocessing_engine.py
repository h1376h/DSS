"""
Test Preprocessing Engine
=========================

Comprehensive tests for the PreprocessingEngine module
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.preprocessing_engine import PreprocessingEngine


class TestPreprocessingEngine(unittest.TestCase):
    """Test cases for PreprocessingEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessing_engine = PreprocessingEngine()
        
        # Create test datasets
        self.test_data_classification = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5] * 20,
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            'categorical': ['A', 'B', 'C', 'D', 'E'] * 20,
            'target_binary': [0, 1, 0, 1, 0] * 20,
            'target_multiclass': [0, 1, 2, 0, 1] * 20
        })
        
        self.test_data_regression = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target_continuous': np.random.randn(100)
        })
        
        self.test_data_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5] * 20,
            'feature2': [0.1, np.nan, 0.3, 0.4, 0.5] * 20,
            'categorical': ['A', 'B', np.nan, 'D', 'E'] * 20,
            'target': [0, 1, 0, 1, 0] * 20
        })
    
    def test_analyze_target_column_classification_suitable(self):
        """Test target column analysis for classification-suitable data"""
        # Test binary classification
        analysis = self.preprocessing_engine.analyze_target_column(
            self.test_data_classification, 'target_binary'
        )
        
        self.assertIn('classification_suitable', analysis)
        self.assertTrue(analysis['classification_suitable'])
        self.assertGreater(analysis['classification_confidence'], 80)
        self.assertEqual(analysis['primary_recommendation'], 'classification')
        
        # Test multiclass classification
        analysis = self.preprocessing_engine.analyze_target_column(
            self.test_data_classification, 'target_multiclass'
        )
        
        self.assertTrue(analysis['classification_suitable'])
        self.assertEqual(analysis['primary_recommendation'], 'classification')
    
    def test_analyze_target_column_regression_suitable(self):
        """Test target column analysis for regression-suitable data"""
        analysis = self.preprocessing_engine.analyze_target_column(
            self.test_data_regression, 'target_continuous'
        )
        
        self.assertIn('regression_suitable', analysis)
        self.assertTrue(analysis['regression_suitable'])
        self.assertGreater(analysis['regression_confidence'], 80)
        self.assertEqual(analysis['primary_recommendation'], 'regression')
    
    def test_analyze_target_column_categorical(self):
        """Test target column analysis for categorical data"""
        analysis = self.preprocessing_engine.analyze_target_column(
            self.test_data_classification, 'categorical'
        )
        
        self.assertTrue(analysis['classification_suitable'])
        self.assertEqual(analysis['classification_confidence'], 100)
        self.assertEqual(analysis['primary_recommendation'], 'classification')
    
    def test_get_preprocessing_options_classification(self):
        """Test preprocessing options for classification"""
        options = self.preprocessing_engine.get_preprocessing_options(
            self.test_data_classification, 'target_binary', 'classification'
        )
        
        self.assertIn('scaling_options', options)
        self.assertIn('encoding_options', options)
        self.assertIn('feature_engineering', options)
        self.assertIn('recommendations', options)
        
        # Check that suitable options are provided
        suitable_scaling = [opt for opt in options['scaling_options'] if opt['suitable']]
        self.assertGreater(len(suitable_scaling), 0)
    
    def test_get_preprocessing_options_regression(self):
        """Test preprocessing options for regression"""
        options = self.preprocessing_engine.get_preprocessing_options(
            self.test_data_regression, 'target_continuous', 'regression'
        )
        
        self.assertIn('scaling_options', options)
        self.assertIn('feature_engineering', options)
        
        # Should have scaling options for numeric features
        suitable_scaling = [opt for opt in options['scaling_options'] if opt['suitable']]
        self.assertGreater(len(suitable_scaling), 0)
    
    def test_preprocess_data_basic(self):
        """Test basic data preprocessing"""
        features, target = self.preprocessing_engine.preprocess_data(
            self.test_data_classification, 'target_binary'
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        self.assertEqual(len(features), len(target))
        self.assertNotIn('target_binary', features.columns)
    
    def test_preprocess_data_with_config(self):
        """Test data preprocessing with configuration"""
        config = {
            'scaling_method': 'standard_scaler',
            'encoding_method': 'one_hot_encoding'
        }
        
        features, target = self.preprocessing_engine.preprocess_data(
            self.test_data_classification, 'target_binary', config
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        processed_features = self.preprocessing_engine._handle_missing_values(
            self.test_data_missing[['feature1', 'feature2', 'categorical']]
        )
        
        # Check that missing values are handled
        self.assertFalse(processed_features.isnull().any().any())
    
    def test_apply_scaling(self):
        """Test feature scaling"""
        numeric_features = self.test_data_classification[['feature1', 'feature2']]
        
        # Test standard scaling
        scaled_features = self.preprocessing_engine._apply_scaling(
            numeric_features, 'standard_scaler'
        )
        
        self.assertIsInstance(scaled_features, pd.DataFrame)
        self.assertEqual(scaled_features.shape, numeric_features.shape)
    
    def test_apply_encoding(self):
        """Test categorical encoding"""
        categorical_features = self.test_data_classification[['categorical']]
        
        # Test one-hot encoding
        encoded_features = self.preprocessing_engine._apply_encoding(
            categorical_features, 'one_hot_encoding'
        )
        
        self.assertIsInstance(encoded_features, pd.DataFrame)
        # Should have more columns after one-hot encoding
        self.assertGreaterEqual(len(encoded_features.columns), len(categorical_features.columns))
    
    def test_error_handling(self):
        """Test error handling in preprocessing"""
        # Test with non-existent column - should raise ValueError
        with self.assertRaises(ValueError):
            self.preprocessing_engine.analyze_target_column(
                self.test_data_classification, 'non_existent_column'
            )
        
        # Test with empty dataset - should raise ValueError
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.preprocessing_engine.analyze_target_column(
                empty_df, 'target'
            )
    
    def test_data_quality_issues_detection(self):
        """Test data quality issues detection"""
        options = self.preprocessing_engine.get_preprocessing_options(
            self.test_data_missing, 'target', 'classification'
        )
        
        if 'data_quality_issues' in options and options['data_quality_issues']:
            self.assertIn('missing_values', [issue['issue'] for issue in options['data_quality_issues']])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
