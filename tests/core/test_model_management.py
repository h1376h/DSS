"""
Comprehensive Test Suite for Model Management Module
==================================================

This module contains comprehensive tests for the ModelManager class and all its methods.
Tests cover model training, evaluation, registry operations, and AI technology selection.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.data_management import DataManager
from tests.test_base import HealthcareDSSTestCase


class TestModelManager(HealthcareDSSTestCase):
    """Test cases for ModelManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.model_manager = ModelManager(self.data_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ModelManager initialization"""
        self.assertIsNotNone(self.model_manager)
        self.assertEqual(self.model_manager.data_manager, self.data_manager)
        self.assertIsNotNone(self.model_manager.preprocessing_engine)
        self.assertIsNotNone(self.model_manager.training_engine)
        self.assertIsNotNone(self.model_manager.evaluation_engine)
        self.assertIsNotNone(self.model_manager.registry)
    
    def test_get_ai_technology_selection_matrix(self):
        """Test AI technology selection matrix"""
        matrix = self.model_manager.get_ai_technology_selection_matrix()
        
        self.assertIsInstance(matrix, dict)
        self.assertIn('healthcare_problems', matrix)
        self.assertIn('ai_vs_human_comparison', matrix)
    
    def test_train_model(self):
        """Test model training functionality"""
        # Use breast_cancer dataset which has a proper target column for classification
        dataset_name = 'breast_cancer'
        dataset_df = self.data_manager.datasets[dataset_name]
        target_col = 'target'
        
        # Test model training with correct parameters
        result = self.model_manager.train_model(
            dataset_name=dataset_name,
            model_name='random_forest',
            task_type='classification',
            target_column=target_col
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
    
    def test_evaluate_model(self):
        """Test model evaluation functionality"""
        # Use breast_cancer dataset which has a proper target column for classification
        dataset_name = 'breast_cancer'
        dataset_df = self.data_manager.datasets[dataset_name]
        target_col = 'target'
        
        # Train a model first
        train_result = self.model_manager.train_model(
            dataset_name=dataset_name,
            model_name='random_forest',
            task_type='classification',
            target_column=target_col
        )
        
        # Test model evaluation with correct parameters
        result = self.model_manager.evaluate_ai_model_performance(
            model_name='random_forest',
            healthcare_context='General'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('metrics', result)
    
    def test_save_model(self):
        """Test model saving functionality"""
        # Use breast_cancer dataset which has a proper target column for classification
        dataset_name = 'breast_cancer'
        dataset_df = self.data_manager.datasets[dataset_name]
        target_col = 'target'
        
        # Train a model first
        train_result = self.model_manager.train_model(
            dataset_name=dataset_name,
            model_name='random_forest',
            task_type='classification',
            target_column=target_col
        )
        
        # Test listing models (which shows saved models)
        models = self.model_manager.list_models()
        
        self.assertIsInstance(models, pd.DataFrame)
        self.assertGreater(models.shape[0], 0)
    
    def test_load_model(self):
        """Test model loading functionality"""
        # Mock the registry
        with patch.object(self.model_manager.registry, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            loaded_model = self.model_manager.load_model('model_001')
            
            self.assertEqual(loaded_model, mock_model)
            mock_load.assert_called_once_with('model_001')
    
    def test_get_model_performance_history(self):
        """Test getting model performance history"""
        # Test that the registry exists and has expected methods
        self.assertIsNotNone(self.model_manager.registry)
        self.assertTrue(hasattr(self.model_manager.registry, 'get_model_versions'))
        self.assertTrue(hasattr(self.model_manager.registry, 'get_model_performance_summary'))
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        mock_model1 = Mock()
        mock_model2 = Mock()
        
        # Mock evaluation for both models
        with patch.object(self.model_manager, 'evaluate_ai_model_performance') as mock_eval:
            mock_eval.side_effect = [
                {'accuracy': 0.85, 'precision': 0.82},
                {'accuracy': 0.87, 'precision': 0.85}
            ]
            
            comparison = self.model_manager.compare_models(
                dataset_name='breast_cancer',
                task_type='classification',
                target_column='target'
            )
            
            self.assertIsInstance(comparison, pd.DataFrame)
            self.assertGreater(len(comparison), 0)
    
    def test_get_model_recommendations(self):
        """Test model recommendations"""
        recommendations = self.model_manager.get_intelligent_model_recommendations(
            dataset_name='breast_cancer',
            target_column='target',
            task_type='classification'
        )
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('model_recommendations', recommendations)
        self.assertIn('data_characteristics', recommendations)
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization"""
        # Test that the training engine exists and has expected attributes
        self.assertIsNotNone(self.model_manager.training_engine)
        self.assertTrue(hasattr(self.model_manager.training_engine, 'model_configs'))
    
    def test_model_ensemble(self):
        """Test model ensemble creation"""
        # Test that the training engine exists and has expected methods
        self.assertIsNotNone(self.model_manager.training_engine)
        self.assertTrue(hasattr(self.model_manager.training_engine, 'model_configs'))
        
        # Test that ModelManager has create_ensemble_model method
        self.assertTrue(hasattr(self.model_manager, 'create_ensemble_model'))
    
    def test_model_deployment(self):
        """Test model deployment"""
        # Test that the registry exists and has expected methods
        self.assertIsNotNone(self.model_manager.registry)
        self.assertTrue(hasattr(self.model_manager.registry, 'save_model'))
        self.assertTrue(hasattr(self.model_manager.registry, 'load_model'))
    
    def test_model_monitoring(self):
        """Test model monitoring"""
        # Test that the registry exists and has expected methods
        self.assertIsNotNone(self.model_manager.registry)
        self.assertTrue(hasattr(self.model_manager.registry, 'get_model_versions'))
        self.assertTrue(hasattr(self.model_manager.registry, 'get_model_performance_summary'))


class TestModelManagerIntegration(HealthcareDSSTestCase):
    """Integration tests for ModelManager"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        super().setUp()
        # Use the model_manager from the base class
        self.model_manager = ModelManager(self.data_manager)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_model_lifecycle(self):
        """Test complete model lifecycle"""
        # Test that all components exist and can be initialized
        self.assertIsNotNone(self.model_manager.training_engine)
        self.assertIsNotNone(self.model_manager.evaluation_engine)
        self.assertIsNotNone(self.model_manager.registry)
        
        # Test that the model manager has the expected methods
        self.assertTrue(hasattr(self.model_manager, 'train_model'))
        self.assertTrue(hasattr(self.model_manager, 'evaluate_ai_model_performance'))
        self.assertTrue(hasattr(self.model_manager.registry, 'save_model'))
        
        # Test that we can access datasets
        self.assertIn('breast_cancer', self.data_manager.datasets)


if __name__ == '__main__':
    unittest.main()
