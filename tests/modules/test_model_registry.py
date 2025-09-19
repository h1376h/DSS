"""
Test Model Registry
===================

Comprehensive tests for the ModelRegistry module
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.analytics.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(models_dir=self.temp_dir)
        
        # Create test model data with real model
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple real model for testing
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        self.test_model_data = {
            'model': model,  # Real model
            'model_name': 'test_model',
            'task_type': 'classification',
            'dataset_name': 'test_dataset',
            'target_column': 'target',
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1_score': 0.81
            },
            'training_data': {
                'X_train': X_train,
                'X_test': pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10]}),
                'y_train': y_train,
                'y_test': pd.Series([1, 0]),
                'y_pred': np.array([1, 0])
            },
            'preprocessing_config': {'scaling_method': 'standard_scaler'},
            'timestamp': '2024-01-01T00:00:00'
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        self.assertIsInstance(self.registry.models_dir, Path)
        self.assertTrue(self.registry.models_dir.exists())
        self.assertIsInstance(self.registry.model_cache, dict)
    
    def test_save_model(self):
        """Test model saving"""
        model_key = 'test_model_001'
        
        result = self.registry.save_model(model_key, self.test_model_data)
        
        self.assertTrue(result)
        
        # Check that files were created
        model_file = self.registry.models_dir / f"{model_key}_model.joblib"
        metadata_file = self.registry.models_dir / f"{model_key}_metadata.json"
        
        self.assertTrue(model_file.exists())
        self.assertTrue(metadata_file.exists())
        
        # Check that model is cached
        self.assertIn(model_key, self.registry.model_cache)
    
    def test_load_model(self):
        """Test model loading"""
        model_key = 'test_model_002'
        
        # First save a model
        self.registry.save_model(model_key, self.test_model_data)
        
        # Then load it
        loaded_data = self.registry.load_model(model_key)
        
        self.assertIsNotNone(loaded_data)
        self.assertIn('model', loaded_data)
        self.assertIn('metadata', loaded_data)
        self.assertIn('metrics', loaded_data)
        self.assertIn('feature_names', loaded_data)
        
        # Check that metrics are preserved
        self.assertEqual(loaded_data['metrics']['accuracy'], 0.85)
    
    def test_load_model_from_cache(self):
        """Test model loading from cache"""
        model_key = 'test_model_003'
        
        # Save and load model (should be cached)
        self.registry.save_model(model_key, self.test_model_data)
        loaded_data = self.registry.load_model(model_key)
        
        # Load again (should come from cache)
        cached_data = self.registry.load_model(model_key)
        
        self.assertIsNotNone(cached_data)
        self.assertEqual(loaded_data['metrics'], cached_data['metrics'])
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model"""
        loaded_data = self.registry.load_model('nonexistent_model')
        
        self.assertIsNone(loaded_data)
    
    def test_list_models(self):
        """Test listing models"""
        # Save multiple models
        for i in range(3):
            model_key = f'test_model_{i:03d}'
            self.registry.save_model(model_key, self.test_model_data)
        
        # List models
        models_df = self.registry.list_models()
        
        self.assertIsInstance(models_df, pd.DataFrame)
        self.assertGreaterEqual(len(models_df), 3)
        
        # Check expected columns
        expected_columns = ['model_key', 'model_name', 'task_type', 'created_at']
        for col in expected_columns:
            self.assertIn(col, models_df.columns)
    
    def test_list_models_with_status_filter(self):
        """Test listing models with status filter"""
        # Save a model
        model_key = 'test_model_status'
        self.registry.save_model(model_key, self.test_model_data)
        
        # List active models
        active_models = self.registry.list_models(status='active')
        self.assertIsInstance(active_models, pd.DataFrame)
        
        # List all models
        all_models = self.registry.list_models(status='all')
        self.assertIsInstance(all_models, pd.DataFrame)
    
    def test_get_model_performance_summary(self):
        """Test getting model performance summary"""
        # Save a model
        model_key = 'test_model_performance'
        self.registry.save_model(model_key, self.test_model_data)
        
        # Get performance summary
        summary_df = self.registry.get_model_performance_summary()
        
        self.assertIsInstance(summary_df, pd.DataFrame)
        
        if len(summary_df) > 0:
            # Check expected columns
            expected_columns = ['Model Key', 'Model Name', 'Task Type', 'Accuracy']
            for col in expected_columns:
                self.assertIn(col, summary_df.columns)
    
    def test_delete_model(self):
        """Test model deletion"""
        model_key = 'test_model_delete'
        
        # Save a model
        self.registry.save_model(model_key, self.test_model_data)
        
        # Verify it exists
        loaded_data = self.registry.load_model(model_key)
        self.assertIsNotNone(loaded_data)
        
        # Delete the model
        result = self.registry.delete_model(model_key)
        self.assertTrue(result)
        
        # Verify it's deleted
        loaded_data = self.registry.load_model(model_key)
        self.assertIsNone(loaded_data)
        
        # Check that it's removed from cache
        self.assertNotIn(model_key, self.registry.model_cache)
    
    def test_delete_nonexistent_model(self):
        """Test deleting non-existent model"""
        result = self.registry.delete_model('nonexistent_model')
        self.assertFalse(result)
    
    def test_update_model_status(self):
        """Test updating model status"""
        model_key = 'test_model_status_update'
        
        # Save a model
        self.registry.save_model(model_key, self.test_model_data)
        
        # Update status
        result = self.registry.update_model_status(model_key, 'inactive')
        self.assertTrue(result)
        
        # Verify status was updated
        models_df = self.registry.list_models(status='active')
        if len(models_df) > 0:
            self.assertNotIn(model_key, models_df['model_key'].values)
    
    def test_get_model_versions(self):
        """Test getting model versions"""
        model_key = 'test_model_versions'
        
        # Save a model
        self.registry.save_model(model_key, self.test_model_data)
        
        # Get versions
        versions_df = self.registry.get_model_versions(model_key)
        
        self.assertIsInstance(versions_df, pd.DataFrame)
    
    def test_clear_cache(self):
        """Test clearing model cache"""
        model_key = 'test_model_cache'
        
        # Save a model (should be cached)
        self.registry.save_model(model_key, self.test_model_data)
        
        # Verify it's in cache
        self.assertIn(model_key, self.registry.model_cache)
        
        # Clear cache
        self.registry.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.registry.model_cache), 0)
    
    def test_get_cache_info(self):
        """Test getting cache information"""
        model_key = 'test_model_cache_info'
        
        # Save a model
        self.registry.save_model(model_key, self.test_model_data)
        
        # Get cache info
        cache_info = self.registry.get_cache_info()
        
        self.assertIn('cached_models', cache_info)
        self.assertIn('cache_size', cache_info)
        self.assertIn('cache_details', cache_info)
        
        self.assertIn(model_key, cache_info['cached_models'])
        self.assertGreater(cache_info['cache_size'], 0)
    
    def test_error_handling_save_model(self):
        """Test error handling in save_model"""
        # Test with invalid model data
        invalid_data = {'invalid': 'data'}
        
        result = self.registry.save_model('invalid_model', invalid_data)
        
        # Should handle gracefully
        self.assertIsInstance(result, bool)
    
    def test_database_operations(self):
        """Test database operations"""
        # Test that database is properly initialized
        self.assertTrue(os.path.exists(self.registry.registry_db))
        
        # Test that we can query the database
        models_df = self.registry.list_models()
        self.assertIsInstance(models_df, pd.DataFrame)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
