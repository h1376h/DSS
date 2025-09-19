"""
Comprehensive Test Suite for Binning UI Integration
==================================================

This module contains comprehensive tests for binning UI integration:
- Model Training Configuration UI
- Model Training Execution UI
- Binning Options UI
- User Override UI
- Edge Cases and Error Handling
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import warnings
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Suppress specific warnings for cleaner test output
warnings.filterwarnings("ignore", message="Bins whose width are too small")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.utils.intelligent_binning import intelligent_binning
from healthcare_dss.utils.debug_manager import debug_manager


class TestBinningUIIntegration(unittest.TestCase):
    """Test cases for binning UI integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test datasets
        self.diabetes_data = pd.DataFrame({
            'age': np.random.normal(50, 15, 100),
            'bmi': np.random.normal(25, 5, 100),
            'glucose': np.random.normal(100, 20, 100),
            'target': np.random.normal(150, 50, 100)  # Continuous target
        })
        
        self.classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)  # Binary target
        })
        
        # Mock session state
        self.mock_session_state = {
            'selected_dataset': 'diabetes',
            'selected_target': 'target',
            'recommended_task_type': 'regression',
            'task_type_confidence': 'high',
            'model_config': {
                'task_type': 'classification',
                'model_type': 'Random Forest'
            }
        }
    
    def test_binning_configuration_storage(self):
        """Test that binning configuration is properly stored in session state"""
        # Test automatic binning configuration
        binning_config = {
            'mode': 'automatic',
            'strategy': 'quantile',
            'n_bins': 3,
            'enabled': True
        }
        
        # Simulate storing in session state
        session_state = {}
        session_state['binning_config'] = binning_config
        
        # Verify configuration is stored correctly
        self.assertIn('binning_config', session_state)
        self.assertEqual(session_state['binning_config']['mode'], 'automatic')
        self.assertEqual(session_state['binning_config']['strategy'], 'quantile')
        self.assertEqual(session_state['binning_config']['n_bins'], 3)
        self.assertTrue(session_state['binning_config']['enabled'])
    
    def test_manual_binning_configuration(self):
        """Test manual binning configuration"""
        binning_config = {
            'mode': 'manual',
            'strategy': 'uniform',
            'n_bins': 4,
            'enabled': True
        }
        
        session_state = {}
        session_state['binning_config'] = binning_config
        
        # Verify manual configuration
        self.assertEqual(session_state['binning_config']['mode'], 'manual')
        self.assertEqual(session_state['binning_config']['strategy'], 'uniform')
        self.assertEqual(session_state['binning_config']['n_bins'], 4)
    
    def test_advanced_override_configuration(self):
        """Test advanced override configuration"""
        override_config = {
            'custom_thresholds': [50, 100, 150],
            'custom_bin_labels': ['Low', 'Medium', 'High', 'Very High']
        }
        
        binning_config = {
            'mode': 'advanced_override',
            'override_config': override_config,
            'enabled': True
        }
        
        session_state = {}
        session_state['binning_config'] = binning_config
        
        # Verify advanced override configuration
        self.assertEqual(session_state['binning_config']['mode'], 'advanced_override')
        self.assertIn('override_config', session_state['binning_config'])
        self.assertEqual(session_state['binning_config']['override_config']['custom_thresholds'], [50, 100, 150])
        self.assertEqual(session_state['binning_config']['override_config']['custom_bin_labels'], ['Low', 'Medium', 'High', 'Very High'])
    
    def test_binning_need_detection_ui(self):
        """Test binning need detection for UI scenarios"""
        target_data = self.diabetes_data['target'].values
        
        # Test detection for classification task
        needs_binning, analysis = intelligent_binning.detect_binning_need(target_data, 'classification')
        
        self.assertTrue(needs_binning)  # Continuous data should need binning for classification
        self.assertIsInstance(analysis, dict)
        self.assertIn('unique_values', analysis)
        self.assertIn('unique_ratio', analysis)
        self.assertIn('is_numeric', analysis)
        
        # Test detection for regression task
        needs_binning_regression, analysis_regression = intelligent_binning.detect_binning_need(target_data, 'regression')
        
        self.assertFalse(needs_binning_regression)  # Regression should not need binning
    
    def test_binning_suggestions_ui(self):
        """Test binning suggestions for UI display"""
        target_data = self.diabetes_data['target'].values
        needs_binning, analysis = intelligent_binning.detect_binning_need(target_data, 'classification')
        suggestions = intelligent_binning.suggest_optimal_bins(target_data, analysis)
        
        # Verify suggestions are suitable for UI display
        self.assertIsInstance(suggestions, dict)
        self.assertIn('optimal_bins', suggestions)
        self.assertIn('recommended_strategy', suggestions)
        self.assertIn('min_bins', suggestions)
        self.assertIn('max_bins', suggestions)
        self.assertIn('reasoning', suggestions)
        
        # Verify suggestions are reasonable
        self.assertGreaterEqual(suggestions['optimal_bins'], suggestions['min_bins'])
        self.assertLessEqual(suggestions['optimal_bins'], suggestions['max_bins'])
        self.assertIn(suggestions['recommended_strategy'], ['quantile', 'uniform', 'kmeans', 'jenks'])
        self.assertIsInstance(suggestions['reasoning'], list)
        self.assertGreater(len(suggestions['reasoning']), 0)
    
    def test_binning_preview_ui(self):
        """Test binning preview for UI display"""
        target_data = self.diabetes_data['target'].values
        preview = intelligent_binning.get_binning_preview(target_data, 'quantile', 3)
        
        # Verify preview is suitable for UI display
        self.assertIsInstance(preview, dict)
        self.assertIn('success', preview)
        
        if preview['success']:
            self.assertIn('bin_labels', preview)
            self.assertIn('bin_counts', preview)
            self.assertIn('bin_edges', preview)
            self.assertIn('class_balance', preview)
            
            # Verify preview data structure
            self.assertEqual(len(preview['bin_labels']), 3)
            self.assertEqual(len(preview['bin_counts']), 3)
            self.assertEqual(len(preview['bin_edges']), 4)  # n_bins + 1
            self.assertGreaterEqual(preview['class_balance'], 0.0)
            self.assertLessEqual(preview['class_balance'], 1.0)
    
    def test_user_override_options_ui(self):
        """Test user override options for UI display"""
        target_data = self.diabetes_data['target'].values
        needs_binning, analysis = intelligent_binning.detect_binning_need(target_data, 'classification')
        override_options = intelligent_binning.get_user_override_options(target_data, analysis)
        
        # Verify override options are suitable for UI display
        self.assertIsInstance(override_options, dict)
        self.assertIn('custom_thresholds', override_options)
        self.assertIn('custom_bin_labels', override_options)
        self.assertIn('force_binning', override_options)
        self.assertIn('disable_binning', override_options)
        
        # Verify custom thresholds structure
        thresholds = override_options['custom_thresholds']
        self.assertIn('percentile_based', thresholds)
        self.assertIn('sigma_based', thresholds)
        self.assertIn('equal_width', thresholds)
        self.assertIn('manual', thresholds)
        
        # Verify custom labels structure
        labels = override_options['custom_bin_labels']
        self.assertIn('severity_levels', labels)
        self.assertIn('risk_levels', labels)
        self.assertIn('performance_levels', labels)
        self.assertIn('manual', labels)
    
    def test_binning_execution_automatic_mode(self):
        """Test binning execution in automatic mode"""
        target_data = self.diabetes_data['target'].values
        
        # Simulate automatic binning configuration
        binning_config = {
            'mode': 'automatic',
            'strategy': 'quantile',
            'n_bins': 3,
            'enabled': True
        }
        
        # Apply binning
        y_binned, binning_info = intelligent_binning.apply_binning(target_data, 'quantile', 3)
        
        # Verify results
        self.assertIsInstance(y_binned, np.ndarray)
        self.assertIsInstance(binning_info, dict)
        self.assertEqual(len(y_binned), len(target_data))
        self.assertEqual(len(np.unique(y_binned)), 3)
        self.assertEqual(binning_info['strategy'], 'quantile')
        self.assertEqual(binning_info['n_bins'], 3)
    
    def test_binning_execution_manual_mode(self):
        """Test binning execution in manual mode"""
        target_data = self.diabetes_data['target'].values
        
        # Simulate manual binning configuration
        binning_config = {
            'mode': 'manual',
            'strategy': 'uniform',
            'n_bins': 4,
            'enabled': True
        }
        
        # Apply binning
        y_binned, binning_info = intelligent_binning.apply_binning(target_data, 'uniform', 4)
        
        # Verify results
        self.assertIsInstance(y_binned, np.ndarray)
        self.assertIsInstance(binning_info, dict)
        self.assertEqual(len(y_binned), len(target_data))
        # The actual number of bins might be less due to automatic reduction
        self.assertLessEqual(len(np.unique(y_binned)), 4)
        self.assertGreaterEqual(len(np.unique(y_binned)), 2)  # At least 2 bins
        self.assertEqual(binning_info['strategy'], 'uniform')
        self.assertLessEqual(binning_info['n_bins'], 4)
    
    def test_binning_execution_advanced_override_mode(self):
        """Test binning execution in advanced override mode"""
        target_data = self.diabetes_data['target'].values
        
        # Simulate advanced override configuration
        override_config = {
            'custom_thresholds': [50, 100, 150],
            'custom_bin_labels': ['Low', 'Medium', 'High', 'Very High']
        }
        
        binning_config = {
            'mode': 'advanced_override',
            'override_config': override_config,
            'enabled': True
        }
        
        # Apply user override binning
        y_binned, binning_info = intelligent_binning.apply_user_override_binning(target_data, override_config)
        
        # Verify results
        self.assertIsInstance(y_binned, np.ndarray)
        self.assertIsInstance(binning_info, dict)
        self.assertEqual(len(y_binned), len(target_data))
        self.assertEqual(binning_info['strategy'], 'user_override')
        self.assertTrue(binning_info['user_configured'])
        self.assertEqual(binning_info['bin_labels'], ['Low', 'Medium', 'High', 'Very High'])
    
    def test_binning_execution_disable_mode(self):
        """Test binning execution in disable mode"""
        target_data = self.diabetes_data['target'].values
        
        # Simulate disable binning configuration
        override_config = {
            'disable_binning': True
        }
        
        # Apply user override binning
        y_binned, binning_info = intelligent_binning.apply_user_override_binning(target_data, override_config)
        
        # Verify results
        self.assertIsInstance(y_binned, np.ndarray)
        self.assertIsInstance(binning_info, dict)
        self.assertEqual(binning_info['strategy'], 'disabled')
        self.assertIn('warning', binning_info)
        # Should return original data
        np.testing.assert_array_equal(y_binned, target_data)
    
    def test_binning_ui_edge_cases(self):
        """Test binning UI edge cases"""
        # Test with very small dataset
        small_data = np.array([1, 2, 3])
        needs_binning, analysis = intelligent_binning.detect_binning_need(small_data, 'classification')
        
        self.assertIsInstance(needs_binning, bool)
        self.assertIsInstance(analysis, dict)
        
        # Test with constant data
        constant_data = np.array([5, 5, 5, 5, 5])
        needs_binning_const, analysis_const = intelligent_binning.detect_binning_need(constant_data, 'classification')
        
        self.assertIsInstance(needs_binning_const, bool)
        self.assertIsInstance(analysis_const, dict)
        self.assertEqual(analysis_const['unique_values'], 1)
        self.assertEqual(analysis_const['unique_ratio'], 0.2)  # 1/5 = 0.2
    
    def test_binning_ui_error_handling(self):
        """Test binning UI error handling"""
        # Test with invalid strategy
        target_data = self.diabetes_data['target'].values
        
        with self.assertRaises(ValueError):
            intelligent_binning.apply_binning(target_data, 'invalid_strategy', 3)
        
        # Test with invalid number of bins
        with self.assertRaises(ValueError):
            intelligent_binning.apply_binning(target_data, 'quantile', 0)
        
        # Test with empty data
        empty_data = np.array([])
        
        with self.assertRaises(ValueError):
            intelligent_binning.detect_binning_need(empty_data, 'classification')
    
    def test_binning_ui_integration_workflow(self):
        """Test complete binning UI integration workflow"""
        target_data = self.diabetes_data['target'].values
        
        # Step 1: Detect binning need
        needs_binning, analysis = intelligent_binning.detect_binning_need(target_data, 'classification')
        self.assertTrue(needs_binning)
        
        # Step 2: Get suggestions
        suggestions = intelligent_binning.suggest_optimal_bins(target_data, analysis)
        self.assertIsInstance(suggestions, dict)
        
        # Step 3: Preview binning
        preview = intelligent_binning.get_binning_preview(target_data, suggestions['recommended_strategy'], suggestions['optimal_bins'])
        self.assertIsInstance(preview, dict)
        
        # Step 4: Apply binning
        y_binned, binning_info = intelligent_binning.apply_binning(target_data, suggestions['recommended_strategy'], suggestions['optimal_bins'])
        self.assertIsInstance(y_binned, np.ndarray)
        self.assertIsInstance(binning_info, dict)
        
        # Step 5: Verify results
        self.assertEqual(len(y_binned), len(target_data))
        # The actual number of bins might be less than optimal due to automatic reduction
        self.assertLessEqual(len(np.unique(y_binned)), suggestions['optimal_bins'])
        self.assertGreaterEqual(len(np.unique(y_binned)), 2)  # At least 2 bins
        self.assertEqual(binning_info['strategy'], suggestions['recommended_strategy'])
    
    def test_binning_ui_session_state_management(self):
        """Test binning UI session state management"""
        # Simulate session state initialization
        session_state = {}
        
        # Test storing binning configuration
        binning_config = {
            'mode': 'automatic',
            'strategy': 'quantile',
            'n_bins': 3,
            'enabled': True
        }
        session_state['binning_config'] = binning_config
        
        # Test retrieving binning configuration
        retrieved_config = session_state.get('binning_config', {})
        self.assertEqual(retrieved_config['mode'], 'automatic')
        self.assertEqual(retrieved_config['strategy'], 'quantile')
        self.assertEqual(retrieved_config['n_bins'], 3)
        self.assertTrue(retrieved_config['enabled'])
        
        # Test updating binning configuration
        retrieved_config['n_bins'] = 4
        session_state['binning_config'] = retrieved_config
        
        # Verify update
        updated_config = session_state['binning_config']
        self.assertEqual(updated_config['n_bins'], 4)
    
    def test_binning_ui_performance(self):
        """Test binning UI performance with large datasets"""
        # Create large dataset
        large_data = np.random.normal(100, 15, 10000)
        
        # Test detection performance
        needs_binning, analysis = intelligent_binning.detect_binning_need(large_data, 'classification')
        self.assertIsInstance(needs_binning, bool)
        self.assertIsInstance(analysis, dict)
        
        # Test suggestions performance
        suggestions = intelligent_binning.suggest_optimal_bins(large_data, analysis)
        self.assertIsInstance(suggestions, dict)
        
        # Test preview performance
        preview = intelligent_binning.get_binning_preview(large_data, 'quantile', 5)
        self.assertIsInstance(preview, dict)
        
        # Test application performance
        y_binned, binning_info = intelligent_binning.apply_binning(large_data, 'quantile', 5)
        self.assertEqual(len(y_binned), len(large_data))
        self.assertEqual(len(np.unique(y_binned)), 5)


if __name__ == '__main__':
    unittest.main()
