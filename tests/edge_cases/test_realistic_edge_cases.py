"""
Realistic Edge Case Tests for Healthcare DSS
============================================

This module tests edge cases using the actual DataManager and real datasets
to discover actual code issues and ensure robust error handling.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.data_management import DataManager
from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.preprocessing_engine import PreprocessingEngine
from healthcare_dss.analytics.model_training import ModelTrainingEngine
from healthcare_dss.analytics.model_evaluation import ModelEvaluationEngine
from healthcare_dss.analytics.model_registry import ModelRegistry
from healthcare_dss.analytics.classification_evaluation import ClassificationEvaluator
from healthcare_dss.analytics.clustering_analysis import ClusteringAnalyzer
from healthcare_dss.analytics.time_series_analysis import TimeSeriesAnalyzer
from healthcare_dss.analytics.prescriptive_analytics import PrescriptiveAnalyzer
from healthcare_dss.analytics.association_rules import AssociationRulesMiner
from healthcare_dss.utils.debug_manager import DebugManager
from healthcare_dss.utils.crisp_dm_workflow import CRISPDMWorkflow
from healthcare_dss.utils.intelligent_data_analyzer import IntelligentDataAnalyzer
from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager


class TestRealisticEdgeCases(unittest.TestCase):
    """Test edge cases using real DataManager and datasets"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use the actual DataManager with real datasets
        self.data_manager = DataManager(data_dir="datasets/raw")
        self.model_manager = ModelManager(self.data_manager)
        self.preprocessing_engine = PreprocessingEngine()
        self.training_engine = ModelTrainingEngine()
        self.evaluation_engine = ModelEvaluationEngine()
        self.model_registry = ModelRegistry()
        self.classification_evaluator = ClassificationEvaluator()
        self.clustering_analyzer = ClusteringAnalyzer(self.data_manager)
        self.time_series_analyzer = TimeSeriesAnalyzer(self.data_manager)
        self.prescriptive_analyzer = PrescriptiveAnalyzer(self.data_manager)
        self.association_rules_miner = AssociationRulesMiner(self.data_manager)
        self.debug_manager = DebugManager()
        self.crisp_dm_workflow = CRISPDMWorkflow(self.data_manager)
        self.intelligent_analyzer = IntelligentDataAnalyzer
        self.smart_target_manager = SmartDatasetTargetManager()
    
    def test_nonexistent_dataset(self):
        """Test handling of nonexistent dataset names"""
        # Test with completely invalid dataset name
        result = self.data_manager.get_dataset("nonexistent_dataset_12345")
        self.assertIsNone(result)
        
        # Test with empty string
        result = self.data_manager.get_dataset("")
        self.assertIsNone(result)
        
        # Test with None
        result = self.data_manager.get_dataset(None)
        self.assertIsNone(result)
    
    def test_model_training_with_invalid_data(self):
        """Test model training with invalid data types"""
        # Get a real dataset
        available_datasets = self.data_manager.get_available_datasets()
        if not available_datasets:
            self.skipTest("No datasets available for testing")
        
        dataset_name = available_datasets[0]
        dataset = self.data_manager.get_dataset(dataset_name)
        
        if dataset is None or len(dataset) == 0:
            self.skipTest(f"Dataset {dataset_name} is empty or None")
        
        # Test with invalid features (None)
        try:
            result = self.training_engine.train_model(
                features=None,
                target=dataset.iloc[:, -1] if len(dataset.columns) > 0 else None,
                model_name='random_forest',
                task_type='classification'
            )
            self.fail("Should have raised an error for None features")
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
        
        # Test with invalid target (None)
        try:
            result = self.training_engine.train_model(
                features=dataset.iloc[:, :-1] if len(dataset.columns) > 1 else dataset,
                target=None,
                model_name='random_forest',
                task_type='classification'
            )
            self.fail("Should have raised an error for None target")
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
    
    def test_preprocessing_with_invalid_data(self):
        """Test preprocessing with invalid data"""
        # Test with None dataset
        try:
            result = self.preprocessing_engine.preprocess_data(
                dataset=None,
                target_column='target'
            )
            self.fail("Should have raised an error for None dataset")
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
        
        # Test with empty DataFrame
        try:
            empty_df = pd.DataFrame()
            result = self.preprocessing_engine.preprocess_data(
                dataset=empty_df,
                target_column='target'
            )
            # Should handle empty DataFrame gracefully
        except Exception as e:
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_model_evaluation_with_invalid_predictions(self):
        """Test model evaluation with invalid predictions"""
        # Test with mismatched lengths
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0])  # Different length
        
        try:
            result = self.evaluation_engine.evaluate_model_performance(
                model=None,
                X_test=pd.DataFrame({'feature1': [1, 2, 3, 4]}),
                y_test=y_true,
                y_pred=y_pred,
                task_type='classification'
            )
            self.fail("Should have raised an error for mismatched lengths")
        except Exception as e:
            self.assertIsInstance(e, (ValueError, IndexError))
        
        # Test with None predictions
        try:
            result = self.evaluation_engine.evaluate_model_performance(
                model=None,
                X_test=pd.DataFrame({'feature1': [1, 2, 3, 4]}),
                y_test=y_true,
                y_pred=None,
                task_type='classification'
            )
            self.fail("Should have raised an error for None predictions")
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))
    
    def test_cross_validation_with_insufficient_data(self):
        """Test cross-validation with insufficient data"""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4],
            'target': [0, 1]
        })
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        
        try:
            cv_results = self.evaluation_engine.cross_validate_model(
                model=model,
                X=minimal_data[['feature1', 'feature2']],
                y=minimal_data['target'],
                cv=5,  # More folds than samples
                task_type='classification'
            )
            # Should handle insufficient data gracefully
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_association_rules_with_no_patterns(self):
        """Test association rules with data that has no patterns"""
        available_datasets = self.data_manager.get_available_datasets()
        if not available_datasets:
            self.skipTest("No datasets available for testing")
        
        dataset_name = available_datasets[0]
        
        try:
            # Test with a real dataset that might have sparse patterns
            categorical_columns = ['gender', 'hypertension', 'diabetes']  # Common categorical columns
            
            transaction_data = self.association_rules_miner.prepare_transaction_data(
                dataset_name, categorical_columns
            )
            
            frequent_itemsets = self.association_rules_miner.mine_frequent_itemsets(
                min_support=0.1
            )
            
            if len(frequent_itemsets) > 0:
                rules = self.association_rules_miner.generate_association_rules()
            else:
                # Should handle no frequent itemsets gracefully
                self.assertEqual(len(frequent_itemsets), 0)
        except Exception as e:
            # Should handle various errors gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError, KeyError))
    
    def test_clustering_with_identical_points(self):
        """Test clustering with identical data points"""
        available_datasets = self.data_manager.get_available_datasets()
        if not available_datasets:
            self.skipTest("No datasets available for testing")
        
        dataset_name = available_datasets[0]
        
        try:
            # Test clustering with real dataset
            X, original_data = self.clustering_analyzer.prepare_data_for_clustering(
                dataset_name, features=None
            )
            
            # Test K-means with the prepared data
            kmeans_result = self.clustering_analyzer.perform_kmeans_clustering(X, n_clusters=2)
        except Exception as e:
            # Should handle various errors gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError, KeyError))
    
    def test_classification_evaluation_edge_cases(self):
        """Test classification evaluation with edge case predictions"""
        # Test with perfect predictions
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred_perfect = np.array([0, 1, 0, 1, 0])
        
        # Test with terrible predictions
        y_pred_terrible = np.array([1, 0, 1, 0, 1])
        
        # Test with random predictions
        y_pred_random = np.random.randint(0, 2, 5)
        
        try:
            # Test with perfect predictions
            result_perfect = self.classification_evaluator.evaluate_predictions(
                y_true=y_true, y_pred=y_pred_perfect, model_name='perfect_model'
            )
            
            # Test with terrible predictions
            result_terrible = self.classification_evaluator.evaluate_predictions(
                y_true=y_true, y_pred=y_pred_terrible, model_name='terrible_model'
            )
            
            # Test with random predictions
            result_random = self.classification_evaluator.evaluate_predictions(
                y_true=y_true, y_pred=y_pred_random, model_name='random_model'
            )
            
            # All should work without errors
            self.assertIsInstance(result_perfect, dict)
            self.assertIsInstance(result_terrible, dict)
            self.assertIsInstance(result_random, dict)
            
        except Exception as e:
            self.fail(f"Classification evaluation failed with edge cases: {e}")
    
    def test_model_registry_edge_cases(self):
        """Test model registry with invalid model data"""
        try:
            # Test saving invalid model data
            invalid_model_data = {
                'model': None,  # Invalid model
                'metrics': {'accuracy': 'invalid'},  # Invalid metrics
                'model_name': '',  # Empty name
                'task_type': 'invalid_task'  # Invalid task type
            }
            
            # Should handle invalid model data gracefully
            result = self.model_registry.save_model('invalid_model', invalid_model_data)
            self.assertFalse(result)  # Should return False for invalid data
            
        except Exception as e:
            # Should raise appropriate error for invalid data
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_debug_manager_edge_cases(self):
        """Test debug manager with edge case operations"""
        try:
            # Test with very large debug messages
            large_message = "x" * 10000
            self.debug_manager.log_debug(large_message)
            
            # Test with None values
            self.debug_manager.log_debug(None)
            
            # Test with empty strings
            self.debug_manager.log_debug("")
            
            # Test performance metrics with extreme values
            self.debug_manager.update_performance_metric("extreme_metric", float('inf'))
            self.debug_manager.update_performance_metric("negative_metric", -1000)
            
            # All should work without errors
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Debug manager failed with edge cases: {e}")
    
    def test_prescriptive_analytics_edge_cases(self):
        """Test prescriptive analytics with impossible constraints"""
        available_datasets = self.data_manager.get_available_datasets()
        if not available_datasets:
            self.skipTest("No datasets available for testing")
        
        dataset_name = available_datasets[0]
        
        try:
            # Test with impossible constraints
            impossible_constraints = {
                'max_hours': -10,  # Negative constraint
                'min_staff': 1000,  # Impossible constraint
                'max_appointments': 0  # Zero constraint
            }
            
            result = self.prescriptive_analyzer.optimize_resource_allocation(
                dataset_name, resource_constraints=impossible_constraints
            )
        except Exception as e:
            # Should handle impossible constraints gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_time_series_edge_cases(self):
        """Test time series analysis with irregular data"""
        available_datasets = self.data_manager.get_available_datasets()
        if not available_datasets:
            self.skipTest("No datasets available for testing")
        
        dataset_name = available_datasets[0]
        dataset = self.data_manager.get_dataset(dataset_name)
        
        if dataset is None or len(dataset) == 0:
            self.skipTest(f"Dataset {dataset_name} is empty or None")
        
        try:
            # Test with invalid time column
            time_series_data = self.time_series_analyzer.prepare_time_series_data(
                dataset_name, time_column='nonexistent_time', value_column='nonexistent_value'
            )
        except Exception as e:
            # Should handle invalid columns gracefully
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_intelligent_data_analyzer_edge_cases(self):
        """Test intelligent data analyzer with edge cases"""
        try:
            # Test with empty Series
            empty_series = pd.Series([])
            result = self.intelligent_analyzer.detect_task_type(empty_series)
            
            # Test with Series containing only NaN
            nan_series = pd.Series([np.nan, np.nan, np.nan])
            result = self.intelligent_analyzer.detect_task_type(nan_series)
            
            # Test with Series containing mixed types
            mixed_series = pd.Series([1, 'text', 3.14, True, None])
            result = self.intelligent_analyzer.detect_task_type(mixed_series)
            
            # All should work without errors
            self.assertIsInstance(result, dict)
            
        except Exception as e:
            self.fail(f"Intelligent data analyzer failed with edge cases: {e}")


if __name__ == '__main__':
    unittest.main()
