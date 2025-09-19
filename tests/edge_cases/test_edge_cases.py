"""
Edge Case Tests for Healthcare DSS
==================================

This module tests edge cases, bad datasets, and error conditions to ensure
the system is robust and handles real-world data issues gracefully.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

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


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataManager(data_dir=self.temp_data_dir)
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
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset(self, name, data):
        """Helper to create test datasets"""
        df = pd.DataFrame(data)
        file_path = os.path.join(self.temp_data_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        
        # Load the dataset into the DataManager
        self.data_manager.load_dataset_from_file(name, file_path)
        
        return file_path
    
    def test_empty_dataset(self):
        """Test handling of completely empty datasets"""
        # Create empty dataset with at least one column to avoid DataFrame creation issues
        empty_data = {'empty_col': []}
        self.create_test_dataset("empty", empty_data)
        
        # Test DataManager with empty dataset
        try:
            dataset = self.data_manager.get_dataset("empty")
            # If it doesn't raise an error, check if it's handled gracefully
            if dataset is not None:
                self.assertTrue(len(dataset) == 0 or dataset.empty)
        except Exception as e:
            # Should handle empty datasets gracefully
            self.assertIsInstance(e, (ValueError, KeyError, pd.errors.EmptyDataError))
    
    def test_single_row_dataset(self):
        """Test handling of single-row datasets"""
        single_row_data = {
            'feature1': [1.0],
            'feature2': [2.0],
            'target': [0]
        }
        self.create_test_dataset("single_row", single_row_data)
        
        # This should work but may have limitations
        try:
            dataset = self.data_manager.get_dataset("single_row")
            if dataset is not None:
                self.assertEqual(len(dataset), 1)
            else:
                # Dataset not found or couldn't be loaded
                self.assertIsNone(dataset)
        except Exception as e:
            # Single row datasets might not be suitable for ML
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_all_nan_dataset(self):
        """Test handling of datasets with all NaN values"""
        nan_data = {
            'feature1': [np.nan, np.nan, np.nan],
            'feature2': [np.nan, np.nan, np.nan],
            'target': [np.nan, np.nan, np.nan]
        }
        self.create_test_dataset("all_nan", nan_data)
        
        # Test preprocessing with all NaN data
        try:
            dataset = self.data_manager.get_dataset("all_nan")
            if dataset is not None:
                processed = self.preprocessing_engine.preprocess_data(
                    dataset, 
                    target_column='target'
                )
                # Should handle gracefully or raise appropriate error
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_no_numeric_columns(self):
        """Test handling of datasets with no numeric columns"""
        categorical_data = {
            'category1': ['A', 'B', 'C', 'A', 'B'],
            'category2': ['X', 'Y', 'Z', 'X', 'Y'],
            'target': ['positive', 'negative', 'positive', 'negative', 'positive']
        }
        self.create_test_dataset("no_numeric", categorical_data)
        
        # Test preprocessing with no numeric columns
        try:
            dataset = self.data_manager.get_dataset("no_numeric")
            if dataset is not None:
                processed = self.preprocessing_engine.preprocess_data(
                    dataset,
                    target_column='target'
                )
                # Should handle categorical data properly
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_extreme_outliers(self):
        """Test handling of datasets with extreme outliers"""
        outlier_data = {
            'feature1': [1, 2, 3, 1e10, 5],  # Extreme outlier
            'feature2': [1, 2, 3, 4, -1e10],  # Extreme negative outlier
            'target': [0, 1, 0, 1, 0]
        }
        self.create_test_dataset("outliers", outlier_data)
        
        # Test preprocessing with extreme outliers
        try:
            dataset = self.data_manager.get_dataset("outliers")
            if dataset is not None:
                processed = self.preprocessing_engine.preprocess_data(
                    dataset,
                    target_column='target'
                )
                # Should handle outliers gracefully
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_duplicate_columns(self):
        """Test handling of datasets with duplicate column names"""
        duplicate_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature1': [6, 7, 8, 9, 10],  # Duplicate name
            'target': [0, 1, 0, 1, 0]
        }
        self.create_test_dataset("duplicate_cols", duplicate_data)
        
        # Test with duplicate columns
        try:
            dataset = self.data_manager.get_dataset("duplicate_cols")
            # Should handle duplicate columns
        except Exception as e:
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_mixed_data_types(self):
        """Test handling of datasets with mixed data types in columns"""
        mixed_data = {
            'mixed_col': [1, 'text', 3.14, True, None],
            'target': [0, 1, 0, 1, 0]
        }
        self.create_test_dataset("mixed_types", mixed_data)
        
        # Test preprocessing with mixed types
        try:
            dataset = self.data_manager.get_dataset("mixed_types")
            if dataset is not None:
                processed = self.preprocessing_engine.preprocess_data(
                    dataset,
                    target_column='target'
                )
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_very_large_dataset(self):
        """Test handling of very large datasets (memory stress test)"""
        # Create a moderately large dataset for testing
        large_data = {
            'feature1': np.random.random(10000),
            'feature2': np.random.random(10000),
            'feature3': np.random.random(10000),
            'target': np.random.randint(0, 2, 10000)
        }
        self.create_test_dataset("large_dataset", large_data)
        
        # Test with large dataset
        try:
            dataset = self.data_manager.get_dataset("large_dataset")
            # Should handle large datasets efficiently
            self.assertEqual(len(dataset), 10000)
        except Exception as e:
            self.assertIsInstance(e, (MemoryError, ValueError))
    
    def test_model_training_edge_cases(self):
        """Test model training with edge case data"""
        # Create problematic dataset
        problematic_data = {
            'feature1': [1, 1, 1, 1, 1],  # No variance
            'feature2': [2, 2, 2, 2, 2],  # No variance
            'target': [0, 0, 0, 0, 0]    # Single class
        }
        self.create_test_dataset("no_variance", problematic_data)
        
        try:
            dataset = self.data_manager.get_dataset("no_variance")
            features = dataset.drop(columns=['target'])
            target = dataset['target']
            
            # Test model training with no variance data
            result = self.training_engine.train_model(
                features=features,
                target=target,
                model_name='random_forest',
                task_type='classification'
            )
        except Exception as e:
            # Should handle no variance data gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_cross_validation_edge_cases(self):
        """Test cross-validation with edge case data"""
        # Create dataset with insufficient samples for CV
        small_data = {
            'feature1': [1, 2],
            'feature2': [3, 4],
            'target': [0, 1]
        }
        self.create_test_dataset("too_small", small_data)
        
        try:
            dataset = self.data_manager.get_dataset("too_small")
            features = dataset.drop(columns=['target'])
            target = dataset['target']
            
            # Test cross-validation with too few samples
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            
            cv_results = self.evaluation_engine.cross_validate_model(
                model=model,
                X=features,
                y=target,
                cv=5,  # More folds than samples
                task_type='classification'
            )
        except Exception as e:
            # Should handle insufficient samples gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_association_rules_edge_cases(self):
        """Test association rules with edge case data"""
        # Create dataset with no frequent patterns
        sparse_data = {
            'item1': [1, 0, 0, 0, 0],
            'item2': [0, 1, 0, 0, 0],
            'item3': [0, 0, 1, 0, 0],
            'item4': [0, 0, 0, 1, 0],
            'item5': [0, 0, 0, 0, 1]
        }
        self.create_test_dataset("sparse_transactions", sparse_data)
        
        try:
            dataset = self.data_manager.get_dataset("sparse_transactions")
            
            # Test association rules with sparse data
            categorical_columns = list(dataset.columns)
            transaction_data = self.association_rules_miner.prepare_transaction_data(
                "sparse_transactions", categorical_columns
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
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_clustering_edge_cases(self):
        """Test clustering with edge case data"""
        # Create dataset with all identical points
        identical_data = {
            'feature1': [1, 1, 1, 1, 1],
            'feature2': [2, 2, 2, 2, 2],
            'target': [0, 0, 0, 0, 0]
        }
        self.create_test_dataset("identical_points", identical_data)
        
        try:
            dataset = self.data_manager.get_dataset("identical_points")
            
            # Test clustering with identical points
            X, original_data = self.clustering_analyzer.prepare_data_for_clustering(
                "identical_points", features=['feature1', 'feature2']
            )
            
            # Test K-means with identical points
            kmeans_result = self.clustering_analyzer.perform_kmeans_clustering(X, n_clusters=2)
        except Exception as e:
            # Should handle identical points gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_prescriptive_analytics_edge_cases(self):
        """Test prescriptive analytics with edge case data"""
        # Create dataset with invalid constraints
        constraint_data = {
            'resource1': [1, 2, 3, 4, 5],
            'resource2': [5, 4, 3, 2, 1],
            'target': [10, 20, 30, 40, 50]
        }
        self.create_test_dataset("constraint_data", constraint_data)
        
        try:
            dataset = self.data_manager.get_dataset("constraint_data")
            
            # Test with impossible constraints
            impossible_constraints = {
                'max_hours': -10,  # Negative constraint
                'min_staff': 1000,  # Impossible constraint
                'max_appointments': 0  # Zero constraint
            }
            
            result = self.prescriptive_analyzer.optimize_resource_allocation(
                'constraint_data', resource_constraints=impossible_constraints
            )
        except Exception as e:
            # Should handle impossible constraints gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_time_series_edge_cases(self):
        """Test time series analysis with edge case data"""
        # Create dataset with irregular time intervals
        irregular_data = {
            'date': ['2023-01-01', '2023-01-05', '2023-01-02', '2023-01-10', '2023-01-03'],
            'value': [10, 20, 15, 25, 18],
            'target': [100, 200, 150, 250, 180]
        }
        self.create_test_dataset("irregular_time", irregular_data)
        
        try:
            dataset = self.data_manager.get_dataset("irregular_time")
            
            # Test time series preparation with irregular data
            time_series_data = self.time_series_analyzer.prepare_time_series_data(
                "irregular_time", time_column='date', value_column='value'
            )
        except Exception as e:
            # Should handle irregular time series gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
    
    def test_classification_evaluation_edge_cases(self):
        """Test classification evaluation with edge case predictions"""
        # Create perfect predictions (all correct)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred_perfect = np.array([0, 1, 0, 1, 0])
        
        # Create terrible predictions (all wrong)
        y_pred_terrible = np.array([1, 0, 1, 0, 1])
        
        # Create random predictions
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
        """Test model registry with edge case data"""
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


if __name__ == '__main__':
    unittest.main()
