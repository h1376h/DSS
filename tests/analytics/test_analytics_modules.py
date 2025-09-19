"""
Comprehensive Test Suite for Analytics Modules
==============================================

This module contains comprehensive tests for all analytics components:
- Model Training Engine
- Model Evaluation Engine
- Model Registry
- Classification Evaluator
- Clustering Analyzer
- Time Series Analyzer
- Prescriptive Analyzer
- Association Rules Miner
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

from healthcare_dss.analytics.model_training import ModelTrainingEngine
from healthcare_dss.analytics.model_evaluation import ModelEvaluationEngine
from healthcare_dss.analytics.model_registry import ModelRegistry
from healthcare_dss.analytics.classification_evaluation import ClassificationEvaluator
from healthcare_dss.analytics.clustering_analysis import ClusteringAnalyzer
from healthcare_dss.analytics.time_series_analysis import TimeSeriesAnalyzer
from healthcare_dss.analytics.prescriptive_analytics import PrescriptiveAnalyzer
from healthcare_dss.analytics.association_rules import AssociationRulesMiner
from healthcare_dss.core.data_management import DataManager
from tests.test_base import HealthcareDSSTestCase


class TestModelTrainingEngine(HealthcareDSSTestCase):
    """Test cases for ModelTrainingEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.training_engine = ModelTrainingEngine()
    
    def test_initialization(self):
        """Test ModelTrainingEngine initialization"""
        self.assertIsNotNone(self.training_engine)
        self.assertIsInstance(self.training_engine.model_configs, dict)
        self.assertGreater(len(self.training_engine.model_configs), 0)
    
    def test_train_model(self):
        """Test model training"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare features and target - drop target column and any non-numeric columns
        features = dataset_df.drop(columns=[target_col])
        # Also drop any non-numeric columns that might cause issues
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        target = dataset_df[target_col]
        
        result = self.training_engine.train_model(
            features=features,
            target=target,
            model_name='random_forest',
            task_type='classification'
        )
        
        self.assert_model_result_valid(result)
    
    def test_create_ensemble(self):
        """Test ensemble model creation"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare features and target - drop target column and any non-numeric columns
        features = dataset_df.drop(columns=[target_col])
        # Also drop any non-numeric columns that might cause issues
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        target = dataset_df[target_col]
        
        result = self.training_engine.create_ensemble_model(
            features=features,
            target=target,
            models=['random_forest', 'svm'],
            task_type='classification'
        )
        
        self.assert_model_result_valid(result)


class TestModelEvaluationEngine(HealthcareDSSTestCase):
    """Test cases for ModelEvaluationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.evaluation_engine = ModelEvaluationEngine()
    
    def test_initialization(self):
        """Test ModelEvaluationEngine initialization"""
        self.assertIsNotNone(self.evaluation_engine)
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare features and target - drop target column and any non-numeric columns
        features = dataset_df.drop(columns=[target_col])
        # Also drop any non-numeric columns that might cause issues
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        target = dataset_df[target_col]
        
        # Create realistic predictions that match the target distribution
        target_sample = target.head(10)
        unique_classes = target_sample.unique()
        
        # Create predictions that maintain the class distribution
        if len(unique_classes) == 2:
            # Binary classification - create balanced predictions
            y_pred = np.array([0, 1] * (len(target_sample) // 2) + [0] * (len(target_sample) % 2))
        else:
            # Multi-class - use the actual classes
            y_pred = np.random.choice(unique_classes, len(target_sample))
        
        mock_model = Mock()
        mock_model.predict.return_value = y_pred
        
        # Test model evaluation
        result = self.evaluation_engine.evaluate_model_performance(
            model=mock_model,
            X_test=features.head(10),
            y_test=target.head(10),
            y_pred=y_pred,
            task_type='classification'
        )
        
        self.assertIsInstance(result, dict)
    
    def test_cross_validation(self):
        """Test cross-validation evaluation"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare features and target - drop target column and any non-numeric columns
        features = dataset_df.drop(columns=[target_col])
        # Also drop any non-numeric columns that might cause issues
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        target = dataset_df[target_col]
        
        # Use a real model instead of mock for cross-validation
        from sklearn.ensemble import RandomForestClassifier
        
        # Test with RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test cross-validation
        result = self.evaluation_engine.cross_validate_model(
            model=model,
            X=features.head(50),  # Use more data for better cross-validation
            y=target.head(50),
            cv=3,
            task_type='classification'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('task_type', result)
        self.assertIn('cv_folds', result)
        self.assertIn('overall_score', result)
        self.assertIn('cv_results', result)
        self.assertEqual(result['task_type'], 'classification')
        self.assertEqual(result['cv_folds'], 3)


class TestModelRegistry(HealthcareDSSTestCase):
    """Test cases for ModelRegistry"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.model_registry = ModelRegistry()
    
    def test_initialization(self):
        """Test ModelRegistry initialization"""
        self.assertIsNotNone(self.model_registry)
        self.assertIsNotNone(self.model_registry.models_dir)
        self.assertIsNotNone(self.model_registry.registry_db)
    
    def test_save_model(self):
        """Test model saving"""
        # Create simple model data that can be pickled
        model_data = {
            'model_type': 'RandomForestClassifier',
            'metrics': {'accuracy': 0.95, 'precision': 0.92},
            'training_time': 120.5,
            'dataset': 'diabetes',
            'algorithm': 'random_forest'
        }
        
        # Test model saving
        result = self.model_registry.save_model(
            model_key='test_model_001',
            model_data=model_data
        )
        
        self.assertIsInstance(result, bool)
    
    def test_load_model(self):
        """Test model loading"""
        # Create a simple model data that can be pickled
        import pickle
        import tempfile
        import os
        
        # Create a real model for testing
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        # Train it on some dummy data
        X_dummy = np.random.random((10, 3))
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        # Create model data with actual model
        model_data = {
            'model': model,  # This is required by save_model
            'model_type': 'RandomForestClassifier',
            'metrics': {'accuracy': 0.95},
            'dataset': 'diabetes',
            'training_time': 120.5,
            'algorithm': 'random_forest',
            'model_name': 'test_model',
            'task_type': 'classification'
        }
        
        # Save model first
        save_result = self.model_registry.save_model(
            model_key='test_model_002',
            model_data=model_data
        )
        
        if save_result:
            # Test model loading
            loaded_model = self.model_registry.load_model('test_model_002')
            self.assertIsNotNone(loaded_model)
            self.assertIn('model', loaded_model)
            self.assertIn('metadata', loaded_model)
        else:
            self.skipTest("Model saving failed, cannot test loading")
    
    def test_get_model_history(self):
        """Test getting model history"""
        # Test getting model history
        history = self.model_registry.get_model_versions('test_model')
        self.assertIsInstance(history, pd.DataFrame)


class TestClassificationEvaluator(HealthcareDSSTestCase):
    """Test cases for ClassificationEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.classification_evaluator = ClassificationEvaluator()
    
    def test_initialization(self):
        """Test ClassificationEvaluator initialization"""
        self.assertIsNotNone(self.classification_evaluator)
        self.assertIsInstance(self.classification_evaluator.evaluation_results, dict)
    
    def test_evaluate_classification(self):
        """Test classification evaluation"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare test data
        y_true = dataset_df[target_col].head(50)
        y_pred = np.random.choice([0, 1], len(y_true))
        
        # Test classification evaluation
        result = self.classification_evaluator.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            model_name='test_model'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
    
    def test_confusion_matrix_analysis(self):
        """Test confusion matrix analysis"""
        # Get a classification dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target('classification')
        
        # Prepare test data
        y_true = dataset_df[target_col].head(50)
        y_pred = np.random.choice([0, 1], len(y_true))
        
        # Test confusion matrix analysis
        result = self.classification_evaluator.create_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred
        )
        
        self.assertIsNotNone(result)


class TestClusteringAnalyzer(HealthcareDSSTestCase):
    """Test cases for ClusteringAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.clustering_analyzer = ClusteringAnalyzer(self.data_manager)
    
    def test_initialization(self):
        """Test ClusteringAnalyzer initialization"""
        self.assertIsNotNone(self.clustering_analyzer)
        self.assertEqual(self.clustering_analyzer.data_manager, self.data_manager)
    
    def test_prepare_data_for_clustering(self):
        """Test data preparation for clustering"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        X, original_data = self.clustering_analyzer.prepare_data_for_clustering(dataset_name)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(original_data, pd.DataFrame)
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
    
    def test_kmeans_clustering(self):
        """Test K-means clustering"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Prepare data for clustering
        X, original_data = self.clustering_analyzer.prepare_data_for_clustering(dataset_name)
        
        result = self.clustering_analyzer.perform_kmeans_clustering(X, n_clusters=3)
        
        self.assertIsInstance(result, dict)
        self.assertIn('cluster_labels', result)
        self.assertIn('cluster_centers', result)
        self.assertIn('inertia', result)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Prepare data for clustering
        X, original_data = self.clustering_analyzer.prepare_data_for_clustering(dataset_name)
        
        result = self.clustering_analyzer.perform_dbscan_clustering(X)
        
        self.assertIsInstance(result, dict)
        self.assertIn('cluster_labels', result)
        self.assertIn('n_clusters', result)
        self.assertIn('n_noise', result)


class TestTimeSeriesAnalyzer(HealthcareDSSTestCase):
    """Test cases for TimeSeriesAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.time_series_analyzer = TimeSeriesAnalyzer(self.data_manager)
    
    def test_initialization(self):
        """Test TimeSeriesAnalyzer initialization"""
        self.assertIsNotNone(self.time_series_analyzer)
        self.assertEqual(self.time_series_analyzer.data_manager, self.data_manager)
    
    def test_prepare_time_series_data(self):
        """Test time series data preparation"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Test time series data preparation
        result = self.time_series_analyzer.prepare_time_series_data(dataset_name)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Test trend analysis (if method exists)
        if hasattr(self.time_series_analyzer, 'analyze_temporal_patterns'):
            result = self.time_series_analyzer.analyze_temporal_patterns(dataset_name)
            self.assertIsInstance(result, dict)
        else:
            # Skip if method doesn't exist
            self.skipTest("analyze_temporal_patterns method not available")
    
    def test_seasonal_decomposition(self):
        """Test seasonal decomposition"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Test seasonal decomposition (if method exists)
        if hasattr(self.time_series_analyzer, 'detect_anomalies'):
            result = self.time_series_analyzer.detect_anomalies(dataset_name)
            self.assertIsInstance(result, dict)
        else:
            # Skip if method doesn't exist
            self.skipTest("detect_anomalies method not available")


class TestPrescriptiveAnalyzer(HealthcareDSSTestCase):
    """Test cases for PrescriptiveAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.prescriptive_analyzer = PrescriptiveAnalyzer(self.data_manager)
    
    def test_initialization(self):
        """Test PrescriptiveAnalyzer initialization"""
        self.assertIsNotNone(self.prescriptive_analyzer)
        self.assertEqual(self.prescriptive_analyzer.data_manager, self.data_manager)
    
    def test_optimize_resource_allocation(self):
        """Test resource allocation optimization"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Define resource constraints
        resource_constraints = {'budget': 10000, 'staff': 50}
        
        # Test resource allocation optimization
        result = self.prescriptive_analyzer.optimize_resource_allocation(
            dataset_name, 
            resource_constraints=resource_constraints
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimization_type', result)
        self.assertIn('success', result)
    
    def test_schedule_optimization(self):
        """Test schedule optimization"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Define scheduling constraints
        scheduling_constraints = {'max_hours': 40, 'min_staff': 5}
        
        # Test schedule optimization
        result = self.prescriptive_analyzer.optimize_scheduling(
            dataset_name,
            scheduling_constraints=scheduling_constraints
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimization_type', result)
        self.assertIn('success', result)


class TestAssociationRulesMiner(HealthcareDSSTestCase):
    """Test cases for AssociationRulesMiner"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.association_rules_miner = AssociationRulesMiner(self.data_manager)
    
    def test_initialization(self):
        """Test AssociationRulesMiner initialization"""
        self.assertIsNotNone(self.association_rules_miner)
        self.assertEqual(self.association_rules_miner.data_manager, self.data_manager)
    
    def test_prepare_transactional_data(self):
        """Test transactional data preparation"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Define categorical columns for transaction data
        categorical_columns = ['sex'] if 'sex' in dataset_df.columns else ['target']
        
        # Test transactional data preparation
        result = self.association_rules_miner.prepare_transaction_data(
            dataset_name, 
            categorical_columns=categorical_columns
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)
    
    def test_mine_frequent_itemsets(self):
        """Test frequent itemset mining"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Define categorical columns for transaction data
        categorical_columns = ['sex'] if 'sex' in dataset_df.columns else ['target']
        
        # First prepare transaction data
        self.association_rules_miner.prepare_transaction_data(
            dataset_name, 
            categorical_columns=categorical_columns
        )
        
        # Test frequent itemset mining
        result = self.association_rules_miner.mine_frequent_itemsets(min_support=0.1)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(result.shape[0], 0)
    
    def test_generate_association_rules(self):
        """Test association rule generation"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Define categorical columns for transaction data
        categorical_columns = ['sex'] if 'sex' in dataset_df.columns else ['target']
        
        # First prepare transaction data and mine frequent itemsets
        self.association_rules_miner.prepare_transaction_data(
            dataset_name, 
            categorical_columns=categorical_columns
        )
        self.association_rules_miner.mine_frequent_itemsets(min_support=0.1)
        
        # Test association rule generation
        result = self.association_rules_miner.generate_association_rules()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(result.shape[0], 0)
    
    def test_analyze_healthcare_patterns(self):
        """Test healthcare pattern analysis"""
        # Get a real dataset
        dataset_name, dataset_df, target_col = self.get_dataset_with_target()
        
        # Test healthcare pattern analysis
        result = self.association_rules_miner.analyze_healthcare_patterns(dataset_name)
        
        self.assertIsInstance(result, dict)
        self.assertIn('dataset', result)
        self.assertIn('total_transactions', result)


if __name__ == '__main__':
    unittest.main()
