"""
Comprehensive Integration Test Suite
===================================

This module contains integration tests that test the interaction between
different components of the Healthcare DSS system.
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

from healthcare_dss.core.data_management import DataManager
from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.knowledge_management import KnowledgeManager
from healthcare_dss.ui.kpi_dashboard import KPIDashboard
from healthcare_dss.ui.user_interface import DashboardManager
from healthcare_dss.analytics import (
    ModelTrainingEngine, ModelEvaluationEngine, ModelRegistry,
    ClassificationEvaluator, ClusteringAnalyzer, TimeSeriesAnalyzer,
    PrescriptiveAnalyzer, AssociationRulesMiner
)
from healthcare_dss.utils.crisp_dm_workflow import CRISPDMWorkflow
from healthcare_dss.utils.debug_manager import debug_manager
from tests.test_base import HealthcareDSSTestCase


class TestSystemIntegration(HealthcareDSSTestCase):
    """Integration tests for the complete Healthcare DSS system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        super().setUp()
    
    def test_data_manager_integration(self):
        """Test DataManager integration with real data"""
        # Use the data manager from the base class
        dm = self.data_manager
        
        # Test that datasets are loaded automatically
        self.assertGreater(len(dm.datasets), 0)
        
        # Test data quality assessment
        for dataset_name in list(dm.datasets.keys())[:3]:  # Test first 3 datasets
            quality = dm.assess_data_quality(dataset_name)
            self.assertIsInstance(quality, dict)
            self.assertIn('completeness_score', quality)
        
        # Test dataset info
        for dataset_name in list(dm.datasets.keys())[:3]:
            info = dm.get_dataset_info(dataset_name)
            self.assertIsInstance(info, dict)
            self.assertIn('shape', info)
    
    def test_model_manager_integration(self):
        """Test ModelManager integration"""
        # Use the model manager from the base class
        mm = self.model_manager
        
        # Test AI technology selection matrix
        matrix = mm.get_ai_technology_selection_matrix()
        self.assertIsInstance(matrix, dict)
        self.assertIn('healthcare_problems', matrix)
    
    def test_knowledge_manager_integration(self):
        """Test KnowledgeManager integration"""
        # Use the knowledge manager from the base class
        km = self.knowledge_manager
        
        # Test knowledge base operations
        self.assertIsNotNone(km)
        
        # Test knowledge base operations
        self.assertIsNotNone(km.db_manager)
        self.assertIsNotNone(km.rule_engine)
        self.assertIsNotNone(km.guidelines_manager)
        self.assertIsNotNone(km.decision_engine)
    
    def test_kpi_dashboard_integration(self):
        """Test KPI Dashboard integration"""
        # Create KPI dashboard with base class data manager
        kpi_dashboard = KPIDashboard(self.data_manager)
        
        # Test KPI dashboard initialization
        self.assertIsNotNone(kpi_dashboard)
        
        # Test KPI calculation
        kpis = kpi_dashboard.calculate_healthcare_kpis()
        self.assertIsInstance(kpis, dict)
        self.assertIn('system_datasets_loaded', kpis)
        
        # Test report generation
        report = kpi_dashboard.generate_kpi_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_dashboard_manager_integration(self):
        """Test DashboardManager integration"""
        # Create dashboard manager with base class components
        dashboard_manager = DashboardManager(self.data_manager, self.model_manager, self.knowledge_manager)
        
        # Test dashboard manager initialization
        self.assertIsNotNone(dashboard_manager)
        
        # Test dashboard configurations
        self.assertTrue(hasattr(dashboard_manager, 'dashboard_configs'))
        self.assertIsInstance(dashboard_manager.dashboard_configs, dict)
    
    def test_analytics_integration(self):
        """Test Analytics modules integration"""
        # Create analytics components
        evaluator = ClassificationEvaluator()
        clustering_analyzer = ClusteringAnalyzer(self.data_manager)
        
        # Test Classification Evaluator initialization
        self.assertIsNotNone(evaluator)
        self.assertIsNotNone(clustering_analyzer)
        
        # Test that components have expected attributes
        self.assertTrue(hasattr(evaluator, 'evaluation_results'))
        self.assertTrue(hasattr(clustering_analyzer, 'data_manager'))
    
    def test_crisp_dm_workflow_integration(self):
        """Test CRISP-DM Workflow integration"""
        # Create CRISP-DM workflow with base class data manager
        workflow = CRISPDMWorkflow(self.data_manager)
        
        # Test workflow initialization
        self.assertIsNotNone(workflow)
        
        # Test full workflow execution
        result = workflow.execute_full_workflow(
            dataset_name='breast_cancer',
            target_column='target',
            business_objective='Predict breast cancer outcomes'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('business_understanding', result)
        self.assertIn('data_understanding', result)
        self.assertIn('data_preparation', result)
        self.assertIn('modeling', result)
        self.assertIn('evaluation', result)
        self.assertIn('deployment', result)
    
    def test_model_registry_integration(self):
        """Test Model Registry integration"""
        registry = ModelRegistry()
        
        # Test registry initialization
        self.assertIsNotNone(registry)
        
        # Test registry methods exist
        self.assertTrue(hasattr(registry, 'save_model'))
        self.assertTrue(hasattr(registry, 'load_model'))
        self.assertTrue(hasattr(registry, 'list_models'))
    
    def test_debug_manager_integration(self):
        """Test Debug Manager integration"""
        from healthcare_dss.utils.debug_manager import debug_manager
        
        # Test debug manager initialization
        self.assertIsNotNone(debug_manager)
        
        # Test performance metrics
        from healthcare_dss.utils.debug_manager import update_performance_metric
        update_performance_metric("integration_test", 1.0)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Use base class components
        dm = self.data_manager
        mm = self.model_manager
        km = self.knowledge_manager
        
        kpi_dashboard = KPIDashboard(dm)
        dashboard_manager = DashboardManager(dm, mm, km)
        
        # Test data management workflow
        self.assertGreater(len(dm.datasets), 0)
        
        # Test KPI calculation
        kpis = kpi_dashboard.calculate_healthcare_kpis()
        self.assertIsInstance(kpis, dict)
        
        # Test dashboard creation
        self.assertIsNotNone(dashboard_manager)
        self.assertTrue(hasattr(dashboard_manager, 'dashboard_configs'))
        
        # Test analytics workflow
        clustering_analyzer = ClusteringAnalyzer(dm)
        self.assertIsNotNone(clustering_analyzer)
        
        # Test CRISP-DM workflow
        workflow = CRISPDMWorkflow(dm)
        self.assertIsNotNone(workflow)
        
        # Test debug logging
        from healthcare_dss.utils.debug_manager import debug_manager
        self.assertIsNotNone(debug_manager)
        
        # Verify all components are working together
        self.assertIsNotNone(dm)
        self.assertIsNotNone(mm)
        self.assertIsNotNone(km)
        self.assertIsNotNone(kpi_dashboard)
        self.assertIsNotNone(dashboard_manager)


class TestPerformanceIntegration(HealthcareDSSTestCase):
    """Performance integration tests"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        super().setUp()
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        import time
        
        start_time = time.time()
        
        # Use base class data manager
        dm = self.data_manager
        
        load_time = time.time() - start_time
        
        # Test that loading completes in reasonable time
        self.assertLess(load_time, 10.0)  # Should load within 10 seconds
        
        # Test KPI calculation performance
        kpi_dashboard = KPIDashboard(dm)
        
        start_time = time.time()
        kpis = kpi_dashboard.calculate_healthcare_kpis()
        kpi_time = time.time() - start_time
        
        # Test that KPI calculation completes in reasonable time
        self.assertLess(kpi_time, 5.0)  # Should calculate within 5 seconds
        self.assertIsInstance(kpis, dict)


if __name__ == '__main__':
    unittest.main()
