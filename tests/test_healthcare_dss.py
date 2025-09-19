"""
Comprehensive Test Suite for Healthcare DSS
==========================================

This module contains comprehensive tests for all subsystems
of the Healthcare Decision Support System including the new
role-based dashboard system and CRISP-DM workflow.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import sqlite3
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.data_management import DataManager
from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.knowledge_management import KnowledgeManager
from healthcare_dss.core.knowledge_models import ClinicalRule, ClinicalGuideline, RuleType, SeverityLevel
from healthcare_dss.ui.user_interface import DashboardManager
from healthcare_dss.config.dashboard_config import config_manager
from healthcare_dss.ui.dashboards.base_dashboard import BaseDashboard
from healthcare_dss.ui.dashboards.clinical_dashboard import ClinicalDashboard
from healthcare_dss.ui.dashboards.executive_dashboard import ExecutiveDashboard
from healthcare_dss.ui.dashboards.financial_dashboard import FinancialDashboard
from healthcare_dss.ui.analytics.crisp_dm_workflow import CRISPDMWorkflow


class TestDataManagement(unittest.TestCase):
    """Test cases for Data Management Subsystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_manager_initialization(self):
        """Test DataManager initialization"""
        self.assertIsNotNone(self.data_manager)
        self.assertIsNotNone(self.data_manager.connection)
        self.assertIsInstance(self.data_manager.datasets, dict)
    
    def test_dataset_loading(self):
        """Test dataset loading functionality"""
        # Check if datasets are loaded
        self.assertGreater(len(self.data_manager.datasets), 0)
        
        # Check specific datasets
        expected_datasets = ['diabetes', 'breast_cancer', 'healthcare_expenditure']
        for dataset in expected_datasets:
            if dataset in self.data_manager.datasets:
                self.assertIsInstance(self.data_manager.datasets[dataset], pd.DataFrame)
                self.assertGreater(self.data_manager.datasets[dataset].shape[0], 0)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        if 'diabetes' in self.data_manager.datasets:
            quality_metrics = self.data_manager.assess_data_quality('diabetes')
            
            # Check required metrics
            required_metrics = ['shape', 'completeness_score', 'missing_values', 'outliers']
            for metric in required_metrics:
                self.assertIn(metric, quality_metrics)
            
            # Check data types
            self.assertIsInstance(quality_metrics['shape'], tuple)
            self.assertIsInstance(quality_metrics['completeness_score'], (int, float))
            self.assertIsInstance(quality_metrics['missing_values'], dict)
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        if 'diabetes' in self.data_manager.datasets:
            features, target = self.data_manager.preprocess_data('diabetes', 'target')
            
            # Check output types
            self.assertIsInstance(features, pd.DataFrame)
            self.assertIsInstance(target, pd.Series)
            
            # Check shapes
            self.assertGreater(features.shape[0], 0)
            self.assertGreater(features.shape[1], 0)
            self.assertEqual(features.shape[0], target.shape[0])
    
    def test_healthcare_expenditure_analysis(self):
        """Test healthcare expenditure analysis"""
        expenditure_analysis = self.data_manager.get_healthcare_expenditure_analysis()
        
        # Check required fields
        required_fields = ['countries', 'total_countries', 'years_covered', 'expenditure_trends']
        for field in required_fields:
            self.assertIn(field, expenditure_analysis)
        
        # Check data types
        self.assertIsInstance(expenditure_analysis['countries'], list)
        self.assertIsInstance(expenditure_analysis['total_countries'], int)
        self.assertIsInstance(expenditure_analysis['years_covered'], list)
        self.assertIsInstance(expenditure_analysis['expenditure_trends'], dict)


class TestModelManagement(unittest.TestCase):
    """Test cases for Model Management Subsystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        self.model_manager = ModelManager(self.data_manager, models_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization"""
        self.assertIsNotNone(self.model_manager)
        self.assertIsNotNone(self.model_manager.data_manager)
        self.assertIsInstance(self.model_manager.registry, object)
    
    def test_model_training(self):
        """Test model training functionality"""
        if 'diabetes' in self.data_manager.datasets:
            # Test training a simple model
            result = self.model_manager.train_model(
                dataset_name='diabetes',
                model_name='random_forest',
                task_type='regression',
                target_column='target'
            )
            
            # Check result structure
            required_fields = ['model_key', 'metrics', 'feature_importance', 'training_data']
            for field in required_fields:
                self.assertIn(field, result)
            
            # Check training_data structure
            training_data = result['training_data']
            self.assertIn('X_train', training_data)
            self.assertIn('X_test', training_data)
            self.assertIn('y_train', training_data)
            self.assertIn('y_test', training_data)
            
            # Calculate samples from training data
            training_samples = len(training_data['X_train'])
            test_samples = len(training_data['X_test'])
            
            # Check metrics
            metrics = result['metrics']
            self.assertIn('r2_score', metrics)
            self.assertIn('rmse', metrics)
            self.assertIn('mae', metrics)
            
            # Check that model was stored in registry
            models_df = self.model_manager.registry.list_models()
            self.assertIn(result['model_key'], models_df['model_key'].values)
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        if 'diabetes' in self.data_manager.datasets:
            # Train a model first
            result = self.model_manager.train_model(
                dataset_name='diabetes',
                model_name='random_forest',
                task_type='regression',
                target_column='target'
            )
            
            # Get the actual model key from training result
            model_key = result['model_key']
            
            # Get test data using the same preprocessing as training
            df = self.data_manager.datasets['diabetes'].copy()
            features, _ = self.model_manager.preprocessing_engine.preprocess_data(df, 'target')
            # Remove columns that are removed during training due to data leakage detection
            if 'progression' in features.columns:
                features = features.drop('progression', axis=1)
            test_features = features.head(1)
            
            # Make prediction using the actual model key
            prediction = self.model_manager.predict(model_key, test_features)
            
            # Check prediction structure
            required_fields = ['predictions', 'model_key']
            for field in required_fields:
                self.assertIn(field, prediction)
            
            # Check prediction values
            self.assertIsInstance(prediction['predictions'], list)
            self.assertGreater(len(prediction['predictions']), 0)
    
    def test_ensemble_model_creation(self):
        """Test ensemble model creation"""
        if 'diabetes' in self.data_manager.datasets:
            # Create ensemble model
            result = self.model_manager.create_ensemble_model(
                dataset_name='diabetes',
                models=['random_forest', 'xgboost', 'linear_regression', 'decision_tree'],
                task_type='regression',
                target_column='target'
            )
            
            # Check result structure
            required_fields = ['model_key', 'metrics', 'individual_models', 'training_data']
            for field in required_fields:
                self.assertIn(field, result)
            
            # Check training_data structure
            training_data = result['training_data']
            self.assertIn('X_train', training_data)
            self.assertIn('X_test', training_data)
            self.assertIn('y_train', training_data)
            self.assertIn('y_test', training_data)
            
            # Calculate samples from training data
            training_samples = len(training_data['X_train'])
            test_samples = len(training_data['X_test'])
            
            # Check that ensemble was stored
            self.assertIn(result['model_key'], self.model_manager.registry.list_models()['model_key'].values)


class TestKnowledgeManagement(unittest.TestCase):
    """Test cases for Knowledge-Based Management Subsystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        self.model_manager = ModelManager(self.data_manager, models_dir=self.temp_dir)
        self.knowledge_manager = KnowledgeManager(
            self.data_manager, 
            self.model_manager, 
            knowledge_db_path=os.path.join(self.temp_dir, "knowledge.db")
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        self.knowledge_manager.close_connection()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_knowledge_manager_initialization(self):
        """Test KnowledgeManager initialization"""
        self.assertIsNotNone(self.knowledge_manager)
        self.assertIsNotNone(self.knowledge_manager.db_manager.connection)
        self.assertIsInstance(self.knowledge_manager.clinical_rules, dict)
        self.assertIsInstance(self.knowledge_manager.clinical_guidelines, dict)
    
    def test_clinical_rule_creation(self):
        """Test clinical rule creation and evaluation"""
        # Create a test rule
        test_rule = ClinicalRule(
            rule_id="test_rule_001",
            name="Test High Blood Pressure Rule",
            description="Test rule for high blood pressure",
            rule_type=RuleType.ALERT,
            conditions={"systolic_bp": "> 140"},
            actions=["Alert healthcare provider"],
            severity=SeverityLevel.HIGH,
            evidence_level="A",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Add rule
        self.knowledge_manager.add_clinical_rule(test_rule)
        
        # Check rule was added
        self.assertIn("test_rule_001", self.knowledge_manager.clinical_rules)
        
        # Test rule evaluation
        patient_data = {"systolic_bp": 150}
        triggered_rules = self.knowledge_manager.evaluate_clinical_rules(patient_data)
        
        # Check if rule was triggered
        rule_triggered = any(rule['rule_id'] == 'test_rule_001' for rule in triggered_rules)
        self.assertTrue(rule_triggered)
    
    def test_clinical_guideline_creation(self):
        """Test clinical guideline creation"""
        # Create a test guideline
        test_guideline = ClinicalGuideline(
            guideline_id="test_guideline_001",
            title="Test Diabetes Guideline",
            description="Test guideline for diabetes management",
            category="diabetes",
            conditions=["age > 45", "bmi > 25"],
            recommendations=["Perform HbA1c test", "Lifestyle counseling"],
            evidence_level="A",
            source="Test Source",
            version="1.0",
            created_at=datetime.now()
        )
        
        # Add guideline
        self.knowledge_manager.add_clinical_guideline(test_guideline)
        
        # Check guideline was added
        self.assertIn("test_guideline_001", self.knowledge_manager.clinical_guidelines)
    
    def test_decision_tree_evaluation(self):
        """Test decision tree evaluation"""
        # Test diabetes risk assessment tree
        patient_data = {
            'age': 50,
            'bmi': 28,
            'family_history': True
        }
        
        result = self.knowledge_manager.evaluate_decision_tree("diabetes_risk_tree", patient_data)
        
        # Check result structure
        required_fields = ['tree_id', 'tree_name', 'outcome', 'recommendations', 'confidence']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check that we got a valid outcome
        self.assertIn(result['outcome'], ['low_risk', 'moderate_risk', 'high_risk'])
    
    def test_multi_criteria_decision_analysis(self):
        """Test multi-criteria decision analysis"""
        alternatives = [
            {'id': 'option_a', 'effectiveness': 0.8, 'cost': 0.3},
            {'id': 'option_b', 'effectiveness': 0.7, 'cost': 0.5},
            {'id': 'option_c', 'effectiveness': 0.9, 'cost': 0.8}
        ]
        
        criteria = [
            {'name': 'effectiveness', 'maximize': True},
            {'name': 'cost', 'maximize': False}
        ]
        
        weights = {'effectiveness': 0.7, 'cost': 0.3}
        
        result = self.knowledge_manager.multi_criteria_decision_analysis(
            alternatives, criteria, weights
        )
        
        # Check result structure
        required_fields = ['ranked_alternatives', 'scores', 'weights', 'recommendation']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check that we got a recommendation
        self.assertIsNotNone(result['recommendation'])
    
    def test_knowledge_search(self):
        """Test knowledge base search functionality"""
        # Search for diabetes-related knowledge
        results = self.knowledge_manager.search_knowledge("diabetes")
        
        # Check that we got results
        self.assertIsInstance(results, list)
        
        # Check result structure
        if results:
            for result in results:
                # Check for common fields that should be present
                self.assertIsInstance(result, dict)
                # At least one of these should be present
                has_guideline_fields = any(field in result for field in ['guideline_id', 'title', 'description'])
                has_rule_fields = any(field in result for field in ['rule_id', 'name'])
                self.assertTrue(has_guideline_fields or has_rule_fields, f"Result missing expected fields: {result}")


class TestUserInterface(unittest.TestCase):
    """Test cases for User Interface Subsystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        self.model_manager = ModelManager(self.data_manager, models_dir=self.temp_dir)
        self.knowledge_manager = KnowledgeManager(
            self.data_manager, 
            self.model_manager, 
            knowledge_db_path=os.path.join(self.temp_dir, "knowledge.db")
        )
        self.dashboard_manager = DashboardManager(
            self.data_manager, 
            self.model_manager, 
            self.knowledge_manager
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        self.knowledge_manager.close_connection()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dashboard_manager_initialization(self):
        """Test DashboardManager initialization"""
        self.assertIsNotNone(self.dashboard_manager)
        self.assertIsNotNone(self.dashboard_manager.data_manager)
        self.assertIsNotNone(self.dashboard_manager.model_manager)
        self.assertIsNotNone(self.dashboard_manager.knowledge_manager)
    
    def test_dashboard_configurations(self):
        """Test dashboard configurations"""
        configs = self.dashboard_manager.dashboard_configs
        
        # Check that we have configurations for different user roles
        expected_roles = ['clinical', 'administrative', 'executive']
        for role in expected_roles:
            self.assertIn(role, configs)
            self.assertIn('title', configs[role])
            self.assertIn('sections', configs[role])
    
    def test_mock_data_generation(self):
        """Test mock data generation methods"""
        # Test clinical alerts
        alerts = self.dashboard_manager._get_clinical_alerts()
        self.assertIsInstance(alerts, list)
        if alerts:
            self.assertIn('title', alerts[0])
            self.assertIn('severity', alerts[0])
        
        # Test patient satisfaction data
        satisfaction_data = self.dashboard_manager._get_patient_satisfaction_data()
        self.assertIsInstance(satisfaction_data, pd.DataFrame)
        self.assertIn('month', satisfaction_data.columns)
        self.assertIn('satisfaction_score', satisfaction_data.columns)
        
        # Test KPI data
        kpi_data = self.dashboard_manager._get_kpi_data()
        self.assertIsInstance(kpi_data, pd.DataFrame)
        self.assertIn('month', kpi_data.columns)
        self.assertIn('kpi', kpi_data.columns)
        self.assertIn('value', kpi_data.columns)


class TestDashboardConfiguration(unittest.TestCase):
    """Test cases for Dashboard Configuration System"""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        self.assertIsNotNone(config_manager)
        self.assertIsInstance(config_manager.system_config, dict)
        self.assertIsInstance(config_manager.roles, dict)
    
    def test_system_configuration(self):
        """Test system configuration retrieval"""
        system_config = config_manager.get_system_config()
        
        # Check required configuration fields
        required_fields = ['debug_mode', 'theme', 'language', 'timezone']
        for field in required_fields:
            self.assertIn(field, system_config)
        
        # Check data types
        self.assertIsInstance(system_config['debug_mode'], bool)
        self.assertIsInstance(system_config['theme'], str)
        self.assertIsInstance(system_config['language'], str)
        self.assertIsInstance(system_config['timezone'], str)
    
    def test_role_configuration(self):
        """Test role configuration retrieval"""
        all_roles = config_manager.get_all_roles()
        
        # Check that we have expected roles
        expected_roles = [
            "Clinical Leadership",
            "Administrative Executive", 
            "Financial Manager",
            "Department Manager",
            "Clinical Staff",
            "Data Analyst"
        ]
        
        for role in expected_roles:
            self.assertIn(role, all_roles)
        
        # Test individual role configuration
        for role in all_roles:
            role_config = config_manager.get_role_config(role)
            self.assertIsNotNone(role_config)
            self.assertIsInstance(role_config.pages, list)
            self.assertGreater(len(role_config.pages), 0)
    
    def test_page_configuration(self):
        """Test page configuration for different roles"""
        # Test Clinical Leadership pages
        clinical_config = config_manager.get_role_config("Clinical Leadership")
        expected_clinical_pages = [
            "Clinical Dashboard",
            "Patient Flow Management",
            "Quality & Safety Monitoring",
            "Resource Allocation Guidance",
            "Strategic Planning",
            "Performance Management",
            "Clinical Analytics",
            "Outcome Analysis",
            "Risk Assessment",
            "Compliance Monitoring"
        ]
        
        for page in expected_clinical_pages:
            self.assertIn(page, clinical_config.pages)
        
        # Test Executive pages
        executive_config = config_manager.get_role_config("Administrative Executive")
        expected_executive_pages = [
            "Executive Dashboard",
            "Regulatory Compliance",
            "Resource Planning",
            "KPI Dashboard",
            "Financial Overview",
            "Operational Analytics",
            "Risk Management",
            "Stakeholder Reports"
        ]
        
        for page in expected_executive_pages:
            self.assertIn(page, executive_config.pages)


class TestBaseDashboard(unittest.TestCase):
    """Test cases for Base Dashboard System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        
        # Mock session state
        import streamlit as st
        st.session_state.data_manager = self.data_manager
        st.session_state.debug_mode = False
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_base_dashboard_initialization(self):
        """Test BaseDashboard initialization"""
        dashboard = BaseDashboard("Test Dashboard")
        
        self.assertEqual(dashboard.dashboard_name, "Test Dashboard")
        self.assertIsNotNone(dashboard.data_manager)
        self.assertIsInstance(dashboard.debug_mode, bool)
    
    def test_metrics_calculation(self):
        """Test metrics calculation methods"""
        dashboard = BaseDashboard("Test Dashboard")
        
        # Test metrics calculation
        metrics = dashboard._calculate_metrics()
        
        # Check that metrics is a dictionary
        self.assertIsInstance(metrics, dict)
        # Note: BaseDashboard returns empty dict by default, which is expected
    
    def test_charts_data_generation(self):
        """Test charts data generation"""
        dashboard = BaseDashboard("Test Dashboard")
        
        # Test charts data
        charts_data = dashboard._get_charts_data()
        
        # Check that charts_data is a dictionary
        self.assertIsInstance(charts_data, dict)


class TestRoleBasedDashboards(unittest.TestCase):
    """Test cases for Role-Based Dashboard System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        
        # Mock session state
        import streamlit as st
        st.session_state.data_manager = self.data_manager
        st.session_state.debug_mode = False
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_clinical_dashboard(self):
        """Test Clinical Dashboard functionality"""
        dashboard = ClinicalDashboard()
        
        # Test metrics calculation
        metrics = dashboard._calculate_metrics()
        required_metrics = ['active_patients', 'quality_score', 'avg_wait_time', 'readmission_rate']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Test charts data
        charts_data = dashboard._get_charts_data()
        required_charts = ['patient_flow', 'department_performance', 'patient_distribution']
        for chart in required_charts:
            self.assertIn(chart, charts_data)
    
    def test_executive_dashboard(self):
        """Test Executive Dashboard functionality"""
        dashboard = ExecutiveDashboard()
        
        # Test metrics calculation
        metrics = dashboard._calculate_metrics()
        required_metrics = ['revenue', 'patient_satisfaction', 'operational_efficiency', 'market_share']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Test charts data
        charts_data = dashboard._get_charts_data()
        required_charts = ['revenue_trend', 'kpi_performance']
        for chart in required_charts:
            self.assertIn(chart, charts_data)
    
    def test_financial_dashboard(self):
        """Test Financial Dashboard functionality"""
        dashboard = FinancialDashboard()
        
        # Test metrics calculation
        metrics = dashboard._calculate_metrics()
        required_metrics = ['monthly_revenue', 'operating_costs', 'profit_margin', 'cash_flow']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Test charts data
        charts_data = dashboard._get_charts_data()
        required_charts = ['financial_performance', 'cost_breakdown']
        for chart in required_charts:
            self.assertIn(chart, charts_data)


class TestCRISPDMWorkflow(unittest.TestCase):
    """Test cases for CRISP-DM Workflow System"""
    
    def test_workflow_initialization(self):
        """Test CRISPDMWorkflow initialization"""
        workflow = CRISPDMWorkflow()
        
        # Check initial state
        self.assertEqual(workflow.current_phase, "Business Understanding")
        self.assertIsInstance(workflow.workflow_data, dict)
        
        # Check all phases are present
        expected_phases = [
            "Business Understanding",
            "Data Understanding", 
            "Data Preparation",
            "Modeling",
            "Evaluation",
            "Deployment"
        ]
        
        for phase in expected_phases:
            self.assertIn(phase, workflow.workflow_data)
            self.assertIn('status', workflow.workflow_data[phase])
            self.assertIn('progress', workflow.workflow_data[phase])
            self.assertIn('tasks', workflow.workflow_data[phase])
            self.assertIn('completed_tasks', workflow.workflow_data[phase])
            self.assertIn('artifacts', workflow.workflow_data[phase])
    
    def test_task_completion(self):
        """Test task completion functionality"""
        workflow = CRISPDMWorkflow()
        
        # Complete a task
        initial_progress = workflow.workflow_data["Business Understanding"]["progress"]
        workflow._complete_task("Business Understanding", "Define business objectives")
        
        # Check progress updated
        updated_progress = workflow.workflow_data["Business Understanding"]["progress"]
        self.assertGreater(updated_progress, initial_progress)
        
        # Check task added to completed tasks
        completed_tasks = workflow.workflow_data["Business Understanding"]["completed_tasks"]
        self.assertIn("Define business objectives", completed_tasks)
    
    def test_workflow_phases(self):
        """Test workflow phase management"""
        workflow = CRISPDMWorkflow()
        
        # Test phase progression
        phases = list(workflow.workflow_data.keys())
        
        # Complete all tasks in first phase
        for task in workflow.workflow_data["Business Understanding"]["tasks"]:
            workflow._complete_task("Business Understanding", task)
        
        # Check phase status updated
        self.assertEqual(workflow.workflow_data["Business Understanding"]["status"], "completed")
        self.assertEqual(workflow.workflow_data["Business Understanding"]["progress"], 100)


class TestSystemIntegration(unittest.TestCase):
    """Test cases for system integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir="datasets/raw", db_path=os.path.join(self.temp_dir, "test.db"))
        self.model_manager = ModelManager(self.data_manager, models_dir=self.temp_dir)
        self.knowledge_manager = KnowledgeManager(
            self.data_manager, 
            self.model_manager, 
            knowledge_db_path=os.path.join(self.temp_dir, "knowledge.db")
        )
        self.dashboard_manager = DashboardManager(
            self.data_manager, 
            self.model_manager, 
            self.knowledge_manager
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.data_manager.close_connection()
        self.knowledge_manager.close_connection()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        if 'diabetes' not in self.data_manager.datasets:
            self.skipTest("Diabetes dataset not available")
        
        # Step 1: Data preprocessing using the same path as model training
        df = self.data_manager.datasets['diabetes'].copy()
        features, target = self.model_manager.preprocessing_engine.preprocess_data(df, 'target')
        self.assertGreater(features.shape[0], 0)
        
        # Step 2: Model training
        model_result = self.model_manager.train_model(
            dataset_name='diabetes',
            model_name='random_forest',
            task_type='regression',
            target_column='target'
        )
        self.assertIn('model_key', model_result)
        
        # Step 3: Model prediction
        # Remove columns that are removed during training due to data leakage detection
        prediction_features = features.copy()
        if 'progression' in prediction_features.columns:
            prediction_features = prediction_features.drop('progression', axis=1)
        test_features = prediction_features.head(1)
        prediction = self.model_manager.predict(model_result['model_key'], test_features)
        self.assertIn('predictions', prediction)
        
        # Step 4: Knowledge-based recommendations
        patient_data = {
            'age': 45,
            'bmi': 28.5,
            'systolic_bp': 135,
            'diastolic_bp': 88,
            'family_history': True
        }
        
        recommendations = self.knowledge_manager.get_clinical_recommendations(patient_data)
        self.assertIsInstance(recommendations, list)
        
        # Step 5: Decision tree evaluation
        decision_result = self.knowledge_manager.evaluate_decision_tree("diabetes_risk_tree", patient_data)
        self.assertIn('outcome', decision_result)
        
        # All steps completed successfully
        self.assertTrue(True)
    
    def test_system_performance_metrics(self):
        """Test system performance metrics"""
        # Data management performance
        total_datasets = len(self.data_manager.datasets)
        self.assertGreater(total_datasets, 0)
        
        # Model management performance
        trained_models = len(self.model_manager.registry.list_models())
        self.assertGreaterEqual(trained_models, 0)
        
        # Knowledge management performance
        knowledge_summary = self.knowledge_manager.get_knowledge_summary()
        self.assertIn('clinical_rules', knowledge_summary)
        self.assertIn('clinical_guidelines', knowledge_summary)
        
        # All subsystems are operational
        self.assertTrue(True)
    
    def test_dashboard_integration(self):
        """Test dashboard integration with core systems"""
        # Test that dashboard manager can access all subsystems
        self.assertIsNotNone(self.dashboard_manager.data_manager)
        self.assertIsNotNone(self.dashboard_manager.model_manager)
        self.assertIsNotNone(self.dashboard_manager.knowledge_manager)
        
        # Test dashboard configurations
        configs = self.dashboard_manager.dashboard_configs
        self.assertIsInstance(configs, dict)
        self.assertGreater(len(configs), 0)
    
    def test_crisp_dm_integration(self):
        """Test CRISP-DM workflow integration"""
        workflow = CRISPDMWorkflow()
        
        # Test workflow can be initialized
        self.assertIsNotNone(workflow)
        self.assertEqual(workflow.current_phase, "Business Understanding")
        
        # Test workflow data structure
        self.assertIsInstance(workflow.workflow_data, dict)
        self.assertEqual(len(workflow.workflow_data), 6)  # 6 phases
        
        # Test task completion affects progress
        initial_progress = workflow.workflow_data["Business Understanding"]["progress"]
        workflow._complete_task("Business Understanding", "Define business objectives")
        updated_progress = workflow.workflow_data["Business Understanding"]["progress"]
        self.assertGreater(updated_progress, initial_progress)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataManagement,
        TestModelManagement,
        TestKnowledgeManagement,
        TestUserInterface,
        TestDashboardConfiguration,
        TestBaseDashboard,
        TestRoleBasedDashboards,
        TestCRISPDMWorkflow,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Healthcare DSS Test Suite...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
    
    print("=" * 50)
