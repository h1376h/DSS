"""
Healthcare Decision Support System (DSS)
========================================

A comprehensive implementation of healthcare DSS architecture with four main subsystems:
1. Data Management Subsystem
2. Model Management Subsystem  
3. Knowledge-Based Management Subsystem
4. User Interface Subsystem

This implementation uses real healthcare datasets for validation and testing.

Module Structure:
- core: Core DSS components (Data, Model, Knowledge Management)
- analytics: Machine learning and analytics components
- ui: User interface and dashboard components
- utils: Utility functions and helper components
- config: Configuration and settings management
"""

__version__ = "1.0.0"
__author__ = "Healthcare DSS Team"

# Core components
from healthcare_dss.core import DataManager, ModelManager, KnowledgeManager, PreprocessingEngine

# Analytics components
from healthcare_dss.analytics import (
    ModelTrainingEngine, ModelEvaluationEngine, ModelRegistry,
    ClassificationEvaluator, ClusteringAnalyzer, TimeSeriesAnalyzer,
    PrescriptiveAnalyzer, AssociationRulesMiner
)

# UI components
from healthcare_dss.ui import KPIDashboard, DashboardManager

# Utility components
from healthcare_dss.utils import debug_manager, debug_write, show_debug_info, log_database_query, log_model_training, update_performance_metric, CRISPDMWorkflow

# Configuration
from healthcare_dss.config import get_config, ensure_directories

__all__ = [
    # Core components
    "DataManager",
    "ModelManager", 
    "KnowledgeManager",
    "PreprocessingEngine",
    
    # Analytics components
    "ModelTrainingEngine",
    "ModelEvaluationEngine",
    "ModelRegistry",
    "ClassificationEvaluator",
    "ClusteringAnalyzer",
    "TimeSeriesAnalyzer",
    "PrescriptiveAnalyzer",
    "AssociationRulesMiner",
    
    # UI components
    "KPIDashboard",
    "DashboardManager",
    
    # Utility components
    "debug_manager",
    "debug_write",
    "show_debug_info", 
    "log_database_query",
    "log_model_training",
    "update_performance_metric",
    "CRISPDMWorkflow",
    
    # Configuration
    "get_config",
    "ensure_directories"
]
