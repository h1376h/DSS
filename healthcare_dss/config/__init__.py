"""
Configuration Management
========================

This module contains configuration and settings management:
- Application settings
- Database configurations
- Model configurations
- Logging configurations
- Error handling
"""

import os
from pathlib import Path
from healthcare_dss.config.logging_config import setup_logging, get_logger, LoggerMixin
from healthcare_dss.config.exceptions import (
    HealthcareDSSError, DataManagementError, ModelManagementError,
    KnowledgeManagementError, AnalyticsError, UIError, ConfigurationError,
    ValidationError, DatabaseError, ModelNotFoundError, DataNotFoundError,
    InvalidDataFormatError, InsufficientDataError, ModelTrainingError,
    ModelEvaluationError, KnowledgeBaseError, ClinicalRuleError,
    DashboardError, StreamlitError, handle_exception, validate_data_format,
    safe_execute
)

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Database paths
DATABASE_PATH = PROJECT_ROOT / "healthcare_dss.db"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base.db"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "model_registry.db"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model configuration
DEFAULT_MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "scaling_method": "standard",
    "encoding_method": "onehot"
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "page_title": "Healthcare DSS",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)

def get_config():
    """Get application configuration"""
    return {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "logs_dir": LOGS_DIR,
        "config_dir": CONFIG_DIR,
        "database_path": DATABASE_PATH,
        "knowledge_base_path": KNOWLEDGE_BASE_PATH,
        "model_registry_path": MODEL_REGISTRY_PATH,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "default_model_params": DEFAULT_MODEL_PARAMS,
        "data_processing_config": DATA_PROCESSING_CONFIG,
        "dashboard_config": DASHBOARD_CONFIG
    }

# Export all configuration components
__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "LOGS_DIR", "CONFIG_DIR",
    "DATABASE_PATH", "KNOWLEDGE_BASE_PATH", "MODEL_REGISTRY_PATH",
    "LOG_LEVEL", "LOG_FORMAT", "DEFAULT_MODEL_PARAMS", "DATA_PROCESSING_CONFIG",
    "DASHBOARD_CONFIG", "ensure_directories", "get_config",
    "setup_logging", "get_logger", "LoggerMixin",
    "HealthcareDSSError", "DataManagementError", "ModelManagementError",
    "KnowledgeManagementError", "AnalyticsError", "UIError", "ConfigurationError",
    "ValidationError", "DatabaseError", "ModelNotFoundError", "DataNotFoundError",
    "InvalidDataFormatError", "InsufficientDataError", "ModelTrainingError",
    "ModelEvaluationError", "KnowledgeBaseError", "ClinicalRuleError",
    "DashboardError", "StreamlitError", "handle_exception", "validate_data_format",
    "safe_execute"
]
