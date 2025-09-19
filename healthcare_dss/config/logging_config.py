"""
Logging Configuration
=====================

Centralized logging configuration for the Healthcare DSS system.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import os

def setup_logging(log_level="INFO", log_dir="logs"):
    """
    Setup centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                log_path / "healthcare_dss.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Create specific loggers for different components
    loggers = {
        'data_management': logging.getLogger('healthcare_dss.core.data_management'),
        'model_management': logging.getLogger('healthcare_dss.core.model_management'),
        'knowledge_management': logging.getLogger('healthcare_dss.core.knowledge_management'),
        'analytics': logging.getLogger('healthcare_dss.analytics'),
        'ui': logging.getLogger('healthcare_dss.ui'),
        'utils': logging.getLogger('healthcare_dss.utils')
    }
    
    # Configure component-specific loggers
    for name, logger in loggers.items():
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add file handler for each component
        handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(handler)
    
    return loggers

def get_logger(component_name):
    """
    Get logger for a specific component
    
    Args:
        component_name: Name of the component
        
    Returns:
        Logger instance for the component
    """
    return logging.getLogger(f'healthcare_dss.{component_name}')

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
    
    def log_info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
