"""
Error Handling and Exceptions
=============================

Custom exceptions and error handling utilities for the Healthcare DSS system.
"""

class HealthcareDSSError(Exception):
    """Base exception class for Healthcare DSS"""
    pass

class DataManagementError(HealthcareDSSError):
    """Exception raised for data management related errors"""
    pass

class ModelManagementError(HealthcareDSSError):
    """Exception raised for model management related errors"""
    pass

class KnowledgeManagementError(HealthcareDSSError):
    """Exception raised for knowledge management related errors"""
    pass

class AnalyticsError(HealthcareDSSError):
    """Exception raised for analytics related errors"""
    pass

class UIError(HealthcareDSSError):
    """Exception raised for user interface related errors"""
    pass

class ConfigurationError(HealthcareDSSError):
    """Exception raised for configuration related errors"""
    pass

class ValidationError(HealthcareDSSError):
    """Exception raised for data validation errors"""
    pass

class DatabaseError(HealthcareDSSError):
    """Exception raised for database related errors"""
    pass

class ModelNotFoundError(ModelManagementError):
    """Exception raised when a model is not found"""
    pass

class DataNotFoundError(DataManagementError):
    """Exception raised when data is not found"""
    pass

class InvalidDataFormatError(DataManagementError):
    """Exception raised when data format is invalid"""
    pass

class InsufficientDataError(DataManagementError):
    """Exception raised when there's insufficient data for processing"""
    pass

class ModelTrainingError(ModelManagementError):
    """Exception raised when model training fails"""
    pass

class ModelEvaluationError(ModelManagementError):
    """Exception raised when model evaluation fails"""
    pass

class KnowledgeBaseError(KnowledgeManagementError):
    """Exception raised for knowledge base related errors"""
    pass

class ClinicalRuleError(KnowledgeManagementError):
    """Exception raised for clinical rule related errors"""
    pass

class DashboardError(UIError):
    """Exception raised for dashboard related errors"""
    pass

class StreamlitError(UIError):
    """Exception raised for Streamlit related errors"""
    pass

def handle_exception(func):
    """
    Decorator to handle exceptions and log them appropriately
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    import functools
    import logging
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HealthcareDSSError as e:
            logging.error(f"Healthcare DSS Error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise HealthcareDSSError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    
    return wrapper

def validate_data_format(data, required_fields=None, data_type=None):
    """
    Validate data format and structure
    
    Args:
        data: Data to validate
        required_fields: List of required fields
        data_type: Expected data type
        
    Raises:
        InvalidDataFormatError: If data format is invalid
        ValidationError: If validation fails
    """
    if data is None:
        raise InvalidDataFormatError("Data cannot be None")
    
    if data_type and not isinstance(data, data_type):
        raise InvalidDataFormatError(f"Expected {data_type}, got {type(data)}")
    
    if required_fields:
        if isinstance(data, dict):
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {missing_fields}")
        elif hasattr(data, '__iter__') and not isinstance(data, str):
            # For list-like data structures
            if len(data) == 0:
                raise ValidationError("Data cannot be empty")
        else:
            raise InvalidDataFormatError("Data must be a dictionary or iterable for field validation")

def safe_execute(func, *args, default_return=None, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Default return value if function fails
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if execution fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        import logging
        logging.warning(f"Safe execution failed for {func.__name__}: {str(e)}")
        return default_return
