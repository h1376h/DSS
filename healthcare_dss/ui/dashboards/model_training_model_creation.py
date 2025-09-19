"""
Model Training Model Creation Module
==================================

This module handles model creation and configuration:
- Dynamic model configuration based on available libraries
- Model instantiation with parameter validation
- Support for various ML algorithms (sklearn, XGBoost, LightGBM)
"""

from typing import Dict, Any
from healthcare_dss.utils.debug_manager import debug_manager


def get_model_configurations() -> Dict[str, Dict]:
    """Get dynamic model configurations based on available libraries and task type"""
    
    configurations = {}
    
    # Random Forest - Always available with sklearn
    configurations["Random Forest"] = {
        "classification": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "default_params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "param_ranges": {
                "n_estimators": (10, 500),
                "max_depth": (1, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            }
        },
        "regression": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "default_params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "param_ranges": {
                "n_estimators": (10, 500),
                "max_depth": (1, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            }
        }
    }
    
    # Linear Models
    configurations["Logistic Regression"] = {
        "classification": {
            "class": "sklearn.linear_model.LogisticRegression",
            "default_params": {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42
            },
            "param_ranges": {
                "C": (0.01, 10.0),
                "max_iter": (100, 2000)
            }
        }
    }
    
    configurations["Linear Regression"] = {
        "regression": {
            "class": "sklearn.linear_model.LinearRegression",
            "default_params": {},
            "param_ranges": {}
        }
    }
    
    configurations["Ridge Regression"] = {
        "regression": {
            "class": "sklearn.linear_model.Ridge",
            "default_params": {
                "alpha": 1.0,
                "random_state": 42
            },
            "param_ranges": {
                "alpha": (0.01, 10.0)
            }
        }
    }
    
    configurations["Lasso Regression"] = {
        "regression": {
            "class": "sklearn.linear_model.Lasso",
            "default_params": {
                "alpha": 1.0,
                "max_iter": 1000,
                "random_state": 42
            },
            "param_ranges": {
                "alpha": (0.01, 10.0),
                "max_iter": (100, 2000)
            }
        }
    }
    
    # SVM Models
    configurations["SVM"] = {
        "classification": {
            "class": "sklearn.svm.SVC",
            "default_params": {
                "C": 1.0,
                "kernel": "rbf",
                "random_state": 42
            },
            "param_ranges": {
                "C": (0.01, 10.0)
            }
        }
    }
    
    configurations["SVR"] = {
        "regression": {
            "class": "sklearn.svm.SVR",
            "default_params": {
                "C": 1.0,
                "kernel": "rbf"
            },
            "param_ranges": {
                "C": (0.01, 10.0)
            }
        }
    }
    
    # Decision Tree
    configurations["Decision Tree"] = {
        "classification": {
            "class": "sklearn.tree.DecisionTreeClassifier",
            "default_params": {
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "param_ranges": {
                "max_depth": (1, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            }
        },
        "regression": {
            "class": "sklearn.tree.DecisionTreeRegressor",
            "default_params": {
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "param_ranges": {
                "max_depth": (1, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            }
        }
    }
    
    # K-Nearest Neighbors
    configurations["K-Nearest Neighbors"] = {
        "classification": {
            "class": "sklearn.neighbors.KNeighborsClassifier",
            "default_params": {
                "n_neighbors": 5,
                "weights": "uniform"
            },
            "param_ranges": {
                "n_neighbors": (1, 20)
            }
        },
        "regression": {
            "class": "sklearn.neighbors.KNeighborsRegressor",
            "default_params": {
                "n_neighbors": 5,
                "weights": "uniform"
            },
            "param_ranges": {
                "n_neighbors": (1, 20)
            }
        }
    }
    
    # Naive Bayes
    configurations["Naive Bayes"] = {
        "classification": {
            "class": "sklearn.naive_bayes.GaussianNB",
            "default_params": {},
            "param_ranges": {}
        }
    }
    
    # Try to add XGBoost if available
    try:
        import xgboost as xgb
        configurations["XGBoost"] = {
            "classification": {
                "class": "xgboost.XGBClassifier",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "param_ranges": {
                    "n_estimators": (10, 500),
                    "max_depth": (1, 10),
                    "learning_rate": (0.01, 0.3)
                }
            },
            "regression": {
                "class": "xgboost.XGBRegressor",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "param_ranges": {
                    "n_estimators": (10, 500),
                    "max_depth": (1, 10),
                    "learning_rate": (0.01, 0.3)
                }
            }
        }
    except ImportError:
        debug_manager.log_debug("XGBoost not available")
    
    # Try to add LightGBM if available
    try:
        import lightgbm as lgb
        configurations["LightGBM"] = {
            "classification": {
                "class": "lightgbm.LGBMClassifier",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "param_ranges": {
                    "n_estimators": (10, 500),
                    "max_depth": (1, 10),
                    "learning_rate": (0.01, 0.3)
                }
            },
            "regression": {
                "class": "lightgbm.LGBMRegressor",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "param_ranges": {
                    "n_estimators": (10, 500),
                    "max_depth": (1, 10),
                    "learning_rate": (0.01, 0.3)
                }
            }
        }
    except ImportError:
        debug_manager.log_debug("LightGBM not available")
    
    return configurations


def create_model(model_type: str, parameters: Dict, task_type: str):
    """Create model instance based on type and parameters using dynamic configuration"""
    
    try:
        configurations = get_model_configurations()
        
        debug_manager.log_debug(f"Creating model: {model_type} for {task_type} task")
        debug_manager.log_debug(f"Available models: {list(configurations.keys())}")
        
        if model_type not in configurations:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(configurations.keys())}")
        
        if task_type not in configurations[model_type]:
            available_tasks = list(configurations[model_type].keys())
            raise ValueError(f"Model {model_type} not supported for {task_type} task. Available tasks: {available_tasks}")
        
        config = configurations[model_type][task_type]
        class_path = config["class"]
        
        debug_manager.log_debug(f"Using class: {class_path}")
        
        # Dynamic import and instantiation
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        debug_manager.log_debug(f"Model parameters: {parameters}")
        
        return model_class(**parameters)
        
    except Exception as e:
        debug_manager.log_debug(f"Error creating model {model_type} for {task_type}: {str(e)}")
        raise
