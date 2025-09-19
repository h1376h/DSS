"""
Model Registry for Healthcare DSS
================================

This module handles model storage, loading, versioning, and management
for the Healthcare Decision Support System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import sqlite3

# Configure logging
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry for storing, versioning, and managing trained models
    
    Provides comprehensive model lifecycle management including storage,
    loading, versioning, and metadata tracking.
    """
    
    def __init__(self, models_dir: str = "models", registry_db: str = "model_registry.db"):
        """
        Initialize the model registry
        
        Args:
            models_dir: Directory to store model files
            registry_db: Database file for model metadata
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.registry_db = registry_db
        self._init_registry_db()
        
        # In-memory model cache
        self.model_cache = {}
        
    def _init_registry_db(self):
        """Initialize the model registry database"""
        try:
            conn = sqlite3.connect(self.registry_db)
            cursor = conn.cursor()
            
            # Create models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_file_path TEXT NOT NULL,
                    metadata_file_path TEXT NOT NULL,
                    performance_metrics TEXT,
                    feature_names TEXT,
                    preprocessing_config TEXT,
                    model_version TEXT DEFAULT '1.0',
                    status TEXT DEFAULT 'active',
                    description TEXT
                )
            ''')
            
            # Create model_versions table for versioning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_key TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_file_path TEXT NOT NULL,
                    metadata_file_path TEXT NOT NULL,
                    performance_metrics TEXT,
                    change_description TEXT,
                    FOREIGN KEY (model_key) REFERENCES models (model_key)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Model registry database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model registry database: {e}")
            raise
    
    def save_model(self, model_key: str, model_data: Dict[str, Any]) -> bool:
        """
        Save a trained model to the registry
        
        Args:
            model_key: Unique identifier for the model
            model_data: Dictionary containing model and metadata
            
        Returns:
            Boolean indicating success
        """
        try:
            # Validate model data
            if not model_data or not isinstance(model_data, dict):
                logger.error("Invalid model_data: must be a non-empty dictionary")
                return False
            
            # Extract model and metadata
            model = model_data.get('model')
            if model is None:
                logger.error("Invalid model_data: 'model' key is required and cannot be None")
                return False
            
            metrics = model_data.get('metrics', {})
            training_data = model_data.get('training_data', {})
            preprocessing_config = model_data.get('preprocessing_config', {})
            
            # Validate model_key
            if not model_key or not isinstance(model_key, str) or model_key.strip() == '':
                logger.error("Invalid model_key: must be a non-empty string")
                return False
            
            # Create model file path
            model_file_path = self.models_dir / f"{model_key}_model.joblib"
            metadata_file_path = self.models_dir / f"{model_key}_metadata.json"
            
            # Save model
            joblib.dump(model, model_file_path)
            
            # Prepare metadata
            metadata = {
                'model_key': model_key,
                'model_name': model_data.get('model_name', 'unknown'),
                'task_type': model_data.get('task_type', 'unknown'),
                'created_at': model_data.get('timestamp', datetime.now().isoformat()),
                'performance_metrics': metrics,
                'preprocessing_config': preprocessing_config,
                'feature_names': training_data.get('X_train', pd.DataFrame()).columns.tolist() if hasattr(training_data.get('X_train', pd.DataFrame()), 'columns') else [],
                'training_samples': len(training_data.get('X_train', [])),
                'test_samples': len(training_data.get('X_test', [])),
                'model_file_path': str(model_file_path),
                'metadata_file_path': str(metadata_file_path)
            }
            
            # Save metadata
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save to database
            self._save_to_database(model_key, model_data, str(model_file_path), str(metadata_file_path))
            
            # Cache model in memory
            self.model_cache[model_key] = {
                'model': model,
                'metadata': metadata,
                'metrics': metrics,
                'feature_names': metadata.get('feature_names', []),
                'preprocessing_config': preprocessing_config,
                'loaded_at': datetime.now()
            }
            
            logger.info(f"Model {model_key} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_key}: {e}")
            return False
    
    def _save_to_database(self, model_key: str, model_data: Dict[str, Any], 
                         model_file_path: str, metadata_file_path: str):
        """Save model information to the registry database"""
        try:
            conn = sqlite3.connect(self.registry_db)
            cursor = conn.cursor()
            
            # Prepare data for database
            metrics_json = json.dumps(model_data.get('metrics', {}), default=str)
            preprocessing_json = json.dumps(model_data.get('preprocessing_config', {}), default=str)
            feature_names_json = json.dumps(model_data.get('training_data', {}).get('X_train', pd.DataFrame()).columns.tolist() if hasattr(model_data.get('training_data', {}).get('X_train', pd.DataFrame()), 'columns') else [], default=str)
            
            # Insert or update model record
            cursor.execute('''
                INSERT OR REPLACE INTO models (
                    model_key, model_name, task_type, dataset_name, target_column,
                    model_file_path, metadata_file_path, performance_metrics,
                    feature_names, preprocessing_config, model_version, status, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_key,
                model_data.get('model_name', 'unknown'),
                model_data.get('task_type', 'unknown'),
                model_data.get('dataset_name', 'unknown'),
                model_data.get('target_column', 'unknown'),
                model_file_path,
                metadata_file_path,
                metrics_json,
                feature_names_json,
                preprocessing_json,
                model_data.get('model_version', '1.0'),
                'active',
                model_data.get('description', '')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise
    
    def load_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Load a model from the registry
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            Dictionary containing model and metadata, or None if not found
        """
        try:
            # Check cache first
            if model_key in self.model_cache:
                logger.info(f"Model {model_key} loaded from cache")
                return self.model_cache[model_key]
            
            # Load from database
            conn = sqlite3.connect(self.registry_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_file_path, metadata_file_path, performance_metrics,
                       feature_names, preprocessing_config
                FROM models WHERE model_key = ? AND status = 'active'
            ''', (model_key,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                logger.warning(f"Model {model_key} not found in registry")
                return None
            
            model_file_path, metadata_file_path, metrics_json, feature_names_json, preprocessing_json = result
            
            # Load model
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found: {model_file_path}")
                return None
            
            model = joblib.load(model_file_path)
            
            # Load metadata
            metadata = {}
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r') as f:
                    metadata = json.load(f)
            
            # Parse JSON fields
            try:
                metrics = json.loads(metrics_json) if metrics_json else {}
                feature_names = json.loads(feature_names_json) if feature_names_json else []
                preprocessing_config = json.loads(preprocessing_json) if preprocessing_json else {}
            except:
                metrics = {}
                feature_names = []
                preprocessing_config = {}
            
            # Cache model
            model_data = {
                'model': model,
                'metadata': metadata,
                'metrics': metrics,
                'feature_names': feature_names,
                'preprocessing_config': preprocessing_config,
                'loaded_at': datetime.now()
            }
            
            self.model_cache[model_key] = model_data
            
            logger.info(f"Model {model_key} loaded successfully")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {e}")
            return None
    
    def list_models(self, status: str = 'active') -> pd.DataFrame:
        """
        List all models in the registry
        
        Args:
            status: Filter by model status ('active', 'inactive', 'all')
            
        Returns:
            DataFrame with model information
        """
        try:
            conn = sqlite3.connect(self.registry_db)
            
            if status == 'all':
                query = '''
                    SELECT model_key, model_name, task_type, dataset_name, target_column,
                           created_at, updated_at, model_version, status, description
                    FROM models ORDER BY created_at DESC
                '''
                df = pd.read_sql_query(query, conn)
            else:
                query = '''
                    SELECT model_key, model_name, task_type, dataset_name, target_column,
                           created_at, updated_at, model_version, status, description
                    FROM models WHERE status = ? ORDER BY created_at DESC
                '''
                df = pd.read_sql_query(query, conn, params=(status,))
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return pd.DataFrame()
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary of all models
        
        Returns:
            DataFrame with model performance metrics
        """
        try:
            conn = sqlite3.connect(self.registry_db)
            
            query = '''
                SELECT model_key, model_name, task_type, performance_metrics,
                       created_at, model_version
                FROM models WHERE status = 'active'
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Parse performance metrics
            performance_data = []
            for _, row in df.iterrows():
                try:
                    metrics = json.loads(row['performance_metrics']) if row['performance_metrics'] else {}
                    
                    performance_data.append({
                        'Model Key': row['model_key'],
                        'Model Name': row['model_name'],
                        'Task Type': row['task_type'],
                        'Accuracy': metrics.get('accuracy', 0),
                        'R² Score': metrics.get('r2_score', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1 Score': metrics.get('f1_score', 0),
                        'Created At': row['created_at'],
                        'Version': row['model_version']
                    })
                except:
                    continue
            
            return pd.DataFrame(performance_data)
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return pd.DataFrame()
    
    def delete_model(self, model_key: str) -> bool:
        """
        Delete a model from the registry
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get model file paths
            conn = sqlite3.connect(self.registry_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_file_path, metadata_file_path FROM models WHERE model_key = ?
            ''', (model_key,))
            
            result = cursor.fetchone()
            
            if result:
                model_file_path, metadata_file_path = result
                
                # Delete files
                if os.path.exists(model_file_path):
                    os.remove(model_file_path)
                if os.path.exists(metadata_file_path):
                    os.remove(metadata_file_path)
                
                # Update database status
                cursor.execute('''
                    UPDATE models SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                    WHERE model_key = ?
                ''', (model_key,))
                
                conn.commit()
                conn.close()
                
                # Remove from cache
                if model_key in self.model_cache:
                    del self.model_cache[model_key]
                
                logger.info(f"Model {model_key} deleted successfully")
                return True
            else:
                conn.close()
                logger.warning(f"Model {model_key} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting model {model_key}: {e}")
            return False
    
    def update_model_status(self, model_key: str, status: str) -> bool:
        """
        Update model status
        
        Args:
            model_key: Unique identifier for the model
            status: New status ('active', 'inactive', 'deprecated')
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = sqlite3.connect(self.registry_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE models SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE model_key = ?
            ''', (status, model_key))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model {model_key} status updated to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            return False
    
    def get_model_versions(self, model_key: str) -> pd.DataFrame:
        """
        Get version history for a model
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            DataFrame with version history
        """
        try:
            conn = sqlite3.connect(self.registry_db)
            
            query = '''
                SELECT version, created_at, performance_metrics, change_description
                FROM model_versions WHERE model_key = ? ORDER BY created_at DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(model_key,))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the in-memory model cache"""
        self.model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache"""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'cache_details': {
                key: {
                    'loaded_at': value['loaded_at'].isoformat(),
                    'model_name': value['metadata'].get('model_name', 'unknown'),
                    'task_type': value['metadata'].get('task_type', 'unknown')
                }
                for key, value in self.model_cache.items()
            }
        }
